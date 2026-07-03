"""Unified clean-from-scratch PIWM VAE — FAITHFUL reproduction of the baseline
(theta_branch + position-equivariance) in ONE run, with 3 documented deltas.

Reproduces (carried from baseline, see SPEC.md):
  arch    PiwmConvVAE + ThetaBranch (θ from a purple-centroid 24x24 crop)
  recon   weighted_mse + edge_weight*gradient_loss + ssim_weight*ssim_loss   <-- (I had dropped edge+ssim)
  state   xy_loss + theta_loss (state_weight)         kl   on scene latent z[4:]
  swap    per-axis z[0:2] swap -> concentration loss (equiv_weight)          <-- position controllability
  optim   Adam, grad_clip 0.5; baseline lr schedule 1e-3 (recon phase) -> 5e-4 (equiv phase)
  select  control-based on a VAL split: once z0-sweep >= ctrl_min, minimize HARD recon-pos  <-- anti-gaming
          (I had used val_θ, which doesn't catch the gameable swap)

Deltas (vs baseline):
  CHANGED  swap mask = soft_color × dominance (cream-free, decoded-robust) instead of soft_purple
  NEW      equiv_warmup (ramp equiv in) replacing the two-stage warm-start
  DROPPED  warm-start -> single from-scratch run (so cap epochs generously, let control-stop decide)

Monitoring/saving: per-epoch control/hard-recon/θ stream; vanish-check every `eval_every`;
BEST checkpoint written to disk on every improvement (kill-safe); LAST saved at the end.
Every hyperparameter is an explicit config field -> recorded in the manifest (no gaps).
"""
import glob
import os
import sys
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

import config
import checkpoints
import controllability

sys.path.insert(0, config.BASELINE_SRC)
sys.path.insert(0, config.BASELINE_SCRIPTS)
from piwm_model.autoencoder import PiwmConvVAE, kl_divergence  # noqa: E402
from piwm_model.data import lander_fully_visible  # noqa: E402
from piwm_model.sprite import purple_mask  # noqa: E402
# baseline recon + selection helpers (import, don't hand-roll)
from train_theta_branch_vae import (  # noqa: E402
    ThetaBranch, lander_weight_map, weighted_mse, gradient_loss, gaussian_window, ssim_loss)
from train_position_equiv import position_control_px, hard_recon_pos_px  # noqa: E402

CROP = 24


# ---------------------------------------------------------------- cream-free swap mask
def soft_color(img, c, tol, T=0.05):
    d = torch.sqrt(((img - c.view(1, 3, 1, 1)) ** 2).sum(1) + 1e-8)
    return torch.sigmoid((tol - d) / T)


def _dominance(img, bias=0.05, T=0.02):
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    return torch.sigmoid((b - r - bias) / T) * torch.sigmoid((b - g - bias) / T)


def lander_mask(img, c, tol, dom_bias):
    """soft_color × dominance: decoded-robust lander, rejects terrain/flame/edges (cream-free)."""
    return soft_color(img, c, tol) * _dominance(img, dom_bias)


def concentration_loss(mask, target):
    B, H, W = mask.shape
    ys = torch.arange(H, device=mask.device, dtype=mask.dtype).view(1, H, 1)
    xs = torch.arange(W, device=mask.device, dtype=mask.dtype).view(1, 1, W)
    d2 = (xs - target[:, 0].view(B, 1, 1)) ** 2 + (ys - target[:, 1].view(B, 1, 1)) ** 2
    return (mask * d2).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1e-6)


# ---------------------------------------------------------------- config (every knob explicit)
@dataclass
class CleanVAEConfig:
    # data
    train_files: int = 345
    val_frac: float = 0.15
    batch: int = 32
    latent_dim: int = 32
    # recon (baseline)
    lander_weight: float = 25.0
    edge_weight: float = 1.0
    ssim_weight: float = 0.5
    kl_weight: float = 1e-4
    state_weight: float = 1.0
    # swap (baseline: per-axis + concentration)
    equiv_weight: float = 5e-4
    recon_warmup: int = 0      # pure-recon epochs (equiv OFF) to establish a SHARP lander BEFORE equiv
    equiv_warmup: int = 5
    per_axis: bool = True
    equiv_detach_residual: bool = False
    # cream-free mask (CHANGED)
    lander_rgb: tuple = (0.361, 0.294, 0.642)
    color_tol: float = 0.6
    dom_bias: float = 0.05
    # optim (baseline lr schedule 1e-3 recon-phase -> 5e-4 equiv-phase)
    lr_warmup: float = 1e-3
    lr_main: float = 5e-4
    grad_clip: float = 0.5
    # selection: GATE on controllability (must render across x AND θ dials, control achieved, not gamed),
    # then OPTIMIZE crisp render (min val_recon) among the gated. Avoids both the v4 failure (sharp but
    # vanishing) and selecting on hard_recon noise. y is NOT gated (terrain floor varies → dialing y down
    # is legitimately off-manifold, would falsely read as vanishing).
    ctrl_min: float = 15.0
    vanish_min: float = 80.0   # gate: dial-x AND dial-θ render% must be >= this to ACCEPT a checkpoint
    hard_recon_max: float = 3.0   # gate: largest-component recon position error (px) — anti-gaming / accuracy
    snapshot_every: int = 4    # save a checkpoint every N epochs REGARDLESS of selection (full trajectory
                               # to inspect / debug the formula / never throw away good weights)
    patience: int = 8
    min_epochs: int = 17
    epochs: int = 70
    # monitoring / saving
    eval_every: int = 10
    save_name: str = "clean_vae_v3"
    seed: int = 0
    device: str = "cuda"
    verbose: bool = True


# ---------------------------------------------------------------- data
def _largest(mask):
    lab, n = ndimage.label(mask)
    return None if n == 0 else (lab == (1 + int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1))))))


def preload(data_dir, max_files, crop=CROP):
    """Visible frames -> imgs u8 (N,3,H,W), crops u8 (N,3,c,c), states, ptrue px (N,2), ep_id."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))[:max_files]
    imgs, crops, states, ptrue, ep = [], [], [], [], []
    for e, p in enumerate(files):
        with np.load(p) as d:
            I, S = d["imgs"], d["states"]
        for t in range(len(I)):
            if not lander_fully_visible(I[t]):
                continue
            comp = _largest(purple_mask(torch.from_numpy(I[t]).permute(2, 0, 1).float() / 255.).numpy())
            if comp is None:
                continue
            ys, xs = np.where(comp); cx, cy = xs.mean(), ys.mean()
            H, W = I[t].shape[0], I[t].shape[1]
            x0 = int(np.clip(round(cx - crop / 2), 0, W - crop)); y0 = int(np.clip(round(cy - crop / 2), 0, H - crop))
            imgs.append(I[t]); crops.append(I[t][y0:y0 + crop, x0:x0 + crop])
            states.append(S[t]); ptrue.append([cx, cy]); ep.append(e)
    return (torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous(),
            torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).contiguous(),
            torch.from_numpy(np.stack(states).astype(np.float32)),
            torch.tensor(ptrue, dtype=torch.float32), np.asarray(ep))


def _batches(imgs, crops, states, ptrue, idx, batch, device, shuffle, rng):
    order = idx[rng.permutation(len(idx))] if shuffle else idx
    for i in range(0, len(order), batch):
        b = torch.as_tensor(order[i:i + batch])
        yield (imgs[b].to(device).float() / 255., crops[b].to(device).float() / 255.,
               states[b].to(device), ptrue[b].to(device))


# ---------------------------------------------------------------- val metrics
@torch.no_grad()
def val_theta_recon(vae, branch, imgs, crops, states, idx, cfg, device, window, batch=64):
    vae.eval(); branch.eval(); th_err = []; rl = 0.0; n = 0
    for i in range(0, len(idx), batch):
        b = torch.as_tensor(idx[i:i + batch])
        frame = imgs[b].to(device).float() / 255.; crop = crops[b].to(device).float() / 255.; st = states[b].to(device)
        mu, logvar = vae.encode(frame); cossin = branch(crop)
        z = mu.clone(); z[:, 2:4] = cossin; recon = vae.decode(z)
        rl += float(weighted_mse(recon, frame, lander_weight_map(frame, cfg.lander_weight))) * frame.size(0); n += frame.size(0)
        thp = torch.atan2(cossin[:, 1], cossin[:, 0])
        d = torch.atan2(torch.sin(thp - st[:, config.TH]), torch.cos(thp - st[:, config.TH])).abs()
        th_err.append(torch.rad2deg(d).cpu())
    return float(torch.cat(th_err).median()), rl / max(n, 1)


@torch.no_grad()
def vanish_check(vae, branch, imgs, crops, states, idx, device, n_base=8):
    """Brief controllability monitor: % of bases rendering a lander when dialing x and θ."""
    vae.eval(); branch.eval()
    cen = [i for i in idx if 0.2 < float(states[i, config.Y]) < 0.9 and abs(float(states[i, config.X])) < 0.4][:n_base]
    if not cen:
        cen = list(idx[:n_base])
    F_ = imgs[cen].to(device).float() / 255.; C_ = crops[cen].to(device).float() / 255.
    mu, _ = vae.encode(F_); mu[:, 2:4] = branch(C_)
    def rate(setter, vals):
        fr = []
        for v in vals:
            z = mu.clone(); setter(z, v)
            dec = vae.decode(z)
            fr.append(np.mean([int(purple_mask(dec[i].cpu()).sum()) >= 8 for i in range(z.size(0))]))
        return round(100 * float(np.mean(fr)))
    import math
    return (rate(lambda z, v: z[:, 0].fill_(float(v)), controllability.X_RANGE),
            rate(lambda z, a: (z[:, 2].fill_(math.cos(a)), z[:, 3].fill_(math.sin(a))), controllability.ALPHA_RANGE))


def _flat_sd(vae, branch):
    sd = {f"vae.{k}": v for k, v in vae.state_dict().items()}
    sd.update({f"branch.{k}": v for k, v in branch.state_dict().items()})
    return sd


# ---------------------------------------------------------------- train
def train_clean_vae(cfg: CleanVAEConfig = None):
    import copy
    cfg = cfg or CleanVAEConfig()
    checkpoints.enable_determinism(cfg.seed)
    dev = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    rng = np.random.default_rng(cfg.seed)

    imgs, crops, states, ptrue, ep = preload(config.TRAIN_DIR, cfg.train_files)
    tr_mask, va_mask = checkpoints.episode_val_split(ep, cfg.val_frac, cfg.seed)
    tr_idx, va_idx = np.where(tr_mask)[0], np.where(va_mask)[0]
    # val subsets (uint8) for the baseline control/hard-recon selection metrics
    vF, vC, vP = imgs[va_idx], crops[va_idx], ptrue[va_idx]

    vae = PiwmConvVAE(latent_dim=cfg.latent_dim).to(dev)
    branch = ThetaBranch().to(dev)
    with torch.no_grad():
        branch(crops[:2].to(dev).float() / 255.)
    opt = torch.optim.Adam(list(vae.parameters()) + list(branch.parameters()), lr=cfg.lr_warmup)
    window = gaussian_window(3, device=dev)
    c = torch.tensor(cfg.lander_rgb, dtype=torch.float32, device=dev)

    hist = {k: [] for k in ("epoch", "lr", "equiv_w", "recon", "edge", "ssim", "xy", "theta", "equiv",
                            "val_theta_deg", "val_recon", "ctrl_x_px", "ctrl_y_px", "hard_recon_px", "sel",
                            "vanish_dialx", "vanish_dialth")}
    best = dict(metric=float("inf"), epoch=-1, state=None, since=0)       # PRIMARY: crispest among gated
    best_pos = dict(metric=float("inf"), epoch=-1, state=None)            # most accurate position among gated
    snapshots = []                                                        # periodic, regardless of selection
    diverged = False

    for epoch in range(cfg.epochs):
        e = epoch - cfg.recon_warmup                     # equiv OFF during recon_warmup, then ramp
        equiv_w = 0.0 if e < 0 else cfg.equiv_weight * min(1.0, e / max(1, cfg.equiv_warmup))
        lr = cfg.lr_warmup if equiv_w < cfg.equiv_weight else cfg.lr_main   # 1e-3 recon-phase -> 5e-4 equiv-phase
        for grp in opt.param_groups:
            grp["lr"] = lr
        vae.train(); branch.train()
        tot = {k: 0.0 for k in ("recon", "edge", "ssim", "xy", "theta", "equiv")}; n = 0
        for frame, crop, st, pt in _batches(imgs, crops, states, ptrue, tr_idx, cfg.batch, dev, True, rng):
            bn = frame.size(0)
            mu, logvar = vae.encode(frame)
            z = vae.reparameterize(mu, logvar); z = z.clone(); cossin = branch(crop); z[:, 2:4] = cossin
            recon = vae.decode(z)
            wmap = lander_weight_map(frame, cfg.lander_weight)
            mse = weighted_mse(recon, frame, wmap)
            edge = gradient_loss(recon, frame, wmap)
            ssim = ssim_loss(recon, frame, window)
            recon_l = mse + cfg.edge_weight * edge + cfg.ssim_weight * ssim
            xy_l = F.mse_loss(mu[:, 0:2], st[:, [config.X, config.Y]])
            th = st[:, config.TH]
            theta_l = F.mse_loss(cossin, torch.stack([torch.cos(th), torch.sin(th)], 1))
            kl_l = kl_divergence(mu[:, 4:], logvar[:, 4:])
            if equiv_w > 0 and bn > 1:
                def swap_loss(dims):
                    perm = torch.randperm(bn, device=dev)
                    zs = z.detach().clone() if cfg.equiv_detach_residual else z.clone()
                    target = pt.clone()
                    for d in dims:
                        zs[:, d] = z[perm, d]; target[:, d] = pt[perm, d]
                    return concentration_loss(lander_mask(vae.decode(zs), c, cfg.color_tol, cfg.dom_bias), target).mean()
                equiv_l = (swap_loss([0]) + swap_loss([1])) if cfg.per_axis else swap_loss([0, 1])
            else:
                equiv_l = torch.zeros((), device=dev)
            loss = recon_l + cfg.kl_weight * kl_l + cfg.state_weight * (xy_l + theta_l) + equiv_w * equiv_l
            if not torch.isfinite(loss):
                print(f"  DIVERGED epoch {epoch} — stopping, keeping best", flush=True); diverged = True; break
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(vae.parameters()) + list(branch.parameters()), cfg.grad_clip)
            opt.step()
            for k, v in [("recon", mse), ("edge", edge), ("ssim", ssim), ("xy", xy_l), ("theta", theta_l), ("equiv", equiv_l)]:
                tot[k] += float(v.detach()) * bn
            n += bn
        if diverged:
            break

        # ---- per-epoch VAL metrics: control (z0/z1 sweep px) + HARD recon-pos + θ + recon ----
        vth, vrec = val_theta_recon(vae, branch, imgs, crops, states, va_idx, cfg, dev, window)
        mx, my = position_control_px(vae, branch, vF, vC, dev)
        hpos = hard_recon_pos_px(vae, branch, vF, vC, vP, dev)
        vx, vt = vanish_check(vae, branch, imgs, crops, states, va_idx, dev)   # EVERY epoch (gate + monitor)
        # GATE: control achieved, renders across x AND θ dials, position accurate / not gamed.
        # Among gated checkpoints OPTIMIZE crisp render (val_recon); ungated are deferred above all gated.
        gated = (mx >= cfg.ctrl_min) and (vx >= cfg.vanish_min) and (vt >= cfg.vanish_min) and (hpos <= cfg.hard_recon_max)
        sel = vrec if gated else 1e4 + vrec
        for k, v in [("epoch", epoch), ("lr", lr), ("equiv_w", round(equiv_w, 6))]:
            hist[k].append(v)
        for k in ("recon", "edge", "ssim", "xy", "theta", "equiv"):
            hist[k].append(tot[k] / max(n, 1))
        hist["val_theta_deg"].append(vth); hist["val_recon"].append(vrec)
        hist["ctrl_x_px"].append(mx); hist["ctrl_y_px"].append(my); hist["hard_recon_px"].append(hpos)
        hist["sel"].append(sel); hist["vanish_dialx"].append(vx); hist["vanish_dialth"].append(vt)

        cand = lambda: copy.deepcopy({"vae": vae.state_dict(), "branch": branch.state_dict()})
        emet = dict(epoch=epoch, val_recon=vrec, val_theta_deg=vth, ctrl_x_px=mx, hard_recon_px=hpos,
                    vanish_dialx=vx, vanish_dialth=vt, gated=bool(gated))
        improved = sel < best["metric"] - 1e-6                      # PRIMARY = crispest among gated
        if improved:
            best.update(metric=sel, epoch=epoch, state=cand(), since=0)
            checkpoints.save_checkpoint(cfg.save_name + "_best", _flat_sd(vae, branch),
                                        metrics=emet, extra=dict(config=asdict(cfg)))   # KILL-SAFE
        else:
            best["since"] += 1
        if gated and hpos < best_pos["metric"]:                     # most accurate position among gated
            best_pos.update(metric=hpos, epoch=epoch, state=cand())
        if cfg.snapshot_every and (epoch % cfg.snapshot_every == 0):   # periodic, REGARDLESS of selection
            nm = f"{cfg.save_name}_ep{epoch}"
            checkpoints.save_checkpoint(nm, _flat_sd(vae, branch), metrics=emet, extra=dict(config=asdict(cfg)))
            snapshots.append(nm)

        ev = f"  vanish[dialx {vx:.0f}% dialθ {vt:.0f}%]"
        if cfg.verbose:
            print(f"  ep {epoch:3d}/{cfg.epochs} lr{lr:.0e} eqw{equiv_w:.1e} | recon {tot['recon']/n:.4f} "
                  f"edge {tot['edge']/n:.4f} ssim {tot['ssim']/n:.3f} equiv {tot['equiv']/n:.1f} | "
                  f"val_θ {vth:.2f}° ctrl_x {mx:.1f}px hardpos {hpos:.1f}px valrec {vrec:.4f} "
                  f"{'GATED' if gated else 'defer'} sel {sel:.4f}{'  *best' if improved else ''}{ev}", flush=True)
        if epoch + 1 >= cfg.min_epochs and best["since"] >= cfg.patience:
            if cfg.verbose:
                print(f"  early stop @ {epoch} (best sel {best['metric']:.1f}@{best['epoch']})", flush=True)
            break

    last_sd = copy.deepcopy(_flat_sd(vae, branch))     # LAST (deepcopy: load_state_dict(best) below
                                                       # copies INTO the live params in place, which would
                                                       # otherwise overwrite last_sd's references with best)
    def _flatten(state):
        sd = {f"vae.{k}": v for k, v in state["vae"].items()}
        sd.update({f"branch.{k}": v for k, v in state["branch"].items()}); return sd
    best_pos_sd = _flatten(best_pos["state"]) if best_pos["state"] is not None else None
    if best["state"] is not None:
        vae.load_state_dict(best["state"]["vae"]); branch.load_state_dict(best["state"]["branch"])
    vth, vrec = val_theta_recon(vae, branch, imgs, crops, states, va_idx, cfg, dev, window)
    mx, my = position_control_px(vae, branch, vF, vC, dev)
    hpos = hard_recon_pos_px(vae, branch, vF, vC, vP, dev)
    metrics = dict(val_theta_mae_deg=vth, val_recon=vrec, ctrl_x_px=mx, ctrl_y_px=my, hard_recon_px=hpos,
                   best_epoch=int(best["epoch"]), best_pos_epoch=int(best_pos["epoch"]),
                   epochs_ran=int(hist["epoch"][-1] + 1) if hist["epoch"] else 0,
                   stopped_early=bool(hist["epoch"] and hist["epoch"][-1] + 1 < cfg.epochs) or diverged,
                   diverged=diverged, n_train=int(len(tr_idx)), n_val=int(len(va_idx)), snapshots=snapshots)
    return dict(state_dict=_flat_sd(vae, branch), last_state_dict=last_sd, best_pos_state_dict=best_pos_sd,
                history=hist, metrics=metrics, snapshots=snapshots,
                extra=dict(config=asdict(cfg), arch="PiwmConvVAE+ThetaBranch",
                           latent_layout="z0=x z1=y z2:4=branch(crop)(cos,sin) z4:=scene latent"))


def plot_training_curves(history, out_path, label="", test_metrics=None, best_epoch=None):
    """PI-quality training graph: train-vs-val recon (overfitting), val θ, controllability (px
    moved per dial), HARD recon-pos (anti-gaming), loss components, vanish-rate; TEST shown as
    the held-out final result in the title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = history; ep = h["epoch"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    ax[0, 0].plot(ep, h["recon"], label="TRAIN recon"); ax[0, 0].plot(ep, h["val_recon"], label="VAL recon")
    ax[0, 0].set_yscale("log"); ax[0, 0].set_title("Reconstruction (train vs val) — overfit check")
    ax[0, 0].set_xlabel("epoch"); ax[0, 0].legend()
    ax[0, 1].plot(ep, h["val_theta_deg"], color="#1f77b4"); ax[0, 1].axhline(0.27, color="gray", ls="--", lw=0.8, label="baseline 0.27°")
    ax[0, 1].set_title("θ read-out (VAL, deg)"); ax[0, 1].set_xlabel("epoch"); ax[0, 1].set_ylabel("deg"); ax[0, 1].legend()
    ax[0, 2].plot(ep, h["ctrl_x_px"], label="ctrl x (z0 sweep)"); ax[0, 2].plot(ep, h["ctrl_y_px"], label="ctrl y (z1 sweep)")
    ax[0, 2].axhline(15, color="r", ls="--", lw=0.8, label="ctrl_min 15px")
    ax[0, 2].set_title("Controllability: lander px moved / dial (VAL)"); ax[0, 2].set_xlabel("epoch"); ax[0, 2].set_ylabel("px"); ax[0, 2].legend()
    ax[1, 0].plot(ep, h["hard_recon_px"], color="#2a2")
    ax[1, 0].set_title("HARD recon-position error (VAL) — anti-gaming"); ax[1, 0].set_xlabel("epoch"); ax[1, 0].set_ylabel("px")
    for k, lab in [("edge", "edge"), ("ssim", "ssim"), ("equiv", "equiv (px²)")]:
        ax[1, 1].plot(ep, h[k], label=lab)
    ax[1, 1].set_yscale("log"); ax[1, 1].set_title("Loss components (train)"); ax[1, 1].set_xlabel("epoch"); ax[1, 1].legend()
    vx = np.array(h["vanish_dialx"]); vt = np.array(h["vanish_dialth"]); epn = np.array(ep)
    mx_, mt_ = np.isfinite(vx), np.isfinite(vt)
    if mx_.any(): ax[1, 2].plot(epn[mx_], vx[mx_], "o-", label="dial-x render%")
    if mt_.any(): ax[1, 2].plot(epn[mt_], vt[mt_], "s-", label="dial-θ render%")
    ax[1, 2].set_ylim(0, 105); ax[1, 2].set_title("Vanish-rate (VAL)"); ax[1, 2].set_xlabel("epoch"); ax[1, 2].set_ylabel("% render"); ax[1, 2].legend()
    if best_epoch is not None and best_epoch >= 0:
        for a in ax.ravel():
            a.axvline(best_epoch, color="k", ls=":", lw=0.8)
    sub = label
    if test_metrics:
        keys = ("readout_theta_med_deg", "vanish_dialx_render_pct", "vanish_dialth_render_pct",
                "decoded_lander_mask_recall", "decoded_lander_color_err")
        sub += "   TEST(held-out): " + " | ".join(f"{k}={test_metrics[k]}" for k in keys if k in test_metrics)
    fig.suptitle(f"Training curves — {sub}  (vertical line = selected best epoch)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return out_path


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = (CleanVAEConfig(train_files=12, epochs=8, equiv_warmup=2, min_epochs=4, eval_every=3,
                          save_name="clean_vae_smoke", device=a.device)
           if a.smoke else CleanVAEConfig(device=a.device))
    t0 = time.time(); out = train_clean_vae(cfg); print(f"done in {time.time()-t0:.1f}s")
    print("metrics:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["metrics"].items()})
    print("payload keys:", list(out.keys()))
