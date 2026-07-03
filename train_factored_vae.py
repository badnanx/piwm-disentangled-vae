"""STAGE 3 of 3, the SHIPPED model, in the VAE's warm-start chain (run the whole chain with reproduce.sh;
see TRAINING.md). Warm-starts from stage 2 and factors the scene out of the lander.

Factored VAE (#2): the scene-latent encoder sees the LANDER-MASKED (scene-only) frame, so pose CANNOT
leak into the scene latent. The lander is drawn from INJECTED physical dims (x, y from labels/control; θ
from the crop-branch). Hypothesis: commanding a new pose then stays CLEAN (no haze), because the scene
latent no longer conflicts with the commanded lander position — fixing the pose-scene entanglement the
earlier target-pose experiments exposed (commanding a far position dragged scene content along).

Latent layout: z0=x (injected), z1=y (injected), z2:4=branch(crop)=(cosθ,sinθ), z4:=scene latent (from the
scene-only encoder). Decoder reconstructs the FULL frame. Controllability is by CONSTRUCTION (the only
lander-position information is the injected physical dims), so this DROPS the swap-equivariance /
gated-selection machinery of train_clean_vae — selection is just best val-recon + early stopping.

Success test (run after): the controllability sweep / pose demo -> far-position haze should
drop toward the realism floor, and the IN-RANGE (clean) region for x/y should widen.
"""
import argparse
import copy
import glob
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F

import config
import checkpoints
import controllability
import factored_data
import train_clean_vae as TC                       # reuse preload + CROP

sys.path.insert(0, config.BASELINE_SRC)
sys.path.insert(0, config.BASELINE_SCRIPTS)
from piwm_model.autoencoder import PiwmConvVAE, kl_divergence  # noqa: E402
from piwm_model.sprite import purple_mask  # noqa: E402
from train_theta_branch_vae import (  # noqa: E402
    ThetaBranch, lander_weight_map, weighted_mse, gradient_loss, gaussian_window, ssim_loss)
from controllability import centroid_px  # noqa: E402
from lander_app import LanderApp, derotate_crop, flat_sd_z, FeatNet, crop_at, feat_loss  # noqa: E402

CROP = TC.CROP


@dataclass
class FactoredConfig:
    train_files: int = 345
    val_frac: float = 0.15
    batch: int = 32
    latent_dim: int = 32
    lander_weight: float = 25.0
    edge_weight: float = 1.0
    ssim_weight: float = 0.5
    kl_weight: float = 1e-4
    state_weight: float = 1.0       # weight on θ-branch supervision
    # swap-equivariance: TEACH controllability (lander must move to a SWAPPED commanded x/y). Without this
    # the decoder only ever sees the TRUE x per frame -> reconstructs but never learns to place the lander
    # at a commanded position. Clean here because the scene latent won't fight the swap.
    equiv_weight: float = 2e-4  # was 5e-4 — lowered after both short runs diverged once swap ramped to full
    recon_warmup: int = 0       # STAGED option: pure-recon epochs (swap OFF) to establish a sharp lander+scene
    equiv_warmup: int = 8       # gentler ramp (was 5)
    per_axis: bool = True
    color_tol: float = 0.6
    dom_bias: float = 0.05
    lander_rgb: tuple = (0.361, 0.294, 0.642)
    # z_lander (B): a position-free APPEARANCE code from the lander crop, injected into z[app_start:] to
    # fix the soft-blob + restore geometric-θ readability. Scene scene latent then = z[4:app_start].
    use_zlander: bool = False
    app_dim: int = 8
    derotate_lander: bool = False     # False=v1 (crop keeps tilt; crisp+θ-readable, θ rides in the code)
    feat_weight: float = 0.0          # perceptual feature-matching on the lander crop (frozen random conv) — legs
    theta_equiv_weight: float = 0.0   # θ-swap-equivariance: dial z[2:4]=commanded θ, FROZEN-branch readback of the
                                      # decoded lander must match it -> fixes off-center θ-dial fragility (θ analogue of the x/y swap)
    init_ckpt: str = ""               # warm-start vae+branch from this checkpoint (e.g. baseline) — inherits θ-rendering
    lr: float = 1e-3
    grad_clip: float = 0.3      # tighter (was 0.5) — divergence fix
    # selection GATE: best epoch must be CONTROLLABLE (dial-x renders + sweeps the lander), then min val_recon
    ctrl_render_min: float = 80.0
    ctrl_px_min: float = 40.0       # lander must sweep >= this many px across the x-dial to count as controllable
    patience: int = 8
    min_epochs: int = 20      # bumped 17->20 so the θ-swap-equivariance ramp (equiv_warmup) fully finishes before early-stop is eligible
    epochs: int = 70
    snapshot_every: int = 6
    save_name: str = "factored_vae_v1"
    seed: int = 0
    device: str = "cuda"
    verbose: bool = True


def scene_cache(imgs_u8, tag):
    """Precompute (and cache) the lander-erased 'scene-only' frames the scene-latent encoder consumes."""
    path = os.path.join(config.HERE, "checkpoints", f"scene_{tag}.pt")
    if os.path.exists(path):
        return torch.load(path)
    out = torch.empty_like(imgs_u8)
    for i in range(imgs_u8.size(0)):
        s, _ = factored_data.scene_only(imgs_u8[i].float() / 255.)
        out[i] = (s * 255.0).round().clamp(0, 255).to(torch.uint8)
        if i % 500 == 0:
            print(f"  scene-only {i}/{imgs_u8.size(0)}", flush=True)
    torch.save(out, path)
    return out


@torch.no_grad()
def control_monitor(vae, scene_u8, states, idx, dev, n=8, app=None, crops=None, app_start=None, derotate=False):
    """Dial x AND y (scene latent from scene-only + injected true pose) -> does the lander RENDER and MOVE
    on each axis? Returns (render %, px-range swept in X, px-range swept in Y). Gating on BOTH position
    axes (not just x) so selection requires real position controllability, not an x-only proxy.
    (θ is verified post-hoc via the success test — it needs the geometric reader, not a centroid shift.)
    With z_lander (app given) the true lander appearance is injected into z[app_start:] so the dialed
    lander looks right; position control is what's measured, so the (upright) z[2:4] here is harmless."""
    vae.eval()
    cen = list(idx[:n])
    mu, _ = vae.encode(scene_u8[cen].to(dev).float() / 255.)
    base = mu.clone(); base[:, 2] = 1.0; base[:, 3] = 0.0                       # upright
    base[:, 0] = states[cen, config.X].to(dev); base[:, 1] = states[cen, config.Y].to(dev)   # true x,y
    if app is not None:
        cr = crops[cen].to(dev).float() / 255.
        cr = derotate_crop(cr, states[cen, config.TH].to(dev)) if derotate else cr
        base[:, app_start:] = app(cr)                                          # inject true lander appearance

    def sweep(dim, vals, axis):
        rend, cs = [], []
        for v in vals:
            z = base.clone(); z[:, dim] = float(v); dec = vae.decode(z)
            rend.append(np.mean([int(purple_mask(dec[i].cpu()).sum()) >= 8 for i in range(z.size(0))]))
            c = centroid_px(purple_mask(dec[0].cpu()).numpy())[axis]
            if np.isfinite(c):
                cs.append(c)
        return float(np.mean(rend)), (max(cs) - min(cs) if len(cs) >= 2 else 0.0)

    rx, pxx = sweep(0, controllability.X_RANGE, 0)
    ry, pxy = sweep(1, controllability.Y_RANGE, 1)
    return round(100 * float(min(rx, ry))), float(pxx), float(pxy)


def train_factored(cfg: FactoredConfig = None):
    cfg = cfg or FactoredConfig()
    checkpoints.enable_determinism(cfg.seed)
    dev = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    rng = np.random.default_rng(cfg.seed)

    imgs, crops, states, ptrue, ep = TC.preload(config.TRAIN_DIR, cfg.train_files)
    scenes = scene_cache(imgs, f"train_{cfg.train_files}")
    # CANONICAL by-file split — SAME episodes held out as the baseline stages (no cross-stage leak).
    # ep is the file-enumerate index over sorted(glob(TRAIN_DIR/*.npz))[:train_files], the SAME domain
    # the baseline scripts permute, so passing n_files here makes the val files identical across stages.
    n_files = len(sorted(glob.glob(os.path.join(config.TRAIN_DIR, "*.npz")))[:cfg.train_files])
    _, val_file_idx = checkpoints.canonical_file_split(n_files, cfg.val_frac, cfg.seed)
    va_mask = np.isin(ep, val_file_idx)
    tr_idx, va_idx = np.where(~va_mask)[0], np.where(va_mask)[0]
    print(f"factored VAE: {len(tr_idx)} train / {len(va_idx)} val frames "
          f"({len(val_file_idx)}/{n_files} val episodes, canonical split seed {cfg.seed})", flush=True)
    # translate-augmentation removed: the shipped model uses no augmentation (see TRAINING.md).

    vae = PiwmConvVAE(latent_dim=cfg.latent_dim).to(dev); branch = ThetaBranch().to(dev)
    app = LanderApp(cfg.app_dim).to(dev) if cfg.use_zlander else None
    app_start = cfg.latent_dim - cfg.app_dim                      # z_lander occupies z[app_start:]
    with torch.no_grad():
        branch(crops[:2].to(dev).float() / 255.)
        if app is not None: app(crops[:2].to(dev).float() / 255.)  # lazy-init the LazyLinear
    if cfg.init_ckpt:                                              # warm-start: inherit baseline θ-rendering + x/y
        d = torch.load(cfg.init_ckpt, map_location=dev)
        vae.load_state_dict(d["vae"]); branch.load_state_dict(d["branch"])
        print(f"warm-started vae+branch from {cfg.init_ckpt}", flush=True)
    branch_frozen = None                                          # frozen reader for θ-swap-equivariance (non-gameable)
    if cfg.theta_equiv_weight > 0:
        branch_frozen = copy.deepcopy(branch).to(dev).eval()
        for p in branch_frozen.parameters():
            p.requires_grad_(False)
        print("θ-swap-equivariance ON (frozen-branch readback)", flush=True)
    params = list(vae.parameters()) + list(branch.parameters()) + (list(app.parameters()) if app is not None else [])
    opt = torch.optim.Adam(params, lr=cfg.lr)
    window = gaussian_window(3, device=dev)
    c = torch.tensor(cfg.lander_rgb, dtype=torch.float32, device=dev)

    def lander_code(crop, th):                                   # z_lander appearance code from the crop
        cr = derotate_crop(crop, th) if cfg.derotate_lander else crop
        return app(cr)

    def fsd():                                                   # checkpoint state-dict (with app if present)
        return flat_sd_z(vae, branch, app) if app is not None else TC._flat_sd(vae, branch)

    featnet = FeatNet().to(dev) if cfg.feat_weight > 0 else None   # frozen perceptual extractor (legs)

    def batches(idx, shuffle):
        order = idx[rng.permutation(len(idx))] if shuffle else idx
        for i in range(0, len(order), cfg.batch):
            b = torch.as_tensor(order[i:i + cfg.batch])
            yield (imgs[b].to(dev).float() / 255., scenes[b].to(dev).float() / 255.,
                   crops[b].to(dev).float() / 255., states[b].to(dev), ptrue[b].to(dev))

    @torch.no_grad()
    def val_recon():
        vae.eval(); branch.eval();
        if app is not None: app.eval()
        rl = 0.0; n = 0
        for full, scene, crop, st, pt in batches(va_idx, False):
            mu, _ = vae.encode(scene); cossin = branch(crop)
            z = mu.clone(); z[:, 0] = st[:, config.X]; z[:, 1] = st[:, config.Y]; z[:, 2:4] = cossin
            if app is not None: z[:, app_start:] = lander_code(crop, st[:, config.TH])
            recon = vae.decode(z)
            rl += float(weighted_mse(recon, full, lander_weight_map(full, cfg.lander_weight))) * full.size(0); n += full.size(0)
        return rl / max(n, 1)

    hist = {k: [] for k in ("epoch", "recon", "kl", "theta", "equiv", "val_recon",
                            "ctrl_render", "ctrl_px_x", "ctrl_px_y", "sel")}
    best = dict(metric=float("inf"), epoch=-1, state=None, since=0); snaps = []; diverged = False
    # STAGED-aware early-stop floor: never stop before the swap has ramped to full + had `patience` epochs
    eff_min_epochs = max(cfg.min_epochs, cfg.recon_warmup + cfg.equiv_warmup + cfg.patience)

    for epoch in range(cfg.epochs):
        e = epoch - cfg.recon_warmup                       # STAGED: swap OFF during recon_warmup, then ramp
        equiv_w = 0.0 if e < 0 else cfg.equiv_weight * min(1.0, (e + 1) / max(1, cfg.equiv_warmup))
        theta_equiv_w = 0.0 if e < 0 else cfg.theta_equiv_weight * min(1.0, (e + 1) / max(1, cfg.equiv_warmup))
        vae.train(); branch.train()
        if app is not None: app.train()
        tot = {k: 0.0 for k in ("recon", "kl", "theta", "equiv", "feat", "theta_eq")}; n = 0; nan_skip = 0
        for full, scene, crop, st, pt in batches(tr_idx, True):
            bn = full.size(0)
            inj_xy = st[:, [config.X, config.Y]]; ptb = pt
            mu, logvar = vae.encode(scene)                 # scene latent from SCENE-ONLY (no lander)
            logvar = logvar.clamp(-8.0, 8.0)               # prevent scene latent-logvar/KL blowup (divergence fix)
            res = vae.reparameterize(mu, logvar)
            cossin = branch(crop)
            z = res.clone()
            z[:, 0] = inj_xy[:, 0]; z[:, 1] = inj_xy[:, 1]; z[:, 2:4] = cossin   # INJECT physical (aug-aware x,y)
            if app is not None: z[:, app_start:] = lander_code(crop, st[:, config.TH])  # INJECT appearance
            recon = vae.decode(z)
            wmap = lander_weight_map(full, cfg.lander_weight)
            mse = weighted_mse(recon, full, wmap)
            recon_l = mse + cfg.edge_weight * gradient_loss(recon, full, wmap) + cfg.ssim_weight * ssim_loss(recon, full, window)
            feat_l = (feat_loss(featnet, crop_at(recon, ptb, CROP), crop)   # perceptual match: decoded vs real lander crop (legs)
                      if featnet is not None else torch.zeros((), device=dev))
            kl_hi = app_start if app is not None else cfg.latent_dim    # don't KL the injected z_lander dims
            kl_l = kl_divergence(mu[:, 4:kl_hi], logvar[:, 4:kl_hi])
            th = st[:, config.TH]
            theta_l = F.mse_loss(cossin, torch.stack([torch.cos(th), torch.sin(th)], 1))
            # SWAP-equivariance: inject another frame's commanded x (or y), require the decoded lander to
            # render THERE (concentration at the swapped target px). This is the controllability signal.
            if equiv_w > 0 and bn > 1:
                def swap_loss(dims):
                    perm = torch.randperm(bn, device=dev)
                    zs = z.clone(); target = ptb.clone()
                    for d in dims:
                        zs[:, d] = z[perm, d]; target[:, d] = ptb[perm, d]
                    return TC.concentration_loss(TC.lander_mask(vae.decode(zs), c, cfg.color_tol, cfg.dom_bias), target).mean()
                equiv_l = (swap_loss([0]) + swap_loss([1])) if cfg.per_axis else swap_loss([0, 1])
            else:
                equiv_l = torch.zeros((), device=dev)
            # θ-swap-equivariance: dial z[2:4] to a swapped θ, FROZEN-branch readback of the decoded lander must match
            if branch_frozen is not None and theta_equiv_w > 0 and bn > 1:
                permt = torch.randperm(bn, device=dev)
                th_sw = st[permt, config.TH]
                tgt_t = torch.stack([torch.cos(th_sw), torch.sin(th_sw)], 1)
                zt = z.clone(); zt[:, 2:4] = tgt_t                                  # command the swapped tilt
                read_t = branch_frozen(crop_at(vae.decode(zt), ptb, CROP))           # frozen reader on the decoded lander
                theta_equiv_l = F.mse_loss(read_t, tgt_t)
            else:
                theta_equiv_l = torch.zeros((), device=dev)
            loss = (recon_l + cfg.kl_weight * kl_l + cfg.state_weight * theta_l + equiv_w * equiv_l
                    + cfg.feat_weight * feat_l + theta_equiv_w * theta_equiv_l)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); nan_skip += 1; continue   # skip bad batch, KEEP params (no backward)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            tot["recon"] += float(mse.detach()) * bn; tot["kl"] += float(kl_l.detach()) * bn
            tot["theta"] += float(theta_l.detach()) * bn; tot["equiv"] += float(equiv_l.detach()) * bn
            tot["feat"] += float(feat_l.detach()) * bn; tot["theta_eq"] += float(theta_equiv_l.detach()) * bn; n += bn
        if diverged:
            break
        n = max(n, 1)
        vr = val_recon()
        cr, cpx_x, cpx_y = control_monitor(vae, scenes, states, va_idx, dev, app=app, crops=crops,
                                           app_start=app_start, derotate=cfg.derotate_lander)
        gated = (cr >= cfg.ctrl_render_min) and (cpx_x >= cfg.ctrl_px_min) and (cpx_y >= cfg.ctrl_px_min)  # x AND y
        sel = vr if gated else 1e4 + vr
        for k, v in [("epoch", epoch), ("recon", tot["recon"]/n), ("kl", tot["kl"]/n), ("theta", tot["theta"]/n),
                     ("equiv", tot["equiv"]/n), ("val_recon", vr), ("ctrl_render", cr),
                     ("ctrl_px_x", cpx_x), ("ctrl_px_y", cpx_y), ("sel", sel)]:
            hist[k].append(v)
        improved = sel < best["metric"] - 1e-6
        cand = lambda: copy.deepcopy({"vae": vae.state_dict(), "branch": branch.state_dict()})
        emet = dict(epoch=epoch, val_recon=vr, ctrl_render=cr, ctrl_px_x=cpx_x, ctrl_px_y=cpx_y, gated=bool(gated))
        if improved:
            best.update(metric=sel, epoch=epoch, state=cand(), since=0)
            checkpoints.save_checkpoint(cfg.save_name + "_best", fsd(), metrics=emet, extra=dict(config=asdict(cfg)))
        else:
            best["since"] += 1
        if cfg.snapshot_every and epoch % cfg.snapshot_every == 0:
            nm = f"{cfg.save_name}_ep{epoch}"
            checkpoints.save_checkpoint(nm, fsd(), metrics=emet, extra=dict(config=asdict(cfg))); snaps.append(nm)
        if cfg.verbose:
            print(f"  ep {epoch:3d}/{cfg.epochs} recon {tot['recon']/n:.4f} kl {tot['kl']/n:.1f} θ {tot['theta']/n:.4f} "
                  f"equiv {tot['equiv']/n:.1f} feat {tot['feat']/n:.3f} θeq {tot['theta_eq']/n:.3f} eqw {equiv_w:.1e} | val_recon {vr:.4f} "
                  f"ctrl[render {cr}% px_x {cpx_x:.0f} px_y {cpx_y:.0f}] {'GATED' if gated else 'defer'}{'  *best' if improved else ''}"
                  f"{f'  nan_skip {nan_skip}' if nan_skip else ''}", flush=True)
        if epoch + 1 >= eff_min_epochs and best["since"] >= cfg.patience:
            print(f"  early stop @ {epoch} (best sel {best['metric']:.4f}@{best['epoch']})", flush=True)
            break

    last_sd = copy.deepcopy(fsd())
    if best["state"] is not None:
        vae.load_state_dict(best["state"]["vae"]); branch.load_state_dict(best["state"]["branch"])
    metrics = dict(best_sel=best["metric"], ever_controllable=bool(best["metric"] < 1e4),
                   best_val_recon=float(hist["val_recon"][best["epoch"]]) if best["epoch"] >= 0 and hist["val_recon"] else None,
                   best_ctrl_px_x=float(hist["ctrl_px_x"][best["epoch"]]) if best["epoch"] >= 0 and hist["ctrl_px_x"] else None,
                   best_ctrl_px_y=float(hist["ctrl_px_y"][best["epoch"]]) if best["epoch"] >= 0 and hist["ctrl_px_y"] else None,
                   best_epoch=int(best["epoch"]), epochs_ran=int(hist["epoch"][-1] + 1) if hist["epoch"] else 0,
                   diverged=diverged, n_train=int(len(tr_idx)), n_val=int(len(va_idx)), snapshots=snaps)
    extra = dict(config=asdict(cfg), arch="FactoredVAE: scene latent encoder + injected physical dims",
                 latent_layout="z0=x(injected) z1=y(injected) z2:4=branch(crop)(cos,sin) z4:=scene-latent (scene-only encode)")
    checkpoints.save_checkpoint(cfg.save_name, fsd(), metrics=metrics, extra=extra)
    checkpoints.save_checkpoint(cfg.save_name + "_last", last_sd, metrics=metrics, extra=extra)   # final epoch (pre-restore)
    return dict(state_dict=fsd(), last_state_dict=last_sd, history=hist, metrics=metrics,
                snapshots=snaps, extra=extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train_files", type=int, default=None)
    ap.add_argument("--recon_warmup", type=int, default=None, help="STAGED: pure-recon epochs before the swap ramps in")
    ap.add_argument("--lander_weight", type=float, default=None, help="recon up-weight on lander pixels (default 25)")
    ap.add_argument("--edge_weight", type=float, default=None, help="gradient/edge loss weight (default 1) — crank to reward crisp legs/silhouette")
    ap.add_argument("--ssim_weight", type=float, default=None, help="SSIM structural loss weight (default 0.5)")
    ap.add_argument("--save_name", default=None, help="checkpoint name (use a fresh one to not clobber factored_vae_v1)")
    ap.add_argument("--use_zlander", action="store_true", help="B: add z_lander appearance code (z[app_start:]) from the crop")
    ap.add_argument("--app_dim", type=int, default=None, help="tail dims z_lander occupies (default 8)")
    ap.add_argument("--derotate_lander", action="store_true", help="upright-normalize the crop (pose-free; θ dial-able via z[2:4])")
    ap.add_argument("--feat_weight", type=float, default=None, help="perceptual feature-matching weight on the lander crop (legs); 0=off")
    ap.add_argument("--theta_equiv_weight", type=float, default=None, help="θ-swap-equivariance weight (fixes off-center θ-dial fragility); 0=off")
    ap.add_argument("--init_ckpt", default=None, help="warm-start vae+branch from this checkpoint (baseline pos_equiv_conc_full/model.pth)")
    ap.add_argument("--lr", type=float, default=None, help="learning rate (lower for warm-start fine-tune, e.g. 5e-4)")
    a = ap.parse_args()
    cfg = FactoredConfig(device=a.device)
    if a.save_name is not None:
        cfg.save_name = a.save_name
    if a.use_zlander:
        cfg.use_zlander = True
    if a.app_dim is not None:
        cfg.app_dim = a.app_dim
    if a.derotate_lander:
        cfg.derotate_lander = True
    if a.feat_weight is not None:
        cfg.feat_weight = a.feat_weight
    if a.theta_equiv_weight is not None:
        cfg.theta_equiv_weight = a.theta_equiv_weight
    if a.init_ckpt is not None:
        cfg.init_ckpt = a.init_ckpt
    if a.lr is not None:
        cfg.lr = a.lr
    if a.recon_warmup is not None:
        cfg.recon_warmup = a.recon_warmup
    if a.lander_weight is not None:
        cfg.lander_weight = a.lander_weight
    if a.edge_weight is not None:
        cfg.edge_weight = a.edge_weight
    if a.ssim_weight is not None:
        cfg.ssim_weight = a.ssim_weight
    if a.smoke:
        cfg.train_files, cfg.epochs, cfg.min_epochs, cfg.patience, cfg.snapshot_every = 12, 4, 2, 3, 2
    if a.epochs is not None:
        cfg.epochs = a.epochs
    if a.train_files is not None:
        cfg.train_files = a.train_files
    res = train_factored(cfg)
    print("DONE", res["metrics"], flush=True)


if __name__ == "__main__":
    main()
