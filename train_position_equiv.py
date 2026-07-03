"""STAGE 2 of 3 in the shipped VAE's warm-start chain (run the whole chain with reproduce.sh; see
TRAINING.md). Warm-starts from stage 1 and makes position controllable at decode time.

PIWM VAE + theta-branch + POSITION-EQUIVARIANCE loss (disentangle position).

Problem: in the theta-branch model the decoder renders lander position from the
RESIDUAL dims, not from the supervised z[0:2] (check_position_control.py shows a
flat sweep). The encoder writes x,y into z[0:2], but nothing forces the decoder to
READ them or forbids the scene-latent dims from also carrying position.

Fix (Principle 2, equivariance) -- a LOSS, not a new encoder. In addition to the
normal reconstruction path, run a SWAP path each step:
  * encode batch -> z; know each frame's true lander pixel centroid p.
  * z_swap[i] = z[i] but z_swap[i,0:2] = z[perm[i],0:2]  (borrow another frame's
    position dims, keep frame i's scene-latent dims/appearance).
  * decode z_swap, take its differentiable soft-centroid, penalize
    ||centroid - p[perm[i]]||^2.
This forces the decoder to put the lander where z[0:2] says (equivariance) AND, by
demanding the lander move away from where the scene-latent dims "want" it, squeezes position
out of the scene-latent dims into z[0:2] (disentanglement).

Warm-start from the canonical theta-branch model (--init_ckpt) is recommended on a
small GPU. Verify after with: scripts/check_position_control.py --ckpt <out>/model.pth

Usage (smoke):
  .venv/bin/python scripts/train_position_equiv.py --train_files 40 --test_files 10 \
      --epochs 15 --init_ckpt outputs/theta_branch_clean/model.pth \
      --output_dir outputs/pos_equiv_smoke
"""
import argparse
import glob
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F

from piwm_model.autoencoder import PiwmConvVAE, kl_divergence
from piwm_model.train_utils import set_seed
from piwm_model.sprite import purple_mask
from piwm_model.sprite import soft_purple, soft_centroid

from train_theta_branch_vae import (
    ThetaBranch, preload, batches, lander_weight_map, weighted_mse,
    gradient_loss, gaussian_window, ssim_loss, largest_component,
    theta_err_deg, CROP,
)


def compute_centroids(F_):
    """True lander pixel centroid (cx, cy) per frame via largest purple component."""
    out = np.full((F_.size(0), 2), np.nan, dtype=np.float32)
    for i in range(F_.size(0)):
        chw = F_[i].float() / 255.0
        comp = largest_component(purple_mask(chw).numpy())
        if comp is None:
            continue
        ys, xs = np.where(comp)
        out[i] = (xs.mean(), ys.mean())
    return torch.from_numpy(out)


def concentration_loss(mask, target):
    """mask (B,H,W) soft purple, target (B,2) px -> per-sample mass-weighted mean
    squared distance of purple to target (px^2). Minimized ONLY when ALL purple mass
    sits at the target -- unlike a centroid match, a haze/ghost far away is heavily
    penalized (can't be cancelled by symmetric spread)."""
    B, H, W = mask.shape
    ys = torch.arange(H, device=mask.device, dtype=mask.dtype).view(1, H, 1)
    xs = torch.arange(W, device=mask.device, dtype=mask.dtype).view(1, 1, W)
    dx = xs - target[:, 0].view(B, 1, 1)
    dy = ys - target[:, 1].view(B, 1, 1)
    d2 = dx * dx + dy * dy
    s = mask.sum(dim=(1, 2)).clamp_min(1e-6)
    return (mask * d2).sum(dim=(1, 2)) / s


@torch.no_grad()
def hard_recon_pos_px(vae, branch, F_, C_, P_, device, n=150):
    """Median HARD (largest-component) recon lander-position error vs true centroid.
    This is the real quality metric the soft equiv loss can game."""
    idx = np.linspace(0, F_.size(0) - 1, min(n, F_.size(0))).round().astype(int)
    errs = []
    for i in idx:
        mu, _ = vae.encode(F_[i:i + 1].to(device).float() / 255.0)
        z = mu.clone(); z[:, 2:4] = branch(C_[i:i + 1].to(device).float() / 255.0)
        img = vae.decode(z)[0].cpu()
        comp = largest_component(purple_mask(img).numpy())
        t = P_[i]
        if comp is None or not torch.isfinite(t).all():
            continue
        ys, xs = np.where(comp)
        errs.append(((xs.mean() - t[0].item()) ** 2 + (ys.mean() - t[1].item()) ** 2) ** 0.5)
    return float(np.median(errs)) if errs else float("nan")


def batches_p(F_, C_, S_, P_, bs, device, shuffle):
    n = F_.size(0)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        b = idx[i:i + bs]
        yield (F_[b].to(device).float() / 255.0,
               C_[b].to(device).float() / 255.0,
               S_[b].to(device),
               P_[b].to(device))


def run_epoch(vae, branch, data, device, args, window, opt=None):
    train = opt is not None
    vae.train(train); branch.train(train)
    tot = {"recon": 0.0, "edge": 0.0, "ssim": 0.0, "xy": 0.0, "theta": 0.0, "equiv": 0.0}
    n = 0
    for frame, crop, state, ptrue in data:
        bn = frame.size(0)
        with torch.set_grad_enabled(train):
            mu, logvar = vae.encode(frame)
            z = vae.reparameterize(mu, logvar)
            cossin = branch(crop)
            z = z.clone()
            z[:, 2:4] = cossin
            recon = vae.decode(z)

            wmap = lander_weight_map(frame, args.lander_weight)
            mse = weighted_mse(recon, frame, wmap)
            edge = gradient_loss(recon, frame, wmap)
            ssim = ssim_loss(recon, frame, window)
            recon_loss = mse + args.edge_weight * edge + args.ssim_weight * ssim

            xy_loss = F.mse_loss(mu[:, 0:2], state[:, [0, 1]])
            th = state[:, 4]
            theta_loss = F.mse_loss(cossin, torch.stack([torch.cos(th), torch.sin(th)], 1))
            kl = kl_divergence(mu[:, 4:], logvar[:, 4:])

            # --- position-equivariance (swap) path ---
            # latent dim d (0=x, 1=y) maps to pixel-centroid axis d (0=cx, 1=cy),
            # so swapping latent dim d means swapping target pixel axis d.
            valid = torch.isfinite(ptrue).all(dim=1)
            equiv = torch.zeros((), device=device)

            def swap_loss(dims):
                perm = torch.randperm(bn, device=device)
                zs = z.detach().clone() if args.equiv_detach_residual else z.clone()
                target = ptrue.clone()
                for d in dims:
                    zs[:, d] = z[perm, d]       # borrow another frame's position dim
                    target[:, d] = ptrue[perm, d]
                m = soft_purple(vae.decode(zs))
                ok = valid & valid[perm]
                if ok.sum() == 0:
                    return torch.zeros((), device=device)
                # concentration: all purple must sit at target (forbids haze/ghosts)
                return concentration_loss(m, target)[ok].mean()

            if args.equiv_weight > 0 and valid.sum() > 1:
                if args.per_axis:
                    # swap x alone (move horizontally, hold height) + y alone
                    equiv = swap_loss([0]) + swap_loss([1])
                else:
                    equiv = swap_loss([0, 1])   # joint 2-D position swap

            loss = (recon_loss + args.kl_weight * kl
                    + args.state_weight * (xy_loss + theta_loss)
                    + args.equiv_weight * equiv)
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(vae.parameters()) + list(branch.parameters()), args.grad_clip)
            opt.step()
        tot["recon"] += float(mse) * bn
        tot["edge"] += float(edge) * bn
        tot["ssim"] += float(ssim) * bn
        tot["xy"] += float(xy_loss) * bn
        tot["theta"] += float(theta_loss) * bn
        tot["equiv"] += float(equiv) * bn
        n += bn
    return {k: v / max(n, 1) for k, v in tot.items()}


@torch.no_grad()
def position_control_px(vae, branch, F_, C_, device, n_base=5):
    """Quick metric: how many px does the lander move when we sweep z[0] / z[1]?"""
    idx = np.linspace(0, F_.size(0) - 1, n_base).round().astype(int)
    xr = torch.linspace(-0.7, 0.7, 7, device=device)
    yr = torch.linspace(0.2, 1.3, 7, device=device)

    def one(dim, vals):
        moved = []
        for i in idx:
            mu, _ = vae.encode(F_[i:i + 1].to(device).float() / 255.0)
            z0 = mu.clone(); z0[:, 2:4] = branch(C_[i:i + 1].to(device).float() / 255.0)
            cs = []
            for v in vals:
                z = z0.clone(); z[0, dim] = v
                c = soft_centroid(soft_purple(vae.decode(z)))[0]
                cs.append(c[dim].item())
            moved.append(max(cs) - min(cs))
        return float(np.median(moved))

    return one(0, xr), one(1, yr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="../data/lunar/extracted/lunar")
    p.add_argument("--output_dir", default="outputs/pos_equiv_smoke")
    p.add_argument("--init_ckpt", default="", help="warm-start theta-branch model")
    p.add_argument("--train_files", type=int, default=40)
    p.add_argument("--test_files", type=int, default=10)
    p.add_argument("--val_frac", type=float, default=0.15,
                   help="fraction of lunartrain EPISODES held out as val for epoch SELECTION; "
                        "lunartest stays held out for the final report number only (no leak)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--lander_weight", type=float, default=25.0)
    p.add_argument("--kl_weight", type=float, default=1e-4)
    p.add_argument("--state_weight", type=float, default=1.0)
    p.add_argument("--edge_weight", type=float, default=1.0)
    p.add_argument("--ssim_weight", type=float, default=0.5)
    p.add_argument("--equiv_weight", type=float, default=5e-4,
                   help="weight on position-equivariance concentration loss (px^2)")
    p.add_argument("--ctrl_min", type=float, default=15.0,
                   help="min z[0]-sweep px movement to consider control 'achieved' for selection")
    p.add_argument("--equiv_detach_residual", action="store_true",
                   help="detach scene-latent dims in swap path (train only z[0:2]->decoder route)")
    p.add_argument("--per_axis", action="store_true",
                   help="swap x and y separately (penalizes x/y cross-talk), not jointly")
    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min_epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.seed >= 0:
        set_seed(args.seed)
        # full determinism for a bulletproof, bitwise-reproducible report (matches
        # notebooks/checkpoints.enable_determinism; CUBLAS var must be set before cuBLAS init).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    # CANONICAL by-FILE (=episode) train/val split. MUST match
    # notebooks/checkpoints.canonical_file_split(n_files, val_frac, seed) so EVERY warm-start stage
    # holds out the SAME episodes for selection (no cross-stage leak). All SELECTION metrics on val;
    # lunartest stays held out and is scored only once at the end (the report number).
    all_train = sorted(glob.glob(os.path.join(args.data_root, "lunartrain", "*.npz")))[:args.train_files]
    n_files = len(all_train)
    perm = np.random.default_rng(args.seed if args.seed >= 0 else 0).permutation(n_files)
    n_val = max(1, int(round(args.val_frac * n_files)))
    val_idx = set(np.sort(perm[:n_val]).tolist())
    val_files = [all_train[i] for i in range(n_files) if i in val_idx]
    train_files = [all_train[i] for i in range(n_files) if i not in val_idx]
    trF, trC, trS = preload(None, None, files=train_files)
    vaF, vaC, vaS = preload(None, None, files=val_files)
    teF, teC, teS = preload(os.path.join(args.data_root, "lunartest"), args.test_files)
    trP = compute_centroids(trF); vaP = compute_centroids(vaF); teP = compute_centroids(teF)
    print(f"preloaded {trF.size(0)} train / {vaF.size(0)} val ({n_val} eps) / {teF.size(0)} test "
          f"+ centroids in {time.time()-t0:.0f}s")

    vae = PiwmConvVAE(latent_dim=args.latent_dim).to(device)
    branch = ThetaBranch().to(device)
    with torch.no_grad():
        branch(trC[:2].to(device).float() / 255.0)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location=device)
        vae.load_state_dict(ck["vae"]); branch.load_state_dict(ck["branch"])
        print(f"warm-started from {args.init_ckpt}")
    opt = torch.optim.Adam(list(vae.parameters()) + list(branch.parameters()), lr=args.lr)
    window = gaussian_window(3, device=device)

    mx0, my0 = position_control_px(vae, branch, vaF, vaC, device)
    print(f"position-control BEFORE (val): z[0] sweep moves {mx0:.1f}px, z[1] sweep moves {my0:.1f}px")

    best, best_sd, best_ep, no_imp = float("inf"), None, 0, 0
    for ep in range(1, args.epochs + 1):
        te0 = time.time()
        tr = run_epoch(vae, branch, batches_p(trF, trC, trS, trP, args.batch, device, True),
                       device, args, window, opt)
        va = run_epoch(vae, branch, batches_p(vaF, vaC, vaS, vaP, args.batch, device, False),
                       device, args, window)
        mx, my = position_control_px(vae, branch, vaF, vaC, device)
        hpos = hard_recon_pos_px(vae, branch, vaF, vaC, vaP, device)
        # selection (ALL on val): once control is achieved (z[0] sweep >= ctrl_min), pick the model
        # with the crispest HARD recon position; before that, defer (large penalty).
        ctrl_ok = mx >= args.ctrl_min
        sel = hpos if ctrl_ok else 1e4 + va["recon"]
        improved = sel < best - 1e-3
        if improved:
            best, best_ep, no_imp = sel, ep, 0
            best_sd = ({k: v.detach().cpu().clone() for k, v in vae.state_dict().items()},
                       {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()})
        else:
            no_imp += 1
        print(f"epoch {ep}/{args.epochs} ({time.time()-te0:.0f}s) | "
              f"mse {tr['recon']:.4f} equiv {tr['equiv']:.0f}px^2 theta {tr['theta']:.4f} | "
              f"val mse {va['recon']:.4f} | ctrl(val) z0 {mx:.1f}px z1 {my:.1f}px | "
              f"HARD recon-pos(val) {hpos:.1f}px | best {best:.1f}@{best_ep}{' *' if improved else ''}")
        if args.patience > 0 and ep >= args.min_epochs and no_imp >= args.patience:
            print(f"early stop @ {ep} (best val @ {best_ep})"); break

    if best_sd is not None:
        vae.load_state_dict(best_sd[0]); branch.load_state_dict(best_sd[1])
    # FINAL eval on HELD-OUT lunartest = the report numbers (selection was on val).
    th = theta_err_deg(branch, teC, teS, device)
    mx, my = position_control_px(vae, branch, teF, teC, device)
    hpos = hard_recon_pos_px(vae, branch, teF, teC, teP, device)
    print(f"\nFINAL theta median {th.median():.2f} deg | HARD recon-pos {hpos:.1f}px | "
          f"position-control AFTER: z[0] {mx:.1f}px, z[1] {my:.1f}px "
          f"(was {mx0:.1f}/{my0:.1f}px)  [held-out lunartest]")
    # model.pth = the BEST-VAL epoch (already restored above); provenance for the report.
    torch.save({"vae": vae.state_dict(), "branch": branch.state_dict(),
                "selected_epoch": best_ep, "best_val_sel": best, "val_frac": args.val_frac,
                "seed": args.seed, "test_theta_median_deg": float(th.median()),
                "test_hard_recon_pos_px": float(hpos)},
               os.path.join(args.output_dir, "model.pth"))
    print(f"saved {args.output_dir}/model.pth  (best-val epoch {best_ep}, sel {best:.2f})")


if __name__ == "__main__":
    main()
