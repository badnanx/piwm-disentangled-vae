"""STAGE 1 of 3 in the shipped VAE's warm-start chain (run the whole chain with reproduce.sh; see
TRAINING.md). Trains the base VAE + tilt reader from scratch; stages 2 and 3 fine-tune these weights.

PIWM VAE with a dedicated theta-branch (P1-style functional split).

Main encoder: full 100x150 frame -> x, y, scene latent dims (it's already great at x/y).
Theta branch: 24x24 lander crop -> z[2:4] = (cos theta, sin theta) (proven ~0.5 deg).
Decoder: assembled latent -> image. Retraining this way also tests whether the
DECODER learns to render orientation once the latent actually carries theta.

The crop is located via the purple-mask centroid (segment to locate, crop raw pixels).

Usage (smoke):
  .venv/bin/python scripts/train_theta_branch_vae.py --train_files 40 --test_files 10 \
      --epochs 20 --preload --output_dir outputs/theta_branch_smoke
"""
import argparse
import glob
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from piwm_model.autoencoder import PiwmConvVAE, kl_divergence
from piwm_model.data import lander_fully_visible
from piwm_model.train_utils import set_seed
from piwm_model.sprite import purple_mask, _PURPLE_BIAS, _PURPLE_MIN

CROP = 24  # lander crop size for the theta branch


# ---------- theta branch ----------
class ThetaBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.LazyLinear(64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, crop):
        return self.head(self.conv(crop))  # (B,2) = (cos, sin)


# ---------- data ----------
def largest_component(mask):
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    big = 1 + int(np.argmax(ndimage.sum(mask, labels, range(1, n + 1))))
    return labels == big


def preload(data_dir, max_files, files=None):
    """Full frames (uint8), lander crops (uint8 CROPxCROP), states.

    Pass an explicit `files` list to load a specific set of episodes (used to carve a
    by-episode train/val split from lunartrain so no frames leak across the split)."""
    if files is None:
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))[:max_files]
    frames, crops, states = [], [], []
    for path in files:
        with np.load(path) as d:
            imgs, st = d["imgs"], d["states"]
        for t in range(len(imgs)):
            if not lander_fully_visible(imgs[t]):
                continue
            img = imgs[t]
            chw = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            comp = largest_component(purple_mask(chw).numpy())
            if comp is None:
                continue
            ys, xs = np.where(comp)
            cx, cy = xs.mean(), ys.mean()
            H, W = img.shape[0], img.shape[1]
            x0 = int(np.clip(round(cx - CROP / 2), 0, W - CROP))
            y0 = int(np.clip(round(cy - CROP / 2), 0, H - CROP))
            frames.append(img)
            crops.append(img[y0:y0 + CROP, x0:x0 + CROP])
            states.append(st[t])
    F_ = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
    C_ = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).contiguous()
    S_ = torch.from_numpy(np.stack(states).astype(np.float32))
    return F_, C_, S_


def batches(F_, C_, S_, bs, device, shuffle):
    n = F_.size(0)
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        b = idx[i:i + bs]
        yield (F_[b].to(device).float() / 255.0,
               C_[b].to(device).float() / 255.0,
               S_[b].to(device))


def lander_weight_map(images, w):
    r, g, bl = images[:, 0], images[:, 1], images[:, 2]
    m = (bl > r + _PURPLE_BIAS) & (bl > g + _PURPLE_BIAS) & (bl > _PURPLE_MIN)
    wm = torch.ones_like(m, dtype=images.dtype)
    wm[m] = w
    return wm.unsqueeze(1)


def weighted_mse(recon, frame, wmap):
    e = wmap.expand_as(recon)
    return (e * (recon - frame) ** 2).sum() / e.sum()


def gradient_loss(recon, frame, wmap):
    """Lander-weighted MSE on finite-difference image gradients (penalizes blur)."""
    rdx = recon[:, :, :, 1:] - recon[:, :, :, :-1]; tdx = frame[:, :, :, 1:] - frame[:, :, :, :-1]
    rdy = recon[:, :, 1:, :] - recon[:, :, :-1, :]; tdy = frame[:, :, 1:, :] - frame[:, :, :-1, :]
    wx = wmap[:, :, :, 1:].expand_as(rdx); wy = wmap[:, :, 1:, :].expand_as(rdy)
    lx = (wx * (rdx - tdx) ** 2).sum() / wx.sum()
    ly = (wy * (rdy - tdy) ** 2).sum() / wy.sum()
    return lx + ly


def gaussian_window(ch, k=7, sigma=1.5, device="cpu"):
    coords = torch.arange(k, device=device).float() - k // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    w2d = g[:, None] * g[None, :]
    return w2d.expand(ch, 1, k, k).contiguous()


def ssim_loss(recon, frame, window):
    """1 - mean SSIM (structural similarity); rewards local structure/contrast."""
    ch = recon.size(1); pad = window.size(-1) // 2
    mu1 = F.conv2d(recon, window, padding=pad, groups=ch)
    mu2 = F.conv2d(frame, window, padding=pad, groups=ch)
    m1, m2, m12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(recon * recon, window, padding=pad, groups=ch) - m1
    s2 = F.conv2d(frame * frame, window, padding=pad, groups=ch) - m2
    s12 = F.conv2d(recon * frame, window, padding=pad, groups=ch) - m12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    smap = ((2 * m12 + C1) * (2 * s12 + C2)) / ((m1 + m2 + C1) * (s1 + s2 + C2))
    return 1 - smap.mean()


def run_epoch(vae, branch, data, device, args, window, opt=None):
    train = opt is not None
    vae.train(train); branch.train(train)
    tot = {"recon": 0.0, "edge": 0.0, "ssim": 0.0, "xy": 0.0, "theta": 0.0}; n = 0
    for frame, crop, state in data:
        bn = frame.size(0)
        with torch.set_grad_enabled(train):
            mu, logvar = vae.encode(frame)
            z = vae.reparameterize(mu, logvar)
            cossin = branch(crop)                      # (B,2)
            z = z.clone()
            z[:, 2:4] = cossin                          # theta from the branch
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
            loss = recon_loss + args.kl_weight * kl + args.state_weight * (xy_loss + theta_loss)
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
        n += bn
    return {k: v / max(n, 1) for k, v in tot.items()}


def theta_err_deg(branch, C_, S_, device):
    branch.eval()
    with torch.no_grad():
        cs = branch(C_.to(device).float() / 255.0).cpu()
    pred = torch.atan2(cs[:, 1], cs[:, 0])
    d = pred - S_[:, 4]
    return (torch.abs(torch.atan2(torch.sin(d), torch.cos(d))) * 180 / math.pi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="../data/lunar/extracted/lunar")
    p.add_argument("--output_dir", default="outputs/theta_branch_smoke")
    p.add_argument("--train_files", type=int, default=40)
    p.add_argument("--test_files", type=int, default=10)
    p.add_argument("--val_frac", type=float, default=0.15,
                   help="fraction of lunartrain EPISODES held out as val for epoch SELECTION; "
                        "lunartest stays held out for the final report number only (no leak)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--lander_weight", type=float, default=25.0)
    p.add_argument("--kl_weight", type=float, default=1e-4)
    p.add_argument("--state_weight", type=float, default=1.0)
    p.add_argument("--edge_weight", type=float, default=1.0,
                   help="weight on gradient/edge sharpness loss (0 = off)")
    p.add_argument("--ssim_weight", type=float, default=0.5,
                   help="weight on (1-SSIM) structural loss (0 = off)")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="max grad norm (prevents the NaN divergence)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--min_epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0, help="set for reproducibility (-1 = off)")
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
    # CANONICAL by-FILE (=episode) train/val split of lunartrain. MUST match
    # notebooks/checkpoints.canonical_file_split(n_files, val_frac, seed) so EVERY warm-start stage
    # (theta-branch, pos-equiv, factored) holds out the SAME episodes for selection (no cross-stage
    # leak). lunartest is held out entirely and scored only once at the end (the report number).
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
    print(f"preloaded {trF.size(0)} train / {vaF.size(0)} val ({n_val} eps) / {teF.size(0)} test "
          f"(frames+{CROP}x{CROP} crops) in {time.time()-t0:.0f}s")

    vae = PiwmConvVAE(latent_dim=args.latent_dim).to(device)
    branch = ThetaBranch().to(device)
    with torch.no_grad():  # init LazyLinear
        branch(trC[:2].to(device).float() / 255.0)
    opt = torch.optim.Adam(list(vae.parameters()) + list(branch.parameters()), lr=args.lr)
    window = gaussian_window(3, device=device)

    best, best_sd, best_ep, no_imp = float("inf"), None, 0, 0
    for ep in range(1, args.epochs + 1):
        te0 = time.time()
        tr = run_epoch(vae, branch, batches(trF, trC, trS, args.batch, device, True), device, args, window, opt)
        va = run_epoch(vae, branch, batches(vaF, vaC, vaS, args.batch, device, False), device, args, window)
        th_med = theta_err_deg(branch, vaC, vaS, device).median()
        improved = va["recon"] < best - 1e-5
        if improved:
            best, best_ep, no_imp = va["recon"], ep, 0
            best_sd = ({k: v.detach().cpu().clone() for k, v in vae.state_dict().items()},
                       {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()})
        else:
            no_imp += 1
        print(f"epoch {ep}/{args.epochs} ({time.time()-te0:.0f}s) | "
              f"train mse {tr['recon']:.4f} edge {tr['edge']:.4f} ssim {tr['ssim']:.3f} "
              f"theta {tr['theta']:.4f} | val mse {va['recon']:.4f} | "
              f"theta-branch median(val) {th_med:.1f} deg | best {best:.4f}@{best_ep}{' *' if improved else ''}")
        if args.patience > 0 and ep >= args.min_epochs and no_imp >= args.patience:
            print(f"early stop @ {ep} (best val recon @ {best_ep})"); break

    if best_sd is not None:
        vae.load_state_dict(best_sd[0]); branch.load_state_dict(best_sd[1])
    th = theta_err_deg(branch, teC, teS, device)   # HELD-OUT lunartest = the report number
    print(f"\nFINAL theta-branch: median {th.median():.2f} deg  mean {th.mean():.2f} deg "
          f"(full-VAE baseline was ~20.6 deg)  [held-out lunartest]")
    # model.pth = the BEST-VAL epoch (already restored above); provenance records which epoch
    # was selected, on what split, and the held-out test number — for the report.
    torch.save({"vae": vae.state_dict(), "branch": branch.state_dict(),
                "selected_epoch": best_ep, "best_val_recon": best, "val_frac": args.val_frac,
                "seed": args.seed, "test_theta_median_deg": float(th.median())},
               os.path.join(args.output_dir, "model.pth"))
    print(f"saved {args.output_dir}/model.pth  (best-val epoch {best_ep}, val recon {best:.4f})")


if __name__ == "__main__":
    main()
