"""Eyeball a z_lander VAE against the report's two demands: (1) is the lander CRISP (silhouette/legs) vs the
factored_staged blob, and (2) is the appearance code actually USED — i.e. does the decoder RENDER from it or
just PASTE a fixed template? (the report's untested render-vs-paste limitation, which is also the
transferability crux).

Top block: rows = REAL / factored_staged (blob baseline) / zlander, over seeded-random representative frames
sorted by θ; a zoomed lander strip makes 15px crispness judgeable.
Bottom block (RENDER-VS-PASTE): hold ONE frame's scene+pose fixed and swap in z_lander from several OTHER
frames. If the rendered lander changes, the decoder renders from the appearance code (good, transferable);
if it's identical, it's pasting a fixed template (the toy shortcut).
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

import config
import checkpoints
import factored_data
import train_clean_vae as TC
from lander_app import LanderApp, derotate_crop

sys.path.insert(0, config.BASELINE_SRC)
sys.path.insert(0, config.BASELINE_SCRIPTS)
from piwm_model.autoencoder import PiwmConvVAE  # noqa: E402
from piwm_model.sprite import purple_mask  # noqa: E402
from train_theta_branch_vae import ThetaBranch  # noqa: E402

CROP = TC.CROP


def load(name, dev):
    sd, man = checkpoints.load_checkpoint(name, map_location=dev)
    cfgd = man["extra"]["config"]; ld = cfgd["latent_dim"]
    vae = PiwmConvVAE(ld).to(dev); branch = ThetaBranch().to(dev)
    with torch.no_grad():
        branch(torch.zeros(1, 3, CROP, CROP, device=dev))
    vae.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("vae.")})
    branch.load_state_dict({k[7:]: v for k, v in sd.items() if k.startswith("branch.")})
    app = None; app_dim = int(cfgd.get("app_dim", 8)); derot = bool(cfgd.get("derotate_lander", False))
    if any(k.startswith("app.") for k in sd):
        app = LanderApp(app_dim).to(dev)
        with torch.no_grad():
            app(torch.zeros(1, 3, CROP, CROP, device=dev))
        app.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("app.")})
        app.eval()
    vae.eval(); branch.eval()
    return dict(vae=vae, branch=branch, app=app, app_start=ld - app_dim, derot=derot, has_app=app is not None)


_AX, _BX = 0.01332539, -0.99301862     # pixel-centroid -> world (x, y); R^2 ~ 0.9999, ~0.03 world max err
_AY, _BY = -0.01999307, 1.39820700


def read_xy(frame_chw):
    """World (x, y) READ FROM THE IMAGE: the lander's largest purple blob centroid mapped to world units.
    The label-free stand-in for the state (x, y); agrees with the state to ~0.03 world (R^2 ~ 0.9999)."""
    mask = purple_mask(frame_chw.cpu()).numpy()
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    big = lab == (1 + int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1)))))
    ys, xs = np.where(big)
    return _AX * xs.mean() + _BX, _AY * ys.mean() + _BY


@torch.no_grad()
def build_z(m, scene, crop, st, app_crop=None, frame=None):
    """Factored latent. If `frame` (the original, lander-containing frames) is given, x and y are READ from
    the image via read_xy (the label-free inference path); otherwise they come from the state `st`. Tilt is
    always read by the branch; the scene latent is the scene encode."""
    mu, _ = m["vae"].encode(scene); z = mu.clone()
    if frame is not None:
        for i in range(frame.size(0)):
            z[i, 0], z[i, 1] = read_xy(frame[i])
    else:
        z[:, 0] = st[:, config.X]; z[:, 1] = st[:, config.Y]
    z[:, 2:4] = m["branch"](crop)
    if m["has_app"]:
        ac = crop if app_crop is None else app_crop
        ac = derotate_crop(ac, st[:, config.TH]) if m["derot"] else ac
        z[:, m["app_start"]:] = m["app"](ac)
    return z


@torch.no_grad()
def encode_frame(m, frame):
    """The one-call IMAGE -> LATENT path: frames (B, 3, H, W) float in [0, 1] -> z (B, latent_dim).
    Label-free (no state needed): x and y come from the purple-blob centroid (read_xy), tilt from the
    branch on a centroid-centred crop, and the scene code from encoding the lander-erased frame. The
    crop is built exactly as in training (train_clean_vae.preload). Frames must contain a visible
    lander; raises ValueError otherwise. For the shipped factored model (no appearance head)."""
    B, _, H, W = frame.shape
    dev = next(m["vae"].parameters()).device
    frame = frame.to(dev)
    crops, scenes = [], []
    for i in range(B):
        comp = TC._largest(purple_mask(frame[i].cpu()).numpy())
        if comp is None:
            raise ValueError(f"frame {i}: no lander (purple blob) found; cannot read the pose")
        ys, xs = np.where(comp)
        cx, cy = xs.mean(), ys.mean()
        x0 = int(np.clip(round(cx - CROP / 2), 0, W - CROP))
        y0 = int(np.clip(round(cy - CROP / 2), 0, H - CROP))
        crops.append(frame[i, :, y0:y0 + CROP, x0:x0 + CROP])
        scenes.append(factored_data.scene_only(frame[i].cpu())[0])
    return build_z(m, torch.stack(scenes).to(dev), torch.stack(crops), None, frame=frame)


def zoom(img_hwc, mask, half=16):
    if mask.sum() == 0:
        return img_hwc
    ys, xs = np.where(mask); cy, cx = int(ys.mean()), int(xs.mean())
    H, W = img_hwc.shape[:2]
    y0 = np.clip(cy - half, 0, max(0, H - 2 * half)); x0 = np.clip(cx - half, 0, max(0, W - 2 * half))
    return img_hwc[y0:y0 + 2 * half, x0:x0 + 2 * half]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zname", default="zlander_inspect1_raw_best")
    ap.add_argument("--baseline", default="factored_vae_staged_best")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if (a.device != "cpu" and torch.cuda.is_available()) else "cpu")
    tag = a.zname.replace("_best", "")

    mz = load(a.zname, dev); mb = load(a.baseline, dev)
    teI, teC, teS, teP, _ = TC.preload(config.TEST_DIR, 30)
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(teI.size(0), size=min(a.n, teI.size(0)), replace=False)
    th_deg = np.degrees(teS[idx, config.TH].numpy()); idx = idx[np.argsort(th_deg)]   # sort by θ (degrees)
    frame = teI[idx].to(dev).float() / 255.; crop = teC[idx].to(dev).float() / 255.; st = teS[idx]
    scene = torch.stack([factored_data.scene_only(frame[i].cpu())[0] for i in range(len(idx))]).to(dev)

    with torch.no_grad():
        dec_b = mb["vae"].decode(build_z(mb, scene, crop, st)).cpu()
        dec_z = mz["vae"].decode(build_z(mz, scene, crop, st)).cpu()
    real = frame.cpu()

    # ---- top block: REAL / factored_staged / zlander, full frame + zoomed lander strip ----
    rows = [("REAL", real), ("factored_staged", dec_b), ("zlander", dec_z)]
    n = len(idx)
    fig, ax = plt.subplots(6, n, figsize=(2.0 * n, 12.0))
    for r, (lab, imgs) in enumerate(rows):
        for j in range(n):
            im = np.clip(imgs[j].permute(1, 2, 0).numpy(), 0, 1)
            ax[r, j].imshow(im); ax[r, j].axis("off")
            if r == 0:
                ax[r, j].set_title(f"θ={np.degrees(float(st[j, config.TH])):+.0f}°", fontsize=8)
            m = purple_mask(imgs[j]).numpy().astype(bool)
            ax[r + 3, j].imshow(np.clip(zoom(im, m), 0, 1)); ax[r + 3, j].axis("off")
        ax[r, 0].text(-0.18, 0.5, lab, rotation=90, va="center", ha="right", transform=ax[r, 0].transAxes, fontsize=10)
        ax[r + 3, 0].text(-0.18, 0.5, lab + "\n(zoom)", rotation=90, va="center", ha="right",
                          transform=ax[r + 3, 0].transAxes, fontsize=9)
    fig.suptitle("z_lander crispness — REAL / factored_staged (blob) / zlander, random frames sorted by θ.\n"
                 "Top 3 rows = full frame; bottom 3 = zoomed lander (judge silhouette + LEGS at 15px).", fontsize=11)
    out = config.fig_dir("factored")
    p1 = os.path.join(out, f"zlander_crispness_{tag}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(p1, dpi=120, bbox_inches="tight"); plt.close(fig)

    saved = [p1]
    # ---- render-vs-paste: fix one frame's scene+pose, swap in z_lander from OTHERS ----
    if mz["has_app"]:
        base_i = 0; donors = list(range(1, min(n, 6)))
        sc0 = scene[base_i:base_i + 1]; cr0 = crop[base_i:base_i + 1]; st0 = st[base_i:base_i + 1]
        panels = []
        with torch.no_grad():
            for di in [base_i] + donors:
                z = build_z(mz, sc0, cr0, st0, app_crop=crop[di:di + 1])
                panels.append((di, mz["vae"].decode(z)[0].cpu()))
        fig2, ax2 = plt.subplots(2, len(panels), figsize=(2.0 * len(panels), 4.4))
        for c, (di, img) in enumerate(panels):
            im = np.clip(img.permute(1, 2, 0).numpy(), 0, 1)
            ax2[0, c].imshow(im); ax2[0, c].axis("off")
            ax2[0, c].set_title("own z_lander" if di == base_i else f"z_lander from #{di}", fontsize=8)
            m = purple_mask(img).numpy().astype(bool)
            ax2[1, c].imshow(np.clip(zoom(im, m), 0, 1)); ax2[1, c].axis("off")
        ax2[1, 0].text(-0.2, 0.5, "zoom", rotation=90, va="center", ha="right", transform=ax2[1, 0].transAxes, fontsize=9)
        fig2.suptitle("RENDER-vs-PASTE: same scene+pose, z_lander swapped from other frames. If the lander "
                      "CHANGES, the decoder RENDERS from the appearance code (transferable); if identical, it pastes a fixed template.", fontsize=9)
        p2 = os.path.join(out, f"zlander_render_vs_paste_{tag}.png")
        plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(p2, dpi=120, bbox_inches="tight"); plt.close(fig2); saved.append(p2)

    print("saved:\n  " + "\n  ".join(saved), flush=True)


if __name__ == "__main__":
    main()
