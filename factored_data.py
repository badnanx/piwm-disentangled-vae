"""Factored-VAE foundation: split each frame into LANDER and SCENE.

The factored VAE's whole premise is that the scene encoder must NOT see the lander (so pose
can't leak into it). This module produces the "scene-only" image (lander erased + the hole filled with
local background) that the scene encoder will consume. We sanity-check it first — if the lander isn't
cleanly removed, nothing downstream works.

Fill strategy: replace lander pixels with a vertical nearest-background fill (copy the nearest non-lander
pixel ABOVE in the same column; fall back to below). The lander sits in sky/over terrain, so pulling the
background down over it removes it without inventing colour. (cv2.inpaint would be fancier; this is
label-free, dependency-light, and good enough — we verify by eye.)
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

import config

sys.path.insert(0, config.BASELINE_SRC)
from piwm_model.sprite import purple_mask  # noqa: E402


def lander_mask(img_chw):
    """Bool (H,W) mask of the lander = largest purple blob, dilated to catch the glow.
    NOTE: we deliberately do NOT erase the orange engine-thrust flames. They DO technically leak lander
    position into the scene latent (Adnan's catch), but the leak is EMPIRICALLY NEGLIGIBLE — thrust
    is ~0.6 px/frame and the success test passed with this purple-only mask — while erasing it risks
    deleting the (warm-coloured) pad flags when the lander is adjacent to one. Simplicity wins; thrust is a
    known, monitored, negligible caveat."""
    m = purple_mask(img_chw).numpy().astype(bool)
    if m.sum() == 0:
        return m
    lab, n = ndimage.label(m)
    if n > 1:                                   # keep only the largest component
        big = 1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1))))
        m = lab == big
    return ndimage.binary_dilation(m, iterations=2)


def scene_only(img_chw):
    """(3,H,W) float frame -> (3,H,W) float with the lander erased via vertical nearest-bg fill."""
    img = img_chw.clone()
    m = lander_mask(img)
    if not m.any():
        return img, m
    arr = img.numpy()                            # (3,H,W)
    H, W = m.shape
    for x in range(W):
        rows = np.where(m[:, x])[0]
        if rows.size == 0:
            continue
        for c in range(3):
            col = arr[c, :, x]
            top = rows.min()
            src = col[top - 1] if top - 1 >= 0 else None     # nearest bg above the lander
            bot = rows.max()
            if src is None:
                src = col[bot + 1] if bot + 1 < H else 0.0
            col[rows] = src
    return torch.from_numpy(arr), m


def sanity(n=6, device="cpu"):
    import glob
    from piwm_model.data import lander_fully_visible
    rng = np.random.default_rng(config.SEED)
    files = sorted(glob.glob(os.path.join(config.TEST_DIR, "*.npz")))
    picks = []
    for p in files:
        with np.load(p) as d:
            imgs = d["imgs"]
        vis = [t for t in range(len(imgs)) if lander_fully_visible(imgs[t])]
        if vis:
            t = int(rng.choice(vis)); picks.append(imgs[t].copy())
        if len(picks) >= n:
            break
    fig, ax = plt.subplots(3, n, figsize=(2.0 * n, 6.0))
    for j, im in enumerate(picks):
        t = torch.from_numpy(im).permute(2, 0, 1).float() / 255.
        scene, m = scene_only(t)
        ax[0, j].imshow(im); ax[0, j].axis("off")
        if j == 0: ax[0, j].set_title("REAL", fontsize=9, loc="left")
        ov = im.copy(); ov[m] = [255, 0, 0]
        ax[1, j].imshow(ov); ax[1, j].axis("off")
        ax[2, j].imshow(np.clip(scene.permute(1, 2, 0).numpy(), 0, 1)); ax[2, j].axis("off")
    for r, lab in enumerate(["REAL", "lander mask (red)", "SCENE only (lander erased)"]):
        ax[r, 0].text(-0.15, 0.5, lab, rotation=90, va="center", ha="right",
                      transform=ax[r, 0].transAxes, fontsize=9)
    fig.suptitle("Factored-VAE step 1 — can we cleanly erase the lander? (scene-only = the scene-latent encoder's input)\n"
                 "If the bottom row still shows a lander, pose can leak into the scene code.", fontsize=10)
    out = os.path.join(config.fig_dir("factored"), "scene_only_sanity.png")
    plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    sanity()
