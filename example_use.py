"""example_use.py -- the shortest path to USING the shipped factored VAE.

Two demos:
  * generate  (needs no data): pick a pose (x, y, tilt) and a scene code, decode an image.
                This shows the decoder as a renderer: latent numbers in, image out.
  * reconstruct (needs the LunarLander data at PIWM_DATA_ROOT): encode a real test frame
                and decode it back.

Run:
    python example_use.py

The latent is 32 numbers, split by meaning:
    z[0]   = x        (world units)          z[2:4] = (cos tilt, sin tilt)
    z[1]   = y        (world units)          z[4:]  = scene code (terrain; NOT the lander)
The lander's pose is z[0:4]; the scene code is everything else. See vae_report.pdf.
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import checkpoints
from zlander_recon_fig import load, build_z          # load(): build the model + load weights

MODEL = "factored_clean_noaug_best"
OUT_DIR = config.fig_dir("vae")


def generate_demo(m, dev, scene_z):
    """Decode the lander at a fixed centre position across a sweep of tilts, on ONE scene code.
    No data needed: this is the pure decoder(latent) -> image path."""
    tilts_deg = [-45, -22, 0, 22, 45]
    x, y = 0.0, 0.55                                   # world units: roughly centred, mid-height
    z = torch.zeros(len(tilts_deg), m["vae"].latent_dim, device=dev)
    for i, t in enumerate(tilts_deg):
        rad = np.radians(t)
        z[i, 0], z[i, 1] = x, y                        # position slots
        z[i, 2], z[i, 3] = np.cos(rad), np.sin(rad)    # tilt as (cos, sin)
        z[i, 4:] = scene_z                             # same terrain for every panel
    with torch.no_grad():
        imgs = m["vae"].decode(z).clamp(0, 1).cpu()

    fig, ax = plt.subplots(1, len(tilts_deg), figsize=(2.2 * len(tilts_deg), 2.4))
    for i, t in enumerate(tilts_deg):
        ax[i].imshow(imgs[i].permute(1, 2, 0).numpy()); ax[i].axis("off")
        ax[i].set_title(f"tilt {t:+d}°", fontsize=10)
    fig.suptitle("generate: decode a chosen pose (x, y, tilt) on one scene code", fontsize=11)
    out = os.path.join(OUT_DIR, "example_generate.png")
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def reconstruct_demo(m, dev):
    """Encode a few real test frames and decode them back. Needs the data at PIWM_DATA_ROOT."""
    import factored_data
    import train_clean_vae as TC
    import sys
    sys.path.insert(0, config.BASELINE_SRC)
    from piwm_model.data import lander_fully_visible

    config.set_seed()
    teI, teC, teS, *_ = TC.preload(config.TEST_DIR, 30)
    vis = [k for k in range(len(teI)) if lander_fully_visible(teI[k].permute(1, 2, 0).numpy())]
    idx = np.random.default_rng(config.SEED).choice(vis, size=5, replace=False)
    fr = teI[idx].to(dev).float() / 255.
    cr = teC[idx].to(dev).float() / 255.
    stb = teS[idx]
    scene = torch.stack([factored_data.scene_only(fr[k].cpu())[0] for k in range(len(idx))]).to(dev)
    with torch.no_grad():
        z = build_z(m, scene, cr, stb, frame=fr)       # x,y read from the image; tilt from the branch
        rec = m["vae"].decode(z).clamp(0, 1).cpu()

    fig, ax = plt.subplots(2, len(idx), figsize=(2.2 * len(idx), 4.4))
    for j in range(len(idx)):
        ax[0, j].imshow(fr[j].cpu().permute(1, 2, 0).numpy()); ax[0, j].axis("off")
        ax[1, j].imshow(rec[j].permute(1, 2, 0).numpy()); ax[1, j].axis("off")
    ax[0, 0].set_ylabel("REAL", fontsize=10); ax[1, 0].set_ylabel("DECODED", fontsize=10)
    fig.suptitle("reconstruct: real test frames (top) vs the model's reconstruction (bottom)", fontsize=11)
    out = os.path.join(OUT_DIR, "example_reconstruction.png")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, manifest = checkpoints.load_checkpoint(MODEL, map_location="cpu")
    print(f"loaded {MODEL}: {manifest['n_params']:,} params, "
          f"SHA-256 {'OK' if manifest.get('_sha256_ok') else 'MISMATCH'}, device={dev}")
    m = load(MODEL, dev)

    # Try the data-backed reconstruction; always run the data-free generate demo.
    try:
        reconstruct_demo(m, dev)
        # reuse a real scene code for the generate demo so the backdrop is a real terrain
        import factored_data, train_clean_vae as TC
        teI, *_ = TC.preload(config.TEST_DIR, 1)
        sc = factored_data.scene_only(teI[0].float() / 255.)[0].unsqueeze(0).to(dev)
        scene_z = m["vae"].encode(sc)[0][0, 4:]
    except Exception as e:
        print(f"(reconstruct demo skipped, no data found: {type(e).__name__}); running generate only")
        scene_z = torch.zeros(m["vae"].latent_dim - 4, device=dev)   # prior-mean scene code
    generate_demo(m, dev, scene_z)
    print("done.")


if __name__ == "__main__":
    main()
