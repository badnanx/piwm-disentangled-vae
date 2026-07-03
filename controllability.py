"""Controllability: dial a physical value into the latent, decode, and check the image shows it.

Reconstruction asks "can the VAE round-trip a real frame?". Controllability asks the
generative question: if we *dial* a physical value into the latent and decode, does the
**image** actually show it? This is the read-out-vs-control distinction — the encoder can
write x,y,θ into z perfectly while the decoder ignores them (flat sweep = the bug). We judge
on the decoded image with the validated judges (centroid for position, geom_theta for tilt),
never on the latent.

Primitives:
  base_latents      : encode a few real frames -> base codes to perturb.
  sweep_position    : hold a base code, dial z[0] (x) or z[1] (y) across the data range,
                      decode, measure the rendered lander centroid (px).
  sweep_theta       : dial z[2:4]=(cos α, sin α) across angles, decode, measure θ geometrically.
  controllability_report : slopes/correlations quantifying how well the decoder obeys each dim,
                           plus grids of decoded images for the eyeball figure.

A control "slope" near the ideal (and high correlation) means the dim controls the image; a flat
sweep means the decoder ignores it. For θ the measured-vs-dialed slope tells the SAME under-render
story as NB03's reconstruction headline, but now for *commanded* poses (the generative case).
"""
import math
import os
import sys

import numpy as np
import torch
from scipy import ndimage

import config
import geom_theta

sys.path.insert(0, config.BASELINE_SRC)
from piwm_model.sprite import purple_mask  # noqa: E402

# data ranges (state units) for the supervised position dims
X_RANGE = np.linspace(-0.7, 0.7, 9)
Y_RANGE = np.linspace(0.2, 1.3, 9)
ALPHA_RANGE = np.linspace(-math.pi / 2, math.pi / 2, 9)   # dial tilt across ±90° to expose the fold


def centroid_px(mask):
    """(cx, cy) of the largest purple blob in a bool HxW mask, or (nan, nan)."""
    comp = geom_theta.largest_component(mask)
    if comp is None or comp.sum() < 8:
        return float("nan"), float("nan")
    ys, xs = np.where(comp)
    return float(xs.mean()), float(ys.mean())


@torch.no_grad()
def base_latents(vae, images, device):
    """Encode base frames -> (n,latent) mu codes to perturb."""
    vae.eval()
    mu, _ = vae.encode(images.to(device))
    return mu


@torch.no_grad()
def _decode_dim(vae, base_mu, dim, value, device):
    """Decode every base code with z[dim] overwritten by `value`. Returns (n,3,H,W) cpu."""
    z = base_mu.clone(); z[:, dim] = value
    return vae.decode(z).cpu()


@torch.no_grad()
def sweep_position(vae, base_mu, dim, values, device):
    """Vary z[dim] over `values`; return (measured_px, grid). measured_px[v] = median rendered
    centroid (x if dim==0 else y) over base frames; grid[v] = first base frame's decoded image."""
    axis = 0 if dim == 0 else 1
    measured, grid = [], []
    for v in values:
        dec = _decode_dim(vae, base_mu, dim, float(v), device)
        cs = [centroid_px(purple_mask(dec[i]).numpy())[axis] for i in range(dec.size(0))]
        measured.append(float(np.nanmedian(cs))); grid.append(dec[0])
    return np.array(measured), torch.stack(grid)


@torch.no_grad()
def sweep_theta(vae, base_mu, alphas, device, geom_reader):
    """Dial z[2:4]=(cos α, sin α); return (measured_theta, grid). measured θ via geom_reader."""
    measured, grid = [], []
    for a in alphas:
        z = base_mu.clone()
        z[:, 2] = math.cos(a); z[:, 3] = math.sin(a)
        dec = vae.decode(z).cpu()
        ths = [geom_reader(purple_mask(dec[i]).numpy()) for i in range(dec.size(0))]
        ths = [t for t in ths if t is not None]
        measured.append(float(np.median(ths)) if ths else float("nan")); grid.append(dec[0])
    return np.array(measured), torch.stack(grid)


def _fit(dialed, measured):
    """slope + correlation of measured vs dialed, ignoring NaNs."""
    ok = np.isfinite(measured)
    if ok.sum() < 3:
        return dict(slope=float("nan"), corr=float("nan"), n=int(ok.sum()))
    d, m = np.asarray(dialed)[ok], measured[ok]
    slope = float(np.polyfit(d, m, 1)[0])
    corr = float(np.corrcoef(d, m)[0, 1])
    return dict(slope=round(slope, 3), corr=round(corr, 3), n=int(ok.sum()))


def controllability_report(vae, geom_reader, device, base_images, n_base=12):
    """Dial x, y, θ; measure the decoded image. Returns per-dim dialed/measured arrays,
    fit summaries, and image grids for the figure."""
    base_mu = base_latents(vae, base_images[:n_base], device)
    px, gx = sweep_position(vae, base_mu, 0, X_RANGE, device)
    py, gy = sweep_position(vae, base_mu, 1, Y_RANGE, device)
    th, gt = sweep_theta(vae, base_mu, ALPHA_RANGE, device, geom_reader)
    return dict(
        x=dict(dialed=X_RANGE, measured_px=px, grid=gx, fit=_fit(X_RANGE, px)),
        y=dict(dialed=Y_RANGE, measured_px=py, grid=gy, fit=_fit(Y_RANGE, py)),
        theta=dict(dialed=ALPHA_RANGE, measured=th, grid=gt,
                   # θ control slope on the central band (where the decoder can render tilt)
                   fit=_fit(np.degrees(ALPHA_RANGE)[np.abs(np.degrees(ALPHA_RANGE)) <= 45],
                            th[np.abs(np.degrees(ALPHA_RANGE)) <= 45] * 180 / math.pi
                            if np.isfinite(th).any() else th)),
    )


if __name__ == "__main__":
    # Smoke test: run the sweeps end-to-end against the SHIPPED model (needs the data).
    import glob
    from piwm_model.data import lander_fully_visible
    from zlander_recon_fig import load
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load("factored_clean_noaug_best", torch.device(dev))["vae"]

    imgs = []
    for p in sorted(glob.glob(os.path.join(config.TEST_DIR, "*.npz")))[:4]:
        with np.load(p) as d:
            for t in range(len(d["imgs"])):
                if lander_fully_visible(d["imgs"][t]):
                    imgs.append(d["imgs"][t])
    base = torch.from_numpy(np.stack(imgs[:12])).permute(0, 3, 1, 2).float() / 255.0
    reader, _ = geom_theta.calibrate_on_real(train_files=20, max_files_eval=8)
    rep = controllability_report(vae, reader, dev, base)
    for k in ("x", "y", "theta"):
        print(f"{k:>5}: fit={rep[k]['fit']}  grid={tuple(rep[k]['grid'].shape)}")
    print("OK — sweeps ran against the shipped model")
