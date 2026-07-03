"""Training-free geometric tilt reader — the 'rhombus' judge.

Reads lander tilt straight from its purple segmentation mask: no network, so no
train/test domain shift. It works identically on REAL and DECODED frames (it's a
function of pixels, not of a learned distribution), which makes it the independent
cross-check the CNN reader needs — and, being transparent, a natural way to read
a physical quantity straight off a generated image for downstream checking.

Method:
  axis : principal axis from the mask's 2nd-order central moments (= PCA of the
         pixel cloud). Gives the tilt line but with a 180-degree (head/feet)
         ambiguity an ellipse can't resolve.
  head : the lander is a RHOMBUS — narrow head, wide feet (splayed legs) — so the
         mass is asymmetric along the axis. The 3rd-order central moment (skew)
         along the axis points toward the heavier/longer-tail side; that fixes
         which end is the head, killing the 180-degree ambiguity.

Tilt-sign convention (verified): upright = 0; +theta = CCW = top/head leans LEFT;
head-tilts-right = negative. Implemented as theta = atan2(-hx, hy) with the head
unit vector (hx, hy) in MATH coords (x right, y UP).

Two discrete choices remain open: is the head/feet axis the
MAJOR or MINOR principal axis, and does the head sit on the +skew or -skew side?
Rather than guess, `GeomThetaReader.calibrate()` picks the combo that best matches
ground-truth theta on real crops (where theta is known) — the same "validate the
judge against ground truth, then report its noise floor" discipline as the
augmenter's sign gate. The calibrated reader reports its own median circular error.
"""
import math
import os
import sys

import numpy as np
from scipy import ndimage

import config

sys.path.insert(0, config.BASELINE_SRC)
from piwm_model.sprite import purple_mask  # noqa: E402  (torch CHW [0,1] -> bool HxW)

import torch  # noqa: E402


# ---------------------------------------------------------------- segmentation
def mask_from_image(img):
    """Largest purple connected component of an image. Accepts HxWx3 or 3xHxW,
    uint8 or float. Returns a bool HxW mask (or None if no lander found)."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] == 3:        # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    chw = torch.as_tensor(arr, dtype=torch.float32).permute(2, 0, 1)
    if float(chw.max()) > 1.5:
        chw = chw / 255.0
    m = purple_mask(chw).numpy()
    return largest_component(m)


def largest_component(mask):
    """Keep only the biggest blob — drops flame specks / stray purple pixels."""
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return None
    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    return labels == (1 + int(np.argmax(sizes)))


# ---------------------------------------------------------------- moments
def _math_coords(mask):
    """Centered pixel cloud in MATH coords (x right, y UP). Returns (mx, my)."""
    ys, xs = np.where(mask)
    mx = xs.astype(np.float64) - xs.mean()
    my = -(ys.astype(np.float64) - ys.mean())   # flip: image row grows downward
    return mx, my


def principal_axes(mask):
    """Return (major_unit, minor_unit, aspect) from 2nd-order central moments.
    `aspect` = sqrt(lambda_major/lambda_minor) — ~1 means near-square (axis unstable)."""
    mx, my = _math_coords(mask)
    cov = np.cov(np.stack([mx, my]))
    vals, vecs = np.linalg.eigh(cov)            # ascending eigenvalues
    minor, major = vecs[:, 0], vecs[:, 1]
    aspect = math.sqrt(max(vals[1], 1e-9) / max(vals[0], 1e-9))
    return major, minor, aspect


def _theta_from_head(head_unit):
    """Head unit vector (math coords) -> tilt in radians under the verified sign."""
    hx, hy = float(head_unit[0]), float(head_unit[1])
    return math.atan2(-hx, hy)                  # upright (0,1) -> 0; +CCW = head leans left


def measure(mask, axis="major", head_skew_sign=1):
    """Geometric tilt of a single mask. `axis` in {major, minor}; `head_skew_sign`
    is +1 if the head sits on the +skew side of the axis, -1 otherwise.
    Returns dict(theta, aspect, npix) — theta in radians (or None if unmeasurable)."""
    mask = largest_component(mask)
    if mask is None or mask.sum() < 8:
        return dict(theta=None, aspect=float("nan"), npix=0 if mask is None else int(mask.sum()))
    major, minor, aspect = principal_axes(mask)
    e = major if axis == "major" else minor
    mx, my = _math_coords(mask)
    proj = mx * e[0] + my * e[1]
    skew = float(np.mean(proj ** 3))
    s = np.sign(skew) if skew != 0 else 1.0
    head = e * s * head_skew_sign               # orient axis toward the head end
    return dict(theta=_theta_from_head(head), aspect=aspect, npix=int(mask.sum()))


# ---------------------------------------------------------------- circular error
def circ_err_deg(pred, true):
    """Circular angular error in degrees, robust to the +-180 seam."""
    return abs(math.degrees(math.atan2(math.sin(pred - true), math.cos(pred - true))))


# ---------------------------------------------------------------- the reader
class GeomThetaReader:
    """Calibrated geometric tilt reader: mask -> theta (radians).

    After `calibrate(masks, thetas)` it has chosen (axis, head_skew_sign) by lowest
    median circular error on real crops, and stored that error as its noise floor.
    """

    def __init__(self, axis="major", head_skew_sign=1):
        self.axis = axis
        self.head_skew_sign = head_skew_sign
        self.floor_deg = None
        self.calibration = None

    def __call__(self, mask):
        return measure(mask, self.axis, self.head_skew_sign)["theta"]

    def calibrate(self, masks, thetas, max_aspect=None):
        """Pick (axis, head_skew_sign) minimizing median circular error vs known
        theta. `masks` are bool HxW; `thetas` are true tilts (radians)."""
        thetas = np.asarray([float(t) for t in thetas])
        results = []
        for axis in ("major", "minor"):
            for hs in (1, -1):
                errs = []
                for mk, th in zip(masks, thetas):
                    out = measure(mk, axis, hs)
                    if out["theta"] is None:
                        continue
                    if max_aspect is not None and out["aspect"] > max_aspect:
                        continue
                    errs.append(circ_err_deg(out["theta"], th))
                if errs:
                    results.append((float(np.median(errs)), axis, hs, len(errs)))
        results.sort()
        best = results[0]
        self.floor_deg, self.axis, self.head_skew_sign, _ = best
        self.calibration = dict(
            chosen=dict(axis=self.axis, head_skew_sign=self.head_skew_sign,
                        median_err_deg=round(self.floor_deg, 2)),
            all=[dict(axis=a, head_skew_sign=h, median_err_deg=round(e, 2), n=n)
                 for e, a, h, n in results])
        return self.calibration


# ---------------------------------------------------------------- self-test / report
def calibrate_on_real(train_files=60, max_files_eval=20, min_pix=12, seed=config.SEED):
    """Build masks+true-theta from REAL frames, calibrate, and report accuracy.
    Ground truth is available on real frames, so this validates the judge fully
    decoder-free. Returns (reader, info)."""
    import glob
    from piwm_model.data import lander_fully_visible
    config.set_seed(seed)
    files = sorted(glob.glob(os.path.join(config.TRAIN_DIR, "*.npz")))
    np.random.default_rng(seed).shuffle(files)   # random episodes, not first-N (correlated)
    fit_masks, fit_th, ev_masks, ev_th = [], [], [], []
    for fi, path in enumerate(files[: train_files + max_files_eval]):
        with np.load(path) as d:
            imgs, st = d["imgs"], d["states"]
        for t in range(len(imgs)):
            if not lander_fully_visible(imgs[t]):
                continue
            mk = mask_from_image(imgs[t])
            if mk is None or mk.sum() < min_pix:
                continue
            (fit_masks if fi < train_files else ev_masks).append(mk)
            (fit_th if fi < train_files else ev_th).append(float(st[t, config.TH]))
    reader = GeomThetaReader()
    cal = reader.calibrate(fit_masks, fit_th)
    # held-out accuracy, and the same broken out by tilt magnitude
    errs = np.array([circ_err_deg(reader(mk), th) for mk, th in zip(ev_masks, ev_th)])
    abdeg = np.degrees(np.abs(ev_th))
    bands = {"|th|<=15": abdeg <= 15, "15-45": (abdeg > 15) & (abdeg <= 45), ">45": abdeg > 45}
    info = dict(
        calibration=cal,
        n_fit=len(fit_masks), n_eval=len(ev_masks),
        median_err_deg=round(float(np.median(errs)), 2),
        mean_err_deg=round(float(np.mean(errs)), 2),
        by_band={k: (round(float(np.median(errs[m])), 2), int(m.sum()))
                 for k, m in bands.items() if m.sum()})
    return reader, info


if __name__ == "__main__":
    reader, info = calibrate_on_real()
    print("chosen:", info["calibration"]["chosen"])
    print("candidates:", info["calibration"]["all"])
    print(f"held-out: n={info['n_eval']}  median={info['median_err_deg']} deg  "
          f"mean={info['mean_err_deg']} deg")
    print("by tilt band (median deg, n):", info["by_band"])
