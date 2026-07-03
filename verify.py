"""verify.py -- check that a (re)trained VAE checkpoint reproduces the shipped one.

Reproducibility is checked at two levels (see checkpoints.py for the philosophy):

  1. SAME HARDWARE: the .pt file's SHA-256 matches the reference exactly. This is a
     bit-for-bit PASS and only happens on the same GPU / library stack the reference
     was trained on (an RTX 3050 Ti here). Determinism flags make this reproducible.

  2. DIFFERENT HARDWARE: exact bits will differ (GPU/cuDNN numerics), so instead the
     key training metrics recorded in the manifest -- validation reconstruction error,
     position controllability (px), and render rate -- are compared to the reference
     within tolerance. This is the realistic "same RESULTS" check for a teammate on a
     different machine.

Usage:
    python verify.py                       # self-check the shipped checkpoint (should be bit-identical)
    python verify.py factored_reproduce    # compare a retrained checkpoint against the shipped reference

Exit code is 0 on PASS, 1 on FAIL, so it can gate a CI / script.
"""
import json
import os
import sys

import checkpoints

# The shipped model everything is compared against. reproduce.sh trains into a
# DIFFERENT name (factored_reproduce), so this reference is never overwritten.
REFERENCE = "factored_clean_noaug_best"

# Tolerances for the different-hardware metric check. Chosen loose enough to absorb
# GPU/library numeric differences but tight enough to catch a genuinely broken run.
TOL = {
    "val_recon": ("abs", 0.0015),   # reference ~0.0048; a correct rerun lands well within this
    "ctrl_px_x": ("rel", 0.15),     # position controllability (px moved when x is dialed)
    "ctrl_px_y": ("rel", 0.15),
    "ctrl_render": ("min", 95.0),   # % of commanded poses that render a lander (reference 100)
}


def _manifest(name):
    _, json_path = checkpoints._paths(name)
    with open(json_path) as f:
        return json.load(f)


def _check_metric(key, ref, cand):
    """Return (ok, message) for one metric under its tolerance rule."""
    if key not in ref or key not in cand:
        return None, f"  {key}: missing (ref={ref.get(key)}, cand={cand.get(key)})"
    r, c = float(ref[key]), float(cand[key])
    kind, tol = TOL[key]
    if kind == "abs":
        ok = abs(c - r) <= tol
        detail = f"|Δ|={abs(c - r):.4g} <= {tol}"
    elif kind == "rel":
        ok = abs(c - r) <= tol * abs(r) if r else c == 0
        detail = f"|Δ|/ref={abs(c - r) / abs(r):.2%} <= {tol:.0%}" if r else "ref=0"
    else:  # "min"
        ok = c >= tol
        detail = f"{c} >= {tol}"
    return ok, f"  {'PASS' if ok else 'FAIL'}  {key}: ref={r:.4g} cand={c:.4g}  ({detail})"


def main():
    candidate = sys.argv[1] if len(sys.argv) > 1 else REFERENCE
    ref_pt, ref_json = checkpoints._paths(REFERENCE)
    cand_pt, cand_json = checkpoints._paths(candidate)

    for label, path in [("reference", ref_json), ("candidate", cand_json)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} manifest not found: {path}")
            return 1

    ref, cand = _manifest(REFERENCE), _manifest(candidate)
    print(f"reference : {REFERENCE}")
    print(f"candidate : {candidate}\n")

    # --- Level 1: exact bit check (same hardware) ---
    ref_sha = checkpoints.sha256_file(ref_pt)
    cand_sha = checkpoints.sha256_file(cand_pt)
    if ref_sha == cand_sha:
        print("BIT-IDENTICAL: candidate .pt SHA-256 matches the reference exactly.")
        print("PASS (bit-for-bit; same hardware and library stack).")
        return 0
    print("Not bit-identical (expected on different hardware); checking metrics within tolerance.\n")

    ref_dev = ref.get("versions", {}).get("device", "?")
    cand_dev = cand.get("versions", {}).get("device", "?")
    print(f"reference device: {ref_dev}\ncandidate device: {cand_dev}\n")

    # --- Level 2: metrics within tolerance (different hardware) ---
    ref_m, cand_m = ref.get("metrics", {}), cand.get("metrics", {})
    results = []
    for key in TOL:
        ok, msg = _check_metric(key, ref_m, cand_m)
        print(msg)
        results.append(ok)
    # gated must be True (the shipped model is controllability-gated)
    gated_ok = bool(cand_m.get("gated", False))
    print(f"  {'PASS' if gated_ok else 'FAIL'}  gated: {cand_m.get('gated')}")
    results.append(gated_ok)

    if any(r is None for r in results):
        print("\nFAIL: a metric was missing from a manifest.")
        return 1
    if all(results):
        print("\nPASS (metrics within tolerance; reproduced the shipped result on different hardware).")
        return 0
    print("\nFAIL: one or more metrics fell outside tolerance.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
