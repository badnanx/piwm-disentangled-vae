"""Cached-checkpoint pattern for this repo's trainable models.

Goal: train the REAL model once, then every rerun is instant. A
RETRAIN switch decides between two paths:
  - LOAD    : read finished weights + manifest (instant). The default once weights
              exist, so a stranger who clones the repo runs the notebook in seconds.
  - RETRAIN : seeded train-from-scratch, then save weights + history + a metrics
              manifest. Trigger with `PIWM_RETRAIN=1` or `retrain=True`.

Reproducibility philosophy (see project_bulletproof_report_vision): bit-identical
weights is NOT a realistic ML goal (GPU nondeterminism, library numerics). We ship
(a) the weights + a SHA-256 (same-hardware bit check), (b) a metrics manifest of the
values from our seeded run, and (c) a tolerance check so a retrain is verified to
land *within tolerance* of the shipped metrics — "same RESULTS", not "same bits".
Determinism flags are set for same-machine reproducibility.

The TEST split is never touched here. When a run needs to monitor/early-stop, it
uses a VALIDATION split carved from TRAIN *by episode* (`episode_val_split`), so the
headline test metric stays untouched until the final report.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

import config

CKPT_DIR = os.path.join(config.HERE, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


# ---------------------------------------------------------------- determinism
def enable_determinism(seed: int = config.SEED) -> None:
    """Seed everything and request deterministic kernels (same-machine repro)."""
    config.set_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed by some cuBLAS GEMMs
    try:
        import torch
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------- val split (canonical, by file)
def canonical_file_split(n_files: int, val_frac: float = 0.15, seed: int = config.SEED):
    """Canonical, stage-INVARIANT validation split by FILE (=episode) index.

    EVERY training stage (theta-branch, position-equivariance, factored) must carve its val
    split with this same call — same (n_files, val_frac, seed) — so the SAME episodes are held
    out for epoch-selection across the WHOLE warm-start pipeline. Otherwise a stage's "val"
    episodes may have been trained on by an earlier stage (cross-stage selection leak). lunartest
    stays out of all of this entirely (the only number reported). The baseline scripts replicate
    this exact numpy logic inline (they can't import this module); keep them in sync.

    Returns (train_file_idx, val_file_idx), sorted int arrays over [0, n_files).
    """
    perm = np.random.default_rng(seed).permutation(n_files)
    n_val = max(1, int(round(val_frac * n_files)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


# ---------------------------------------------------------------- val split (by episode; legacy)
def episode_val_split(ep_id, val_frac: float = 0.15, seed: int = config.SEED):
    """Carve a validation split out of TRAIN *by episode* (disjoint from train).

    Returns (train_mask, val_mask): boolean arrays over the frame axis. Splitting by
    episode (not by frame) prevents highly-correlated consecutive frames from
    straddling the train/val boundary — the same discipline as the train/test split.
    """
    ep_id = np.asarray(ep_id)
    eps = np.unique(ep_id)
    perm = np.random.default_rng(seed).permutation(eps)
    n_val = max(1, int(round(val_frac * len(eps))))
    val_eps = set(perm[:n_val].tolist())
    val_mask = np.isin(ep_id, list(val_eps))
    return ~val_mask, val_mask


# ---------------------------------------------------------------- integrity / versions
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> dict:
    import platform
    info = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["device"] = (torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else "cpu")
    except ImportError:
        pass
    return info


# ---------------------------------------------------------------- save / load
def _paths(name: str):
    return os.path.join(CKPT_DIR, name + ".pt"), os.path.join(CKPT_DIR, name + ".json")


def save_checkpoint(name, state_dict, *, history=None, metrics=None, extra=None) -> dict:
    """Write `<name>.pt` (weights) + `<name>.json` (manifest: seed, sha256, metrics,
    per-epoch history, library versions). Returns the manifest."""
    import torch
    pt_path, json_path = _paths(name)
    torch.save(state_dict, pt_path)
    manifest = {
        "name": name,
        "created": datetime.now(timezone.utc).isoformat(),
        "seed": config.SEED,
        "sha256": sha256_file(pt_path),
        "n_params": int(sum(int(np.prod(v.shape)) for v in state_dict.values())),
        "metrics": metrics or {},
        "history": history or {},
        "versions": _versions(),
        "extra": extra or {},
    }
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_checkpoint(name, map_location="cpu"):
    """Return (state_dict, manifest). Sets manifest['_sha256_ok'] from a re-hash."""
    import torch
    pt_path, json_path = _paths(name)
    with open(json_path) as f:
        manifest = json.load(f)
    state = torch.load(pt_path, map_location=map_location)
    manifest["_sha256_ok"] = (sha256_file(pt_path) == manifest.get("sha256"))
    return state, manifest


def should_retrain(name, retrain=None) -> bool:
    """True if RETRAIN is requested (arg or PIWM_RETRAIN env) or no weights exist yet."""
    if retrain is None:
        retrain = os.environ.get("PIWM_RETRAIN", "0").lower() not in ("0", "", "false", "no")
    return bool(retrain) or not os.path.exists(_paths(name)[0])


# ---------------------------------------------------------------- tolerance check
def metrics_within_tolerance(observed: dict, reference: dict, tol):
    """Compare observed vs reference scalar metrics. `tol` is a per-metric dict of
    absolute tolerances, or a single float applied to all. Returns (ok, report)."""
    report, ok = {}, True
    for k, ref in reference.items():
        if k not in observed or not isinstance(ref, (int, float)):
            continue
        obs = float(observed[k])
        t = float(tol.get(k, np.inf) if isinstance(tol, dict) else tol)
        within = abs(obs - ref) <= t
        report[k] = dict(reference=ref, observed=obs, abs_diff=abs(obs - ref), tol=t, ok=within)
        ok = ok and within
    return ok, report


# ---------------------------------------------------------------- the one-call helper
def load_or_train(name, train_fn, *, retrain=None, tol=None, map_location="cpu"):
    """Load a cached checkpoint, or train-from-scratch and cache it.

    `train_fn()` is called only on the retrain path and must return a dict:
        {"state_dict": ..., "history": {...}, "metrics": {...}, "extra": {...}}.
    On retrain, if a prior manifest exists and `tol` is given, the new metrics are
    checked against the shipped reference (the reproducibility assert). Returns
    (state_dict, manifest); manifest['_status'] is 'loaded' or 'trained'.
    """
    if not should_retrain(name, retrain):
        state, manifest = load_checkpoint(name, map_location)
        manifest["_status"] = "loaded"
        return state, manifest

    # reproducibility reference = the metrics already on disk (if any), read first
    _, json_path = _paths(name)
    reference = None
    if os.path.exists(json_path):
        with open(json_path) as f:
            reference = json.load(f).get("metrics")

    enable_determinism()
    out = train_fn()
    manifest = save_checkpoint(name, out["state_dict"], history=out.get("history"),
                               metrics=out.get("metrics"), extra=out.get("extra"))
    manifest["_status"] = "trained"
    if reference and tol is not None:
        ok, rep = metrics_within_tolerance(out.get("metrics", {}), reference, tol)
        manifest["_repro_ok"] = ok
        manifest["_repro_report"] = rep
    return out["state_dict"], manifest


if __name__ == "__main__":
    # Smoke test: determinism, episode split disjointness, save/load round-trip.
    enable_determinism()
    ep_id = np.repeat(np.arange(20), 30)          # 20 synthetic episodes x 30 frames
    tr, va = episode_val_split(ep_id, val_frac=0.15)
    tr_eps, va_eps = set(ep_id[tr]), set(ep_id[va])
    print(f"episode val split: {tr.sum()} train / {va.sum()} val frames; "
          f"{len(tr_eps)}/{len(va_eps)} episodes; overlap={len(tr_eps & va_eps)}")
    assert not (tr_eps & va_eps), "val split leaks episodes into train!"

    try:
        import torch
        sd = {"w": torch.randn(4, 4)}
        m = save_checkpoint("_smoke", sd, history={"loss": [3.0, 2.0, 1.0]},
                            metrics={"final_loss": 1.0})
        s2, m2 = load_checkpoint("_smoke")
        print(f"save/load ok: n_params={m['n_params']} sha_ok={m2['_sha256_ok']} "
              f"retrain_now={should_retrain('_smoke')}")
        ok, rep = metrics_within_tolerance({"final_loss": 1.02}, {"final_loss": 1.0}, {"final_loss": 0.05})
        print(f"tolerance check ok={ok} ({rep['final_loss']['abs_diff']:.3f} <= {rep['final_loss']['tol']})")
        os.remove(_paths("_smoke")[0]); os.remove(_paths("_smoke")[1])
    except ImportError:
        print("(torch not available — skipped save/load smoke)")
