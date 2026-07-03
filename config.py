"""Shared config for the bulletproof PIWM VAE+diffusion report.

Single source of truth for paths and the global random seed so every notebook
and script is reproducible. Override DATA_ROOT via the PIWM_DATA_ROOT env var.
"""
import os
import random

SEED = 0

# Data: Gymnasium LunarLander-v3, discrete actions, default config (no wind).
# Episodes as .npz (keys: imgs, acts, states, noisy_states_*).
ENV_ID = "LunarLander-v3"
DATA_ROOT = os.environ.get(
    "PIWM_DATA_ROOT", "/home/aaljoubi/research/piwm/data/lunar/extracted/lunar")
TRAIN_DIR = os.path.join(DATA_ROOT, "lunartrain")
TEST_DIR = os.path.join(DATA_ROOT, "lunartest")

HERE = os.path.dirname(os.path.abspath(__file__))

# Model/data helpers are VENDORED into this repo as the `piwm_model` package
# (sprite/data/autoencoder/...), so the repo is standalone. Scripts do
# `sys.path.insert(0, config.BASELINE_SRC); from piwm_model... import ...`;
# pointing BASELINE_SRC at the repo root makes that resolve to the local copy.
# BASELINE_SCRIPTS likewise points at the repo root, where the two vendored helper
# scripts (regressor_theta_check.py, prove_sincos_theta.py) live.
BASELINE_SRC = HERE
BASELINE_SCRIPTS = HERE

FIG_DIR = os.path.join(HERE, "figures")


def fig_dir(sub: str = "") -> str:
    """Return (and create) a figures subfolder. Figures are organized by purpose (e.g. vae/, factored/)."""
    import os as _os
    d = _os.path.join(FIG_DIR, sub)
    _os.makedirs(d, exist_ok=True)
    return d

# State vector layout (LunarLander, 8-dim).
STATE_NAMES = ["x", "y", "vx", "vy", "theta", "omega", "leg1_contact", "leg2_contact"]
X, Y, VX, VY, TH, OM, L1, L2 = range(8)

# Discrete action meanings (LunarLander-v3).
ACTION_NAMES = ["noop", "left engine", "main engine", "right engine"]


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy, and torch (if installed) for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
