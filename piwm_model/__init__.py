"""Model and data helpers for the factored VAE, vendored so this repo is standalone.

Contains the VAE (autoencoder), the purple-lander mask + soft-mask helpers (sprite), the data loaders
(data), and small training utilities (train_utils). Builds on the 4-Principles physically interpretable
world model (arXiv:2503.02143). This is a trimmed VAE-only subset; the baseline's diffusion and dynamics
modules are not included here.
"""
