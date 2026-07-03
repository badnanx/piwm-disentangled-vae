# Factored VAE for Lunar Lander

A small, physically interpretable VAE for Gymnasium Lunar Lander images. Its latent is split into a
physical pose part you set directly (the lander's x, y, and tilt) and a scene code that carries
the terrain but not the lander. That means the lander's pose can be dialed independently, and the model can
be checked on the image it produces rather than trusted blindly.

This repo is the clean, self-contained path from data → trained VAE → using it. The full write-up is
[`docs/vae_report.pdf`](docs/vae_report.pdf).

## Interactive demo

[`pose_demo.html`](pose_demo.html) is a self-contained browser demo: open it in any browser and drag the
x / y / θ sliders to move and rotate the lander. Nothing to install. (On GitHub, download it and open
locally, or view it live at the GitHub Pages URL if Pages is enabled for this repo.)

## Results at a glance

Reconstruction on held-out test frames (real on top, the model's decode on the bottom):

![reconstruction](figures/vae/factored_recon.png)

Dialing one pose axis moves only that axis on the decoded image, with at most about a pixel of leak onto the
others:

![controllability](figures/factored/crosstalk_xy_factored_clean_noaug.png)

Tilt is controllable to about ±45°; beyond that the lander distorts rather than rotating. Full analysis and
limitations are in [`docs/vae_report.pdf`](docs/vae_report.pdf).

## What's here

```
READ THESE
  docs/vae_report.pdf       the write-up (architecture, training, analysis)
  docs/TRAINING.md          how the VAE was trained: the 3-stage chain, loss, commands
  pose_demo.html            interactive slider demo (open in a browser)
  03_vae_and_sincos.ipynb   notebook that explains the VAE and reconstructs test frames

RUN THESE
  example_use.py            load the shipped VAE and use it (reconstruct + generate)
  reproduce.sh              retrain from scratch (data -> stage 1 -> 2 -> 3), then verify
  verify.py                 check a retrained checkpoint reproduces the shipped one
  generate_data.py          regenerate equivalent Lunar Lander data (if you don't have it)
  build_pose_demo_html.py   rebuild pose_demo.html (e.g. from a reproduced checkpoint)

THE CODE
  train_theta_branch_vae.py   stage 1 trainer (base VAE + tilt reader)
  train_position_equiv.py     stage 2 trainer (position control)
  train_factored_vae.py       stage 3 trainer (the shipped, factored model)
  train_clean_vae.py  factored_data.py  zlander_recon_fig.py   model + data loaders
  config.py  checkpoints.py  controllability.py  geom_theta.py  lander_app.py   supporting code
  piwm_model/                 the model, mask, and data helpers (see Attribution)

DATA
  checkpoints/                the trained weights (factored_clean_noaug_best.pt) + manifest
  figures/                    figures used in the report and notebook
```

## Setup

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## The data

The dataset was provided by the team (not generated in this project), so two ways to get data to run
against:

- **You have the dataset (the team):** set `PIWM_DATA_ROOT` to a folder containing `lunartrain/` and
  `lunartest/` of `<i>.npz` episodes (keys `imgs`, `acts`, `states`). No manual filtering needed: the
  training code filters to fully-visible frames and carves the validation split by itself, so the same raw
  episodes give the same training set.
- **You don't have it (anyone else):** `pip install "gymnasium[box2d]"`, then
  `python generate_data.py --n_train 345 --n_test 55 --out ./data/lunar` and set
  `PIWM_DATA_ROOT=./data/lunar`. This makes the same folder structure with random-action episodes.

## Use it

**Just run it** (the fastest way to see it work: reconstruct real frames and generate from a chosen pose):

```bash
python example_use.py
```

### Use the shipped weights in your own project

One `.pt` ships all of it: the VAE (encoder and decoder) and the CNN tilt reader. It holds the weights, not
the model classes, so the code in this repo has to travel with them.

**Simplest: clone this repo** and load with one call. Everything the loader needs is already present:

```python
import torch
from zlander_recon_fig import load, encode_frame   # run from the repo root, or put it on PYTHONPATH
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = load("factored_clean_noaug_best", dev)  # builds the VAE + tilt reader, loads the weights
vae = m["vae"]                              # m["branch"] is the CNN tilt reader

frames = ...                                # your images: (B, 3, 100, 150) float in [0, 1], lander visible
z = encode_frame(m, frames)                 # image -> latent (label-free; pose read off the image)
img = vae.decode(z).clamp(0, 1)             # latent -> image
```

**Just the decoder, dropped into your own tree:** if you only want to turn a latent into an image and would
rather not carry the whole repo, copy three things: the `piwm_model/` package, `checkpoints.py`, and
`config.py`, plus `checkpoints/factored_clean_noaug_best.pt` and its `.json`. Then load the VAE directly:

```python
import torch, math, checkpoints
from piwm_model.autoencoder import PiwmConvVAE
sd, manifest = checkpoints.load_checkpoint("factored_clean_noaug_best")
vae = PiwmConvVAE(manifest["extra"]["config"]["latent_dim"])
vae.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("vae.")})
vae.eval()

# GENERATE from a chosen pose (x, y, tilt): build the 32-dim latent, then decode
z = torch.zeros(1, 32)
z[0, 0], z[0, 1] = 0.0, 0.6                  # x, y  (world units)
z[0, 2] = math.cos(math.radians(20))         # tilt = 20 degrees, stored as (cos, sin)
z[0, 3] = math.sin(math.radians(20))
# z[0, 4:] is the scene code (terrain): zeros gives a generic scene,
# or copy z[4:] from encoding a real frame to reuse that frame's terrain.
img = vae.decode(z)[0].clamp(0, 1)           # (3, 100, 150) image, values in [0, 1]
```

Two things you need to drive it correctly:
- **Latent layout:** `z[0:2]` = (x, y), `z[2:4]` = (cos θ, sin θ), `z[4:]` = the scene code.
- **Pose is injected, not encoded.** At inference, x and y are read off the image (the lander's centroid
  mapped to world units) and tilt from the small CNN reader; the encoder itself only produces the scene code.
  `encode_frame()` runs that whole pipeline (erase the lander, encode the scene, inject the pose) in one
  call, label-free, but it needs the cloned repo, not the minimal copy. `example_use.py` demos both
  directions. Background in `docs/vae_report.pdf` and `docs/TRAINING.md`.

## Reproduce and verify

With `PIWM_DATA_ROOT` set:

```bash
bash reproduce.sh                     # trains stage 1 -> 2 -> 3 from scratch, then verifies
python verify.py factored_reproduce_best   # (reproduce.sh calls this for you)
```

`reproduce.sh` saves the retrained model as `factored_reproduce` so it never overwrites the shipped weights.
A full run takes hours on a laptop GPU; `SMOKE=1 bash reproduce.sh` is a fast end-to-end plumbing check.
`verify.py` passes when the retrain matches the shipped model (bit-for-bit on the same GPU, or key metrics within tolerance
on different hardware), and it flags a genuinely broken run. Full details in
[`docs/TRAINING.md`](docs/TRAINING.md).

To see a reproduction work interactively, rebuild the demo from your checkpoint and drag the sliders:

```bash
PIWM_MODEL=factored_reproduce_best python build_pose_demo_html.py   # writes pose_demo_factored_reproduce_best.html
```

## Reproducibility

`config.SEED = 0` seeds Python, NumPy, and Torch; `checkpoints.enable_determinism()` sets deterministic
cuDNN / cuBLAS flags. Runs reproduce bit-for-bit on the same machine; across GPUs, `verify.py` checks results
within tolerance. Validation is split off by whole episode; the test split is never used for selection.

## Attribution

- The vendored `piwm_model/` package and the VAE design build on the 4-Principles physically interpretable
  world model (arXiv:2503.02143).
- Environment: Gymnasium `LunarLander-v3` (Box2D).
