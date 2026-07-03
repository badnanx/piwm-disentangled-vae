# How the factored VAE was trained

This documents the training behind the shipped `factored_clean_noaug` weights. NB03 §4 gives the loss and
the parameter table in context; this file holds the procedure, the runnable commands, and the data
decisions, kept out of the notebook flow. The actual code is `train_factored_vae.py` (stage 3) and the two
stage scripts it warm-starts from.

## The procedure: a 3-stage warm-start

Each stage initializes from the previous stage's weights and holds out the same episodes for validation
(by episode, via `checkpoints.canonical_file_split`), so no stage selects on data a later stage trains on.
The test split is never used for selection.

1. **Tilt branch** (`train_theta_branch_vae.py`), a small CNN reads (cos θ, sin θ) from a centered lander
   crop, supervised on the true angle.
2. **Position equivariance + concentration** (`train_position_equiv.py`), warm-start from (1); adds a
   per-axis swap-equivariance objective so dialing z[0:2] moves the lander in x and y, plus a concentration
   term to keep the lander compact.
3. **Factored scene-only fine-tune** (`train_factored_vae.py`, the shipped model), warm-start from (2);
   the scene latent z[4:] is encoded from the lander-erased image (so it carries little pose), x and y are
   injected into z[0:2] from the state, and a θ-swap-equivariance term sharpens the tilt dial.

## The loss (stage 3, the shipped model)

From `train_factored_vae.py` (the per-batch assembly):

```
L = recon + 1e-4·KL(scene latent z[4:]) + 1.0·θ_state + w_eq·position_swap + 1.0·θ_swap_equiv
    recon = region-weighted MSE (lander ×25) + 1.0·edge(gradient) + 0.5·SSIM
    θ_state      = MSE on (cos θ, sin θ)            # never raw-angle MSE
    position_swap = per-axis swap-equivariance on z[0:2]      # ramped in over 8 epochs
    θ_swap_equiv  = frozen-branch read-back of a commanded tilt  # ramped in over 8 epochs
```

x and y are injected into z[0:2] from the state, so they are set, not learned by a loss term. The full
parameter values are in the shipped manifest `checkpoints/factored_clean_noaug_best.json` and tabulated in
NB03 §4 (latent dim 32, lr 1e-3, batch 32, early-stopped at epoch 19, val recon 0.0048).

## Running it from scratch

The whole chain is scripted. From the repo root, with `PIWM_DATA_ROOT` pointing at the shared dataset:

```bash
bash reproduce.sh            # stage 1 -> stage 2 -> stage 3, then verify.py
```

It trains the three stages in order (each warm-starting from the previous), saves the final model as
`factored_reproduce` so the shipped `factored_clean_noaug_best` is left intact as the reference, then runs
`verify.py`. A full run takes hours on a laptop GPU; `SMOKE=1 bash reproduce.sh` runs a tiny end-to-end
plumbing check (under-trained, so its metrics will not match, by design).

To run the stages by hand, these are the exact commands `reproduce.sh` uses:

```bash
python train_theta_branch_vae.py --seed 0 --grad_clip 0.5 --lr 5e-4 \
    --train_files 345 --epochs 30 --output_dir outputs/repro_stage1
python train_position_equiv.py --seed 0 --per_axis --equiv_weight 5e-4 \
    --init_ckpt outputs/repro_stage1/model.pth \
    --train_files 345 --epochs 30 --output_dir outputs/repro_stage2
python train_factored_vae.py --init_ckpt outputs/repro_stage2/model.pth \
    --save_name factored_reproduce --train_files 345 --epochs 70 --theta_equiv_weight 1.0 --lr 1e-3
```

Stages 1 and 2 save `model.pth` under `--output_dir`; stage 3 saves `<save_name>_best.pt` under
`checkpoints/`. Caveat: the repo ships the final weights, not the intermediate stage checkpoints, and a
from-scratch rerun reproduces a *functionally equivalent* model, not the exact shipped bytes (a multi-stage
GPU run drifts across hardware). `verify.py` is what confirms the rerun landed on the shipped result.

## Verifying a reproduction

```bash
python verify.py factored_reproduce_best
```

Two levels. On the same GPU and library stack, the `.pt` SHA-256 matches the shipped reference exactly
(bit-for-bit PASS). On different hardware, exact bits differ, so the training metrics (validation
reconstruction, position controllability in px, render rate) are compared to the reference within tolerance.
Because the team runs on the same data, a correct rerun lands well within tolerance even on a different
GPU. `verify.py` exits non-zero on failure and does flag a genuinely broken run: an under-trained model
fails the metric check rather than passing silently.

## Data decisions

- **Filtering:** training uses the fully-visible frames, applied automatically at load time by
  `lander_fully_visible` inside the `preload` functions, so there is no separate filtering step to run: the
  same raw episodes produce the same filtered set. The split is by episode (345 train / 55 test), verified
  disjoint, and carved the same way in every stage via `checkpoints.canonical_file_split` (seed 0).
- **Augmentation: none, and only position was ever tested.** Two augmentation ideas exist; neither is in
  the shipped model, and they differ in how far they were taken:
  - **Rotation augmentation**: demonstrated as a capability and used elsewhere to train the tilt
    *reader* full-circle, but never applied to train any decoder here. The decoder's ±45° tilt cap is
    just real-data scarcity (large tilts are rare); rotation augmentation of the decoder was not tried.
  - **Translate (position) augmentation**: relocating the lander across the frame to give the decoder crisp
    off-centre targets. This was A/B tested against a no-augmentation run
    and showed no meaningful improvement (it did not sharpen off-centre landers, and the no-aug run had
    cleaner legs in the central band), so the shipped model is no-aug. The translate-augmentation code has
    been removed from the trainer.
