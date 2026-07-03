#!/usr/bin/env bash
# reproduce.sh -- retrain the shipped factored VAE from scratch, then verify it.
#
# Runs the three warm-started stages IN THIS REPO and saves stage 3 under a fresh name
# (factored_reproduce), so the shipped checkpoint (factored_clean_noaug_best) is never
# overwritten and stays as the reference for verify.py.
#
#   stage 1  theta-branch VAE            (from scratch)      -> outputs/repro_stage1/model.pth
#   stage 2  position equivariance       (warm-start s1)     -> outputs/repro_stage2/model.pth
#   stage 3  factored scene-only fine-tune (warm-start s2)   -> checkpoints/factored_reproduce{.pt,.json}
#   verify   compare factored_reproduce to the shipped reference
#
# Reproducibility is verified WITHIN TOLERANCE, not bit-for-bit: exact bits only match on the
# same GPU/library stack the reference was trained on (an RTX 3050 Ti). On other hardware,
# verify.py compares the training metrics (val recon, controllability, render rate) instead.
# The shipped weights themselves warm-started from the sibling baseline repo's equivalent
# stage 2; this script reproduces the SAME RESULT self-contained here, checked by verify.py.
#
# Usage:
#   bash reproduce.sh              # full faithful run (hours on a laptop GPU), then verify
#   SMOKE=1 bash reproduce.sh      # tiny run to check the pipeline executes end-to-end
#                                  #   (under-trained: verify's metric check is EXPECTED to fail)
#   PIWM_DATA_ROOT=/path bash reproduce.sh   # point at your LunarLander data
#   PYTHON=.venv/bin/python bash reproduce.sh
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
PY="${PYTHON:-python}"
SEED=0
DATA_ROOT="${PIWM_DATA_ROOT:-$PWD/data/lunar}"    # default = where generate_data.py writes

if [ "${SMOKE:-0}" != "0" ]; then
  echo ">>> SMOKE run: tiny settings, pipeline plumbing check only (metrics will NOT reproduce)."
  TRAIN_FILES=6; S1_EPOCHS=2; S2_EPOCHS=2; S3_EPOCHS=2
else
  echo ">>> FULL run: faithful reproduction (this takes hours on a laptop GPU)."
  TRAIN_FILES=345; S1_EPOCHS=30; S2_EPOCHS=30; S3_EPOCHS=70
fi

echo ">>> data root: $DATA_ROOT"
echo ""

echo "===== STAGE 1: theta-branch VAE (from scratch) ====="
$PY train_theta_branch_vae.py \
  --seed $SEED --grad_clip 0.5 --lr 5e-4 \
  --data_root "$DATA_ROOT" --train_files $TRAIN_FILES --epochs $S1_EPOCHS \
  --output_dir outputs/repro_stage1

echo ""
echo "===== STAGE 2: position equivariance (warm-start from stage 1) ====="
$PY train_position_equiv.py \
  --seed $SEED --per_axis --equiv_weight 5e-4 \
  --init_ckpt outputs/repro_stage1/model.pth \
  --data_root "$DATA_ROOT" --train_files $TRAIN_FILES --epochs $S2_EPOCHS \
  --output_dir outputs/repro_stage2

echo ""
echo "===== STAGE 3: factored scene-only fine-tune (warm-start from stage 2) ====="
$PY train_factored_vae.py \
  --init_ckpt outputs/repro_stage2/model.pth \
  --save_name factored_reproduce \
  --train_files $TRAIN_FILES --epochs $S3_EPOCHS \
  --theta_equiv_weight 1.0 --lr 1e-3 --device cuda

echo ""
echo "===== VERIFY: factored_reproduce vs the shipped reference ====="
$PY verify.py factored_reproduce_best || {
  echo "(if this is a SMOKE run, a metric FAIL here is expected -- it is under-trained.)"
  exit 1
}
echo ">>> reproduce.sh done."
