#!/bin/bash
#SBATCH --job-name=tiger_dockersmoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%j.err
# Full-path smoke of the Docker inference.py: REAL nnU-Net predict (both models)
# + id->RGB + seg-cond T3, on a handful of input frames. Proves the predictor API
# + custom Focal-Tversky trainer load end-to-end (de-risks the container run).
set -e
SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
export TIGER_RESULTS=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/nnUNet_results
export TIGER_T3_CKPT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t3/ckpt_segcond_ftpred/best.pt
export TIGER_REPO=/users/USERNAME/Tiger-SQ-AI/challenge_repo
export nnUNet_raw=/tmp/nnraw nnUNet_preprocessed=/tmp/nnprep
export nnUNet_results=$TIGER_RESULTS

# Build a tiny input dir of RAW RGB frames (3 val frames) in the official input format.
IN=/tmp/tiger_in; rm -rf $IN; mkdir -p $IN
SRC=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/synapse/images
for f in center_1_case_14_9 center_1_case_14_8 center_1_case_15_7L; do
  cp $SRC/${f}.png $IN/
done
echo "input frames:"; ls $IN
OUT=/tmp/tiger_full_out; rm -rf $OUT

echo "=== FULL inference (real nnU-Net predict) ==="
$SEG/python /users/USERNAME/Tiger-SQ-AI/docker/inference.py --input $IN --output $OUT

echo "=== output ==="
find $OUT -type f
echo "--- task3.csv ---"; cat $OUT/task3.csv
echo "=== verify task1 RGB no-unknown round-trip ==="
$SEG/python - <<'EOF'
import sys,numpy as np
from pathlib import Path
sys.path.insert(0,"/users/USERNAME/Tiger-SQ-AI/challenge_repo")
from metrics.classes import rgb_mask_to_label_mask as f2i
from PIL import Image
for p in sorted(Path("/tmp/tiger_full_out/task1").glob("*.png")):
    b=f2i(np.array(Image.open(p).convert("RGB")))
    print(f"  {p.name}: shape={b.shape} unknown_px={(b==255).sum()} (must be 0)")
EOF
echo "=== DONE ==="
