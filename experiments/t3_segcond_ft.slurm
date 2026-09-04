#!/bin/bash
#SBATCH --job-name=t3_segcond_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%j.err
# Priority 1 (cheap synergy): re-run REALISTIC seg-conditioned T3 on the
# FOCAL-TVERSKY T2 masks (predTs/t2_ft) instead of the baseline T2 masks
# (predTs/t2). FT-T2 is +0.034 w3 Dice better -> its masks should lift T3 > 0.580.
set -e
SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
export HF_HOME=/scratch/USERNAME/hf_cache
EX=/users/USERNAME/Tiger-SQ-AI/experiments
T2FTPRED=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/predTs/t2_ft
OUT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t3

if [ -d "$T2FTPRED" ] && ls $T2FTPRED/*.png >/dev/null 2>&1; then
  echo "=== REALISTIC: seg-conditioned with FOCAL-TVERSKY predicted T2 masks (val) ==="
  $SEG/python $EX/train_t3_seg_conditioned.py --mode seg \
      --val-pred-dir $T2FTPRED --outdir $OUT/ckpt_segcond_ftpred
else
  echo "ERROR: no FT-T2 preds at $T2FTPRED -> aborting"
  exit 1
fi
echo "=== DONE ==="
