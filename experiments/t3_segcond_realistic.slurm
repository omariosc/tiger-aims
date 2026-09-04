#!/bin/bash
#SBATCH --job-name=t3_segcond_real
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%j.err
# Seg-conditioned T3 with REALISTIC predicted T2 masks for the val frames.
# Submit with a dependency on the baseline T2 job so its predTs/t2 preds exist:
#   sbatch --dependency=afterok:<T2_BASELINE_JOBID> t3_segcond_realistic.slurm
# Also re-runs the GT-mask (upper-bound) seg + hybrid for the record.
set -e
SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
export HF_HOME=/scratch/USERNAME/hf_cache
EX=/users/USERNAME/Tiger-SQ-AI/experiments
T2PRED=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/predTs/t2
OUT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t3

echo "=== GT-mask upper bound (seg) ==="
$SEG/python $EX/train_t3_seg_conditioned.py --mode seg --outdir $OUT/ckpt_segcond_gt

if [ -d "$T2PRED" ] && ls $T2PRED/*.png >/dev/null 2>&1; then
  echo "=== REALISTIC: seg-conditioned with PREDICTED T2 masks (val) ==="
  $SEG/python $EX/train_t3_seg_conditioned.py --mode seg \
      --val-pred-dir $T2PRED --outdir $OUT/ckpt_segcond_pred
else
  echo "WARN: no T2 preds at $T2PRED -> realistic run skipped"
fi
echo "=== DONE ==="
