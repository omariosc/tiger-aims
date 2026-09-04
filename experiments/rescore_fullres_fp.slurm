#!/bin/bash
#SBATCH --job-name=tiger_rescore_fp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%A_%a.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%A_%a.err
# FULL-RES (downscale=0) official-metric rescore of the FP-suppressed preds vs
# their banked baselines, for the final SUBMITTED nHD+Dice number (official metric
# does NOT downscale). Job ARRAY: one frame-parallel scorer per pred-dir so each
# gets the full 10h wall (prior single-job rescore 5998859 TIMED OUT at 2h).
# Frame-level multiprocessing (--workers) splits the O(n^2) full-res nHD across CPUs.
set -e
SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
EX=/users/USERNAME/Tiger-SQ-AI/experiments
NN=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet
GT_T1=$NN/nnUNet_raw/Dataset201_TigerT1coarse/labelsTr
GT_T2=$NN/nnUNet_raw/Dataset202_TigerT2fine/labelsTr
VC="center_1_case_14 center_1_case_15"
W=12

case $SLURM_ARRAY_TASK_ID in
  0) TASK=t1; GT=$GT_T1; PRED=$NN/predTs/t1;       OUT=$NN/t1_score_fullres.json ;;       # baseline (before)
  1) TASK=t1; GT=$GT_T1; PRED=$NN/predTs/t1_fp;    OUT=$NN/t1_fp_score_fullres.json ;;     # FP-suppressed (after)
  2) TASK=t2; GT=$GT_T2; PRED=$NN/predTs/t2_ft;    OUT=$NN/t2_ft_score_fullres.json ;;     # baseline (before)
  3) TASK=t2; GT=$GT_T2; PRED=$NN/predTs/t2_ft_fp; OUT=$NN/t2_ft_fp_score_fullres.json ;;  # FP-suppressed (after)
  4) TASK=t2; GT=$GT_T2; PRED=$NN/predTs/t2;       OUT=$NN/t2_score_fullres.json ;;        # t2 baseline-loss ref
esac

echo "ARRAY $SLURM_ARRAY_TASK_ID  task=$TASK pred=$PRED -> $OUT  (workers=$W)"
$SEG/python $EX/score_tiger_seg.py --task $TASK --gt $GT \
    --pred $PRED --downscale 0 --val-cases $VC --workers $W --out $OUT
echo "=== DONE array $SLURM_ARRAY_TASK_ID ==="
