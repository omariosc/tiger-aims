#!/bin/bash
#SBATCH --job-name=tiger_rescore_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%j.err
# Resolution audit: re-score the BANKED T1 (baseline) and T2-FT predictions at
# FULL native resolution (downscale=0) -- the OFFICIAL metric does NOT downscale,
# so the submitted score is the full-res one, not our 160px fast-iteration number.
# Tells us whether thin w=3 classes are penalised/helped by the 160px hack.
set -e
SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
EX=/users/USERNAME/Tiger-SQ-AI/experiments
NN=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet
GT_T1=$NN/nnUNet_raw/Dataset201_TigerT1coarse/labelsTr
GT_T2=$NN/nnUNet_raw/Dataset202_TigerT2fine/labelsTr
VC="center_1_case_14 center_1_case_15"

echo "######## T1 baseline FULL-RES ########"
$SEG/python $EX/score_tiger_seg.py --task t1 --gt $GT_T1 \
    --pred $NN/predTs/t1 --downscale 0 --val-cases $VC \
    --out $NN/t1_score_fullres.json

echo "######## T2-FT FULL-RES ########"
$SEG/python $EX/score_tiger_seg.py --task t2 --gt $GT_T2 \
    --pred $NN/predTs/t2_ft --downscale 0 --val-cases $VC \
    --out $NN/t2_ft_score_fullres.json

echo "######## T2 baseline FULL-RES (for the FT vs baseline delta at full res) ########"
$SEG/python $EX/score_tiger_seg.py --task t2 --gt $GT_T2 \
    --pred $NN/predTs/t2 --downscale 0 --val-cases $VC \
    --out $NN/t2_score_fullres.json
echo "=== DONE ==="
