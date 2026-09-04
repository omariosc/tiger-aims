#!/bin/bash
#SBATCH --job-name=tiger_t3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t3/logs/t3_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t3/logs/t3_%j.err
# NO #SBATCH mail (agent self-monitors).
# Task 3: frozen DINOv3 ViT-B/16 (public LVD-1689M) + 14-way sigmoid head.
set -e
SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
export HF_HOME=/scratch/USERNAME/hf_cache
export HF_HUB_OFFLINE=1   # backbone already cached

echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total --format=csv
$SEG/python /users/USERNAME/Tiger-SQ-AI/experiments/train_t3_station_classifier.py \
    --backbone vit_base_patch16_dinov3.lvd1689m \
    --val-cases center_1_case_14 center_1_case_15 \
    --epochs 300 \
    --outdir /scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t3/ckpt
echo "=== T3 DONE ==="
