#!/bin/bash
#SBATCH --job-name=tiger_nnunet_proxy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/train_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/train_%j.err
# NOTE (TIGER real data): NO #SBATCH mail directives (agent self-monitors).
#   For the REAL flipped thoracic data, swap trainer -> nnUNetTrainerNoMirroring
#   (h-flip/mirror augmentation OFF). CholecSeg8k is NOT flipped, so the proxy
#   uses the standard short trainer.

set -e
SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
export nnUNet_raw=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/nnUNet_raw
export nnUNet_preprocessed=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/nnUNet_preprocessed
export nnUNet_results=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/nnUNet_results
export nnUNet_n_proc_DA=12

echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total --format=csv
$SEG/python -c "import torch; print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

echo "=== TRAIN Dataset101 2d fold 0 (10-epoch capped proxy run) ==="
$SEG/nnUNetv2_train 101 2d 0 -tr nnUNetTrainer_10epochs --npz

echo "=== TRAIN DONE ==="

RAW=$nnUNet_raw/Dataset101_CholecSeg8k
PREDDIR=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/predTs
mkdir -p $PREDDIR

echo "=== PREDICT held-out test set ==="
$SEG/nnUNetv2_predict -i $RAW/imagesTs -o $PREDDIR \
    -d 101 -c 2d -f 0 -tr nnUNetTrainer_10epochs --disable_tta

echo "=== nnU-Net OWN eval (pseudo-Dice per class) on the held-out preds ==="
$SEG/nnUNetv2_evaluate_folder $RAW/labelsTs $PREDDIR \
    -djfile $RAW/dataset.json \
    -pfile $nnUNet_preprocessed/Dataset101_CholecSeg8k/nnUNetPlans.json --chill \
    || echo "(nnUNetv2_evaluate_folder failed; challenge scorer below is the key check)"

echo "=== CHALLENGE SCORER on the held-out preds (weighted Dice + nHD + w3 sub-avg) ==="
$SEG/python /users/USERNAME/Tiger-SQ-AI/experiments/score_proxy_with_challenge_metric.py \
    --gt $RAW/labelsTs --pred $PREDDIR

echo "=== ALL DONE ==="
