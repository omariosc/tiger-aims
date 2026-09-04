#!/bin/bash
#SBATCH --job-name=tiger_seg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/seg_%x_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/seg_%x_%j.err
# NO #SBATCH mail (agent self-monitors). REAL flipped thoracic view ->
# nnUNetTrainer_250epochs_NoMirroring (h-flip/mirror augmentation OFF).
#
# Usage:  sbatch --job-name=tiger_t1 train_tiger_seg.slurm 201 t1
#         sbatch --job-name=tiger_t2 train_tiger_seg.slurm 202 t2
set -e
DSID=${1:?dataset id (201=T1coarse, 202=T2fine)}
TASK=${2:?task tag (t1|t2)}
TRAINER=${3:-nnUNetTrainer_250epochs_NoMirroring}   # reduced-epoch, flip-safe

SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
ROOT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet
export nnUNet_raw=$ROOT/nnUNet_raw
export nnUNet_preprocessed=$ROOT/nnUNet_preprocessed
export nnUNet_results=$ROOT/nnUNet_results
export nnUNet_n_proc_DA=12

DSNAME=$(basename $(ls -d $nnUNet_raw/Dataset${DSID}_*))
DSNAME=${DSNAME#Dataset${DSID}_}
RAW=$nnUNet_raw/Dataset${DSID}_${DSNAME}

echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total --format=csv
$SEG/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# Preserve our case-disjoint splits_final.json across preprocessing.
SPLIT=$nnUNet_preprocessed/Dataset${DSID}_${DSNAME}/splits_final.json
SPLIT_BAK=/tmp/splits_${DSID}.json
[ -f "$SPLIT" ] && cp "$SPLIT" "$SPLIT_BAK" && echo "backed up case-disjoint split"

echo "=== PLAN + PREPROCESS Dataset${DSID} ==="
$SEG/nnUNetv2_plan_and_preprocess -d ${DSID} --verify_dataset_integrity -c 2d

# restore our case-disjoint split (overwrites nnU-Net's auto 5-fold KFold)
if [ -f "$SPLIT_BAK" ]; then cp "$SPLIT_BAK" "$SPLIT"; echo "restored case-disjoint split (fold 0)"; fi

echo "=== TRAIN Dataset${DSID} 2d fold 0  ($TRAINER) ==="
$SEG/nnUNetv2_train ${DSID} 2d 0 -tr ${TRAINER} --npz

echo "=== PREDICT held-out val (case 14,15) ==="
PREDDIR=$ROOT/predTs/${TASK}
mkdir -p $PREDDIR
# build a val-images dir (only val-case channels) from imagesTr
VALIMG=/tmp/valimg_${TASK}; rm -rf $VALIMG; mkdir -p $VALIMG
for stem in $(jq -r '.[0].val[]' "$SPLIT"); do
  cp $RAW/imagesTr/${stem}_0000.png $RAW/imagesTr/${stem}_0001.png $RAW/imagesTr/${stem}_0002.png $VALIMG/
done
$SEG/nnUNetv2_predict -i $VALIMG -o $PREDDIR -d ${DSID} -c 2d -f 0 \
    -tr ${TRAINER} --disable_tta

echo "=== CHALLENGE SCORER (weighted Dice + nHD + w=3 sub-avg) ==="
$SEG/python /users/USERNAME/Tiger-SQ-AI/experiments/score_tiger_seg.py \
    --task ${TASK} --gt $RAW/labelsTr --pred $PREDDIR \
    --val-cases center_1_case_14 center_1_case_15 --downscale 160 \
    --out $ROOT/${TASK}_score.json || echo "(scorer failed; train ckpt still valid)"

echo "=== ALL DONE ${TASK} ==="
