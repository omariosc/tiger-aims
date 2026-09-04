#!/bin/bash
#SBATCH --job-name=tiger_synthseed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/seg_%x_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/seg_%x_%j.err
# NO #SBATCH mail (agent self-monitors).
# TASK A: train one extra synth seed (Dataset203 = real112 + 1158 synth, val=cases
# 14/15 synth-free) on the confirmed FINE recipe (250ep NoMirror FocalTversky), seeded
# init only -> identical schedule, mirroring OFF. After training, predict val (28 frames)
# WITH probabilities into t2_synth_seed${SEED} (npz needed for prob-averaging in TASK B).
# Does NOT touch the existing seed0 results / banked outputs / split.
#
# Usage:  sbatch --job-name=tiger_synthseed1 train_tiger_synth_seeds.slurm 1
#         sbatch --job-name=tiger_synthseed2 train_tiger_synth_seeds.slurm 2
set -u
SEED=${1:?usage: sbatch train_tiger_synth_seeds.slurm <SEED 1 or 2>}

TRAINER=nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed${SEED}
DSID=203

SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
ROOT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet
export nnUNet_raw=$ROOT/nnUNet_raw
export nnUNet_preprocessed=$ROOT/nnUNet_preprocessed
export nnUNet_results=$ROOT/nnUNet_results
export nnUNet_n_proc_DA=12

RAW=$nnUNet_raw/Dataset203_TigerT2fineSynth
SPLIT=$nnUNet_preprocessed/Dataset203_TigerT2fineSynth/splits_final.json
VALIN=$ROOT/predTs/t2_synth_valin            # 84 symlinked val frames (3ch x 28)
OUT=$ROOT/predTs/t2_synth_seed${SEED}

echo "=== TASK A synth seed${SEED}: TRAINER=$TRAINER DSID=$DSID ==="
echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total --format=csv
$SEG/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# split MUST already exist (shared with seed0) -> assert, never re-plan.
if [ ! -f "$SPLIT" ]; then echo "splits_final.json MISSING ($SPLIT) -> ABORT"; exit 1; fi
echo "=== split fold0: $(jq -r '.[0].train|length' $SPLIT) train / $(jq -r '.[0].val|length' $SPLIT) val ==="

# val input dir must exist (built by agent) -> assert 84 files.
if [ ! -d "$VALIN" ]; then echo "VALIN missing ($VALIN) -> ABORT"; exit 1; fi
NIN=$(ls $VALIN/*.png 2>/dev/null | wc -l)
echo "=== VALIN = $NIN frames (expect 84) ==="
if [ "$NIN" -ne 84 ]; then echo "VALIN frame count != 84 -> ABORT"; exit 1; fi

echo "=== TRAIN Dataset${DSID} 2d fold 0 ($TRAINER) ==="
$SEG/nnUNetv2_train ${DSID} 2d 0 -tr ${TRAINER} --npz
TRC=$nnUNet_results/Dataset203_TigerT2fineSynth/${TRAINER}__nnUNetPlans__2d/fold_0/checkpoint_final.pth
if [ ! -f "$TRC" ]; then echo "checkpoint_final.pth MISSING after train ($TRC) -> ABORT"; exit 1; fi
echo "=== checkpoint OK: $TRC ==="

echo "=== PREDICT val (28 frames) WITH probabilities -> $OUT ==="
rm -rf "$OUT"; mkdir -p "$OUT"
$SEG/nnUNetv2_predict -i "$VALIN" -o "$OUT" -d ${DSID} \
  -tr ${TRAINER} -c 2d -f 0 --save_probabilities
NNPZ=$(ls $OUT/*.npz 2>/dev/null | wc -l)
NPNG=$(ls $OUT/*.png 2>/dev/null | wc -l)
echo "=== seed${SEED} preds: $NPNG png / $NNPZ npz (expect 28 each) ==="
if [ "$NNPZ" -ne 28 ]; then echo "npz count != 28 -> WARN (TASK B will assert)"; fi

echo "=== TASK A synth seed${SEED} DONE ==="
