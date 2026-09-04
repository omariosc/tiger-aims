#!/bin/bash
#SBATCH --job-name=tiger_synthens
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/seg_%x_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/seg_%x_%j.err
# NO #SBATCH mail (agent self-monitors).
# TASK B (afterok seed1+seed2): build the 3-seed synth ensemble + score; then the
# 4-member cross-dataset argmax-graft fusion. Scores EVERYTHING @160px, FP-suppress first.
#   1. RE-PREDICT seed0 WITH probabilities (existing t2_synthmix was argmax-only) -> t2_synth_seed0
#   2. 3-seed synth ensemble: prob-avg {seed0,seed1,seed2} -> t2_synth_ens -> FP -> score160
#   3. 4-member fusion: graft synth's rare-class wins (id 8 = R_Inf_Pulm_Lig, the ONLY
#      data-floor class synth beats champ-ens on @160) onto champion ens -> t2_fuse4 -> FP -> score160
#   4. append verdict lines to AUTONOMOUS_RESULTS.md  (vs champion ens w3+FP = 0.701)
set -u

SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
ROOT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet
EX=/users/USERNAME/Tiger-SQ-AI/experiments
RES=/users/USERNAME/AUTONOMOUS_RESULTS.md
export nnUNet_raw=$ROOT/nnUNet_raw
export nnUNet_preprocessed=$ROOT/nnUNet_preprocessed
export nnUNet_results=$ROOT/nnUNet_results
D=$(date +%F)
JID=${SLURM_JOB_ID:-local}

GT=$nnUNet_raw/Dataset202_TigerT2fine/labelsTr      # real fine GT (28 val files)
VALIN=$ROOT/predTs/t2_synth_valin
FPCFG=$ROOT/t2_ft_ens_fpsuppress_fullres.json
VC="center_1_case_14 center_1_case_15"

echo "=== TASK B synth-ensemble + fuse  JID=$JID  ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

# ---- assert seed1/seed2 checkpoints + npz preds exist (never trust bare COMPLETED) ----
for S in 1 2; do
  TR=nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed${S}
  CK=$nnUNet_results/Dataset203_TigerT2fineSynth/${TR}__nnUNetPlans__2d/fold_0/checkpoint_final.pth
  D1=$ROOT/predTs/t2_synth_seed${S}
  [ -f "$CK" ] || { echo "MISSING seed${S} checkpoint $CK -> ABORT"; exit 1; }
  N=$(ls $D1/*.npz 2>/dev/null | wc -l)
  echo "seed${S}: ck OK, $N npz preds"
  [ "$N" -eq 28 ] || { echo "seed${S} npz count != 28 -> ABORT"; exit 1; }
done

# ---- 1. RE-PREDICT seed0 WITH probabilities (need npz for prob-avg) ----
S0=$ROOT/predTs/t2_synth_seed0
TR0=nnUNetTrainer_250epochs_NoMirror_FocalTversky
echo "=== (1) re-predict seed0 WITH probabilities -> $S0 ==="
rm -rf "$S0"; mkdir -p "$S0"
$SEG/nnUNetv2_predict -i "$VALIN" -o "$S0" -d 203 -tr ${TR0} -c 2d -f 0 --save_probabilities
N0=$(ls $S0/*.npz 2>/dev/null | wc -l)
echo "seed0: $N0 npz"; [ "$N0" -eq 28 ] || { echo "seed0 npz != 28 -> ABORT"; exit 1; }

# ---- 2. 3-seed synth ensemble (prob-avg) ----
ENS=$ROOT/predTs/t2_synth_ens
echo "=== (2) 3-seed synth ensemble (prob-avg) -> $ENS ==="
$SEG/python $EX/ensemble_npz_avg.py \
  --members $S0 $ROOT/predTs/t2_synth_seed1 $ROOT/predTs/t2_synth_seed2 --out $ENS
[ "$(ls $ENS/*.png 2>/dev/null | wc -l)" -eq 28 ] || { echo "ens png != 28 -> ABORT"; exit 1; }

ENSFP=$ROOT/predTs/t2_synth_ens_fp
$SEG/python $EX/apply_fp_suppress.py --cfg $FPCFG --pred $ENS --out $ENSFP

echo "=== score 3-seed synth ensemble (raw + FP) @160 ==="
$SEG/python $EX/score_tiger_seg.py --task t2 --gt $GT --pred $ENS \
  --downscale 160 --val-cases $VC --out $ROOT/t2_synth_ens_score160.json
$SEG/python $EX/score_tiger_seg.py --task t2 --gt $GT --pred $ENSFP \
  --downscale 160 --val-cases $VC --out $ROOT/t2_synth_ens_fp_score160.json

ENS_FP_W3=$($SEG/python -c "import json;print(json.load(open('$ROOT/t2_synth_ens_fp_score160.json'))['w3_dice'])")
ENS_FP_WD=$($SEG/python -c "import json;print(json.load(open('$ROOT/t2_synth_ens_fp_score160.json'))['weighted_dice'])")
echo "=== 3-seed synth ens w3+FP = $ENS_FP_W3  (weighted=$ENS_FP_WD)  vs champ-ens 0.701 ==="
VERD=$($SEG/python -c "print('CONFIRMED-NEW-CHAMPION' if $ENS_FP_W3 > 0.701 else 'REFUTED')")
echo "[Tiger][$JID] $D 3-seed synth-ensemble w3+FP=$ENS_FP_W3 weighted=$ENS_FP_WD vs 0.701 -> $VERD" >> $RES

# ---- 3. 4-member cross-dataset argmax graft (graft id8 onto champion ens) ----
CHAMP=$ROOT/predTs/t2_ft_ens          # champion 3-FINE-seed ensemble argmax masks
SYN=$ROOT/predTs/t2_synthmix          # synth seed0 argmax masks
FUSE=$ROOT/predTs/t2_fuse4
echo "=== (3) graft synth id8 (R_Inf_Pulm_Lig, only data-floor class synth>champ@160) -> $FUSE ==="
if [ -d "$CHAMP" ] && [ -d "$SYN" ]; then
  $SEG/python $EX/fuse_argmax_graft.py --champ $CHAMP --synth $SYN --graft-ids 8 --out $FUSE
  FUSEFP=$ROOT/predTs/t2_fuse4_fp
  $SEG/python $EX/apply_fp_suppress.py --cfg $FPCFG --pred $FUSE --out $FUSEFP
  $SEG/python $EX/score_tiger_seg.py --task t2 --gt $GT --pred $FUSEFP \
    --downscale 160 --val-cases $VC --out $ROOT/t2_fuse4_fp_score160.json
  F_W3=$($SEG/python -c "import json;print(json.load(open('$ROOT/t2_fuse4_fp_score160.json'))['w3_dice'])")
  F_WD=$($SEG/python -c "import json;print(json.load(open('$ROOT/t2_fuse4_fp_score160.json'))['weighted_dice'])")
  FV=$($SEG/python -c "print('CONFIRMED' if $F_W3 > 0.701 else 'REFUTED')")
  echo "=== fuse4 (graft id8) w3+FP = $F_W3 (weighted=$F_WD) vs 0.701 -> $FV ==="
  echo "[Tiger][$JID] $D fuse4 graft-id8 onto champ-ens w3+FP=$F_W3 weighted=$F_WD vs 0.701 -> $FV" >> $RES
else
  echo "champ($CHAMP) or synth($SYN) dir missing -> SKIP fuse4 (3-seed synth ens is primary)"
  echo "[Tiger][$JID] $D fuse4 SKIPPED (champ/synth argmax dir missing)" >> $RES
fi

echo "=== TASK B DONE ==="
