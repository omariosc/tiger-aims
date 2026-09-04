#!/bin/bash
#SBATCH --job-name=tiger_adopt_ens
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=gpu013
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/users/USERNAME/joblogs/tiger_adopt_ens_%j.out
#SBATCH --error=/users/USERNAME/joblogs/tiger_adopt_ens_%j.err
#
# T2 ADOPTION, ENSEMBLE vs ENSEMBLE -- the comparison that actually decides what we ship.
#
# Job 7703505 compared T2 SINGLE-SEED vs SINGLE-SEED (retrain +0.2761 w3). That isolates DATA
# correctly, but the SHIPPED T2 is a 3-seed Focal-Tversky ensemble, so the incumbent's real bar is
# higher than the single seed it was scored on. This runs 3 seeds per side, prob-averages each,
# and re-applies the SAME pre-registered rule on the SAME frames.
#
# It also avoids 7703505's fatal flaw: scoring goes through adoption_rule_score_dice.py, which
# never calls hausdorff_per_class. 7703505 finished all its predictions and then burned its whole
# 3 h wall inside the O(n^2) Hausdorff without emitting a single number (dossier 0i).
set -uo pipefail
[ -d /users/USERNAME/Tiger-SQ-AI/experiments ] || { echo "[eio-guard] HOME unreadable -> abort"; exit 3; }
SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
NN=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet
export nnUNet_raw=$NN/nnUNet_raw nnUNet_preprocessed=$NN/nnUNet_preprocessed nnUNet_results=$NN/nnUNet_results
SRC=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/adopt_7703505
W=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/adopt_ens_${SLURM_JOB_ID}
mkdir -p "$W"

# reuse the ALREADY-BUILT evaluation frame set (centres 2/3/4/6/7 val frames) rather than
# rebuilding it -- identical frames is what makes this comparable to the single-seed run.
for d in in_212 gt_212; do
  [ -d "$SRC/$d" ] || { echo "FATAL: $SRC/$d missing"; exit 4; }
  cp -r "$SRC/$d" "$W/$d"
done
echo "eval frames: $(ls $W/gt_212/*.png 2>/dev/null | wc -l) gt, $(ls $W/in_212/*.png 2>/dev/null | wc -l) input"

TRS=(nnUNetTrainer_250epochs_NoMirror_FocalTversky \
     nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed1 \
     nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed2)

RC=0
for SIDE in inc new; do
  if [ "$SIDE" = inc ]; then DS=202; DSN=Dataset202_TigerT2fine; else DS=212; DSN=Dataset212_TigerT2fineV2; fi
  MEMBERS=()
  for i in 0 1 2; do
    T=${TRS[$i]}
    O="$W/pp_${SIDE}_s${i}"
    echo "=== predict T2 $SIDE seed$i (d$DS $T) ==="
    $SEG/nnUNetv2_predict -i "$W/in_212" -o "$O" -d "$DS" -c 2d -f 0 -tr "$T" \
        -p nnUNetPlans -chk checkpoint_final.pth --save_probabilities --disable_tta
    r=$?; echo "    rc=$r files=$(ls $O 2>/dev/null | wc -l)"
    [ $r -ne 0 ] && RC=1
    MEMBERS+=("$O")
  done
  echo "=== ensemble T2 $SIDE (3 seeds, softmax prob-average) ==="
  $SEG/nnUNetv2_ensemble -i "${MEMBERS[@]}" -o "$W/p_t2_${SIDE}" -np 8
  echo "    rc=$? files=$(ls $W/p_t2_${SIDE} 2>/dev/null | wc -l)"
done

echo "=== SCORE: ensemble vs ensemble, same frames, same pre-registered rule ==="
# the scorer reads p_t2_inc / p_t2_new; give it the T1 dirs from the earlier run so its
# T1 block still reports (T1 already decided: KEEP, w3 0.4187 vs 0.4048)
for d in gt_211 p_t1_inc p_t1_new; do [ -d "$SRC/$d" ] && ln -sfn "$SRC/$d" "$W/$d"; done
$SEG/python /users/USERNAME/Tiger-SQ-AI/experiments/adoption_rule_score_dice.py "$W"
echo "RESULT: workdir=$W predict_rc=$RC"
exit $RC
