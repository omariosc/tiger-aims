#!/bin/bash
#SBATCH --job-name=tiger_t3ens_rgraft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%j.err
# NO #SBATCH mail (agent self-monitors). Train/eval ONLY via sbatch (this).
#
# NEXT LEVER after the CT205 graft REFUTED (job 6259251 = fill-only no-op, 0.70105 tie).
#
# PART A (PRIMARY, highest recoverable headroom): re-run the SEG-CONDITIONED T3 head on the
#   CURRENT CHAMPION FINE masks = the 3-seed FT ENSEMBLE (predTs/t2_ft_ens_fp). The deployed
#   T3 head (ckpt_segcond_ftpred, F1 0.582) was trained/eval'd on the SINGLE-FT predicted masks;
#   the ensemble masks are materially cleaner (w3 0.701 vs single 0.594) and were NEVER fed to T3.
#   KILL-A: ens-mask seg-cond T3 weighted-F1 > 0.582 (deployed) -> adopt as new T3; else REFUTE
#           (confirms §0f: T3 features depend on structure PRESENCE not mask sharpness -> data-floor).
#
# PART B (near-free CPU): the REPLACE-mode graft never run in 6259251 (only fill). Test
#   replace{8} and replace{8,9,23} of CT205 synth win-classes onto the champion ensemble.
#   KILL-B: replace-graft w3+FP@160 > 0.701 -> new FINE champion; else REFUTE the synth-graft lever.
set -u
SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
ROOT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet
EX=/users/USERNAME/Tiger-SQ-AI/experiments
RES=/users/USERNAME/AUTONOMOUS_RESULTS.md
export HF_HOME=/scratch/USERNAME/hf_cache
export nnUNet_raw=$ROOT/nnUNet_raw
export nnUNet_preprocessed=$ROOT/nnUNet_preprocessed
export nnUNet_results=$ROOT/nnUNet_results
D=$(date +%F); JID=${SLURM_JOB_ID:-local}
GT=$ROOT/nnUNet_raw/Dataset202_TigerT2fine/labelsTr
FPCFG=$ROOT/t2_ft_ens_fpsuppress_fullres.json
VC="center_1_case_14 center_1_case_15"
T3OUT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t3

echo "=== tiger_t3ens_rgraft JID=$JID $D ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

# ============================================================================
# PART A — seg-conditioned T3 on the 3-seed FT ENSEMBLE masks (champion FINE)
# ============================================================================
ENSMASK=$ROOT/predTs/t2_ft_ens_fp        # champion FINE masks (FP-suppressed, 28 val frames)
NMASK=$(ls $ENSMASK/*.png 2>/dev/null | wc -l)
echo "=== PART A: seg-cond T3 on champion ENSEMBLE masks ($NMASK frames) -> ckpt_segcond_ensmask ==="
if [ "$NMASK" -eq 28 ]; then
  $SEG/python $EX/train_t3_seg_conditioned.py --mode seg \
      --val-pred-dir $ENSMASK --outdir $T3OUT/ckpt_segcond_ensmask
  F1ENS=$($SEG/python -c "import json;print(json.load(open('$T3OUT/ckpt_segcond_ensmask/metrics.json'))['final_f1'])" 2>/dev/null || echo NA)
  AUENS=$($SEG/python -c "import json;print(json.load(open('$T3OUT/ckpt_segcond_ensmask/metrics.json'))['final_auroc'])" 2>/dev/null || echo NA)
  echo "=== T3 seg-cond on ENS masks: F1=$F1ENS AUROC=$AUENS  vs deployed (single-FT masks) 0.5815 ==="
  VA=$($SEG/python -c "print('CONFIRMED-T3-IMPROVE' if $F1ENS > 0.5815 else 'REFUTED')" 2>/dev/null || echo REFUTED)
  echo "[Tiger][$JID] $D T3 seg-cond on 3-seed-ENS masks F1=$F1ENS AUROC=$AUENS vs deployed 0.5815 -> $VA" >> $RES
else
  echo "[Tiger][$JID] $D PART A SKIPPED — ens masks count=$NMASK != 28" >> $RES
fi

# ============================================================================
# PART B — REPLACE-mode graft (CT205 synth win-classes -> champion ensemble)
# ============================================================================
CHAMP=$ROOT/predTs/t2_ft_ens          # champion 3-seed FT ensemble argmax (RAW, pre-FP)
SYN=$ROOT/predTs/t2_ct205_ens         # CT205 synth ensemble argmax (donor)
echo "=== PART B: REPLACE-mode graft from CT205 synth-ens onto champion-ens ==="
if [ -d "$CHAMP" ] && [ -d "$SYN" ]; then
  for IDS in "8" "8 9 23"; do
    TAG="t2_ct205_rep$(echo $IDS | tr ' ' '-')"
    OUTD=$ROOT/predTs/$TAG
    echo "--- replace ids [$IDS] -> $TAG ---"
    $SEG/python $EX/fuse_argmax_graft.py --champ $CHAMP --synth $SYN \
        --graft-ids $IDS --mode replace --out $OUTD
    $SEG/python $EX/apply_fp_suppress.py --cfg $FPCFG --pred $OUTD --out ${OUTD}_fp
    $SEG/python $EX/score_tiger_seg.py --task t2 --gt $GT --pred ${OUTD}_fp \
        --downscale 160 --val-cases $VC --out $ROOT/${TAG}_fp_score160.json
    W3=$($SEG/python -c "import json;print(json.load(open('$ROOT/${TAG}_fp_score160.json'))['w3_dice'])")
    WD=$($SEG/python -c "import json;print(json.load(open('$ROOT/${TAG}_fp_score160.json'))['weighted_dice'])")
    VB=$($SEG/python -c "print('CONFIRMED-NEW-CHAMPION' if $W3 > 0.701 else 'REFUTED')")
    echo "=== replace{$IDS} w3+FP = $W3 (weighted=$WD) vs 0.701 -> $VB ==="
    echo "[Tiger][$JID] $D CT205 REPLACE-graft{$IDS} onto champ-ens w3+FP=$W3 weighted=$WD vs 0.701 -> $VB" >> $RES
  done
else
  echo "[Tiger][$JID] $D PART B SKIPPED — champ($CHAMP) or synth donor($SYN) dir missing" >> $RES
fi

echo "=== tiger_t3ens_rgraft DONE ==="
