#!/bin/bash
#SBATCH --job-name=tiger_cbdice
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/seg_%x_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/seg_%x_%j.err
# NO #SBATCH mail (agent self-monitors). cbDice (Centerline Boundary Dice,
# MICCAI'24) loss lever for the thin/tubular w3 tier. Mirroring OFF (flipped
# thoracic view), 250 ep == identical schedule to the banked Focal-Tversky
# trainer so ONLY the loss differs (clean ablation). Writes to SEPARATE pred
# dirs + score files tagged with the trainer -> does NOT clobber banked outputs.
# Inline chain: train -> predict val(14,15) -> FP-suppress -> score @160px.
#
# Usage:
#   sbatch --job-name=tiger_t2_cbdice   train_tiger_seg_cbdice.slurm 202 t2 cbdice
#   sbatch --job-name=tiger_t2_cbdiceft train_tiger_seg_cbdice.slurm 202 t2 cbdiceft
#   sbatch --job-name=tiger_t1_cbdice   train_tiger_seg_cbdice.slurm 201 t1 cbdice
#   sbatch --job-name=tiger_t1_cbdiceft train_tiger_seg_cbdice.slurm 201 t1 cbdiceft
set -u
DSID=${1:?dataset id (201=T1coarse/MERGED, 202=T2fine/FINE)}
TASKBASE=${2:?task tag base (t1|t2)}     # t1=merged(official task2), t2=fine(official task1)
VARIANT=${3:?variant (cbdice|cbdiceft)}

case "$VARIANT" in
  cbdice)   TRAINER=nnUNetTrainer_250epochs_NoMirror_cbDice ;;
  cbdiceft) TRAINER=nnUNetTrainer_250epochs_NoMirror_cbDiceFocalTversky ;;
  *) echo "unknown variant $VARIANT"; exit 2 ;;
esac
# the t2(fine) FP config is t2_ft; the t1(merged) FP config is t1.
if [ "$TASKBASE" = "t2" ]; then FPCFG=t2_ft; TAG=t2_${VARIANT}; else FPCFG=t1; TAG=t1_${VARIANT}; fi

SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
ROOT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet
EX=/users/USERNAME/Tiger-SQ-AI/experiments
RES=/users/USERNAME/AUTONOMOUS_RESULTS.md
export nnUNet_raw=$ROOT/nnUNet_raw
export nnUNet_preprocessed=$ROOT/nnUNet_preprocessed
export nnUNet_results=$ROOT/nnUNet_results
export nnUNet_n_proc_DA=12
D=$(date +%F)
W=8

DSNAME=$(basename $(ls -d $nnUNet_raw/Dataset${DSID}_*))
DSNAME=${DSNAME#Dataset${DSID}_}
RAW=$nnUNet_raw/Dataset${DSID}_${DSNAME}
SPLIT=$nnUNet_preprocessed/Dataset${DSID}_${DSNAME}/splits_final.json
[ "$TASKBASE" = "t1" ] && GT=$RAW/labelsTr || GT=$RAW/labelsTr   # GT == labelsTr of this dataset
VC="center_1_case_14 center_1_case_15"

echo "=== cbDice lever: DSID=$DSID TASKBASE=$TASKBASE VARIANT=$VARIANT TAG=$TAG TRAINER=$TRAINER ==="
echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total --format=csv
$SEG/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# split must already exist (shared with banked runs); assert it.
if [ ! -f "$SPLIT" ]; then echo "splits_final.json MISSING ($SPLIT) -> ABORT"; exit 1; fi
echo "=== split (fold 0) val frames ==="; jq -r '.[0].val[]' "$SPLIT" | head

echo "=== TRAIN Dataset${DSID} 2d fold 0  ($TRAINER) ==="
$SEG/nnUNetv2_train ${DSID} 2d 0 -tr ${TRAINER} --npz

echo "=== PREDICT held-out val (case 14,15) -> predTs/$TAG ==="
PREDDIR=$ROOT/predTs/${TAG}
mkdir -p $PREDDIR
VALIMG=/tmp/valimg_${TAG}; rm -rf $VALIMG; mkdir -p $VALIMG
for stem in $(jq -r '.[0].val[]' "$SPLIT"); do
  cp $RAW/imagesTr/${stem}_0000.png $RAW/imagesTr/${stem}_0001.png $RAW/imagesTr/${stem}_0002.png $VALIMG/
done
$SEG/nnUNetv2_predict -i $VALIMG -o $PREDDIR -d ${DSID} -c 2d -f 0 \
    -tr ${TRAINER} --disable_tta

echo "=== APPLY FP-suppress (banked $FPCFG config) -> predTs/${TAG}_fp ==="
$SEG/python $EX/apply_fp_suppress.py --cfg $ROOT/${FPCFG}_fpsuppress_fullres.json \
    --pred $PREDDIR --out $ROOT/predTs/${TAG}_fp || echo "(FP-suppress apply failed; raw preds still valid)"

echo "=== SCORE @160px (raw + FP-suppressed) ==="
$SEG/python $EX/score_tiger_seg.py --task ${TASKBASE} --gt $GT --pred $PREDDIR \
    --downscale 160 --val-cases $VC --workers $W --out $ROOT/${TAG}_score160.json \
    || echo "(raw score failed)"
$SEG/python $EX/score_tiger_seg.py --task ${TASKBASE} --gt $GT --pred $ROOT/predTs/${TAG}_fp \
    --downscale 160 --val-cases $VC --workers $W --out $ROOT/${TAG}_fp_score160.json \
    || echo "(FP score failed)"

echo "=== VERDICT line -> AUTONOMOUS_RESULTS.md ==="
$SEG/python - "$ROOT" "$TAG" "$TASKBASE" "$VARIANT" "$SLURM_JOB_ID" "$D" >> $RES 2>/dev/null <<'PY'
import json, sys
from pathlib import Path
root, tag, base, variant, jid, d = sys.argv[1:7]
NN = Path(root)
def g(p, k):
    f = NN / p
    if not f.exists(): return None
    try: return json.loads(f.read_text()).get(k)
    except Exception: return None
def fmt(x): return f"{x:.4f}" if isinstance(x,(int,float)) else str(x)
# banked 250-ep FT references (160px) per the dossier
ref = {"t2": (0.498, "t2_ft 250ep"), "t1": (0.349, "t1 250ep")}
ref1k = {"t2": 0.529, "t1": 0.471}
raw_w3 = g(f"{tag}_score160.json","w3_dice"); raw_wd = g(f"{tag}_score160.json","weighted_dice")
fp_w3  = g(f"{tag}_fp_score160.json","w3_dice"); fp_nhd = g(f"{tag}_fp_score160.json","weighted_nhd")
banked, bname = ref[base]
best = fp_w3 if fp_w3 is not None else raw_w3
line = (f"[Tiger][{jid}] {d} cbDice {variant} {base} @160px: "
        f"raw w3={fmt(raw_w3)} wDice={fmt(raw_wd)} | +FP w3={fmt(fp_w3)} nHD={fmt(fp_nhd)} "
        f"vs banked {bname} w3={banked:.3f} (1k {ref1k[base]:.3f})")
if best is None:
    print(line + " -> INCONCLUSIVE (missing score)")
elif best > banked + 0.01:
    print(line + f" -> CONFIRMED vs 250ep (+{best-banked:.3f} w3)")
elif best < banked - 0.01:
    print(line + f" -> REFUTED ({best-banked:+.3f} w3 vs 250ep)")
else:
    print(line + f" -> FLAT vs 250ep (Δw3={best-banked:+.3f})")
PY

echo "=== ALL DONE $TAG ($TRAINER) ==="
