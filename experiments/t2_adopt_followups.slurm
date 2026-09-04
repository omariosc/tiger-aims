#!/bin/bash
#SBATCH --job-name=tiger_t2_follow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=gpu013
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/users/USERNAME/joblogs/tiger_t2_follow_%j.out
#SBATCH --error=/users/USERNAME/joblogs/tiger_t2_follow_%j.err
#
# T2 ADOPTION FOLLOW-UPS. The retrain WON ensemble-vs-ensemble (+0.3253 w3, job 7705611), so the
# container must ship the d212 3-seed ensemble. But three shipped components were tuned against the
# OLD, much weaker FINE model and none of them can be assumed to transfer:
#
#   (1) FP-SUPPRESS thresholds -- tuned on old preds. The dossier states blob sizes shift with a
#       better model ("re-tune on the 1k preds ... since blob sizes shift"). Applying stale absolute
#       thresholds to a stronger model could delete TRUE positives.
#   (2) The synth replace{8} GRAFT (+0.0082) -- its premise is that CT205 synth beats the champion
#       on class 8 (R_Inf_Pulm_Lig). The champion just got much stronger; the premise likely fails.
#   (3) T3 is SEG-CONDITIONED on the FINE masks. FINE w3 went 0.3733 -> 0.6986, so T3's input
#       distribution has changed. T3 was settled on 2026-09-02 assuming the OLD masks.
#
# Everything is measured on the SAME held-out frames (centres 2/3/4/6/7) used by the adoption rule,
# scored Dice-only (the O(n^2) Hausdorff is what killed 7703505).
set -uo pipefail
[ -d /users/USERNAME/Tiger-SQ-AI/experiments ] || { echo "[eio-guard] HOME unreadable -> abort"; exit 3; }
SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
NN=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet
export nnUNet_raw=$NN/nnUNet_raw nnUNet_preprocessed=$NN/nnUNet_preprocessed nnUNet_results=$NN/nnUNet_results
E=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/adopt_ens_7705611
W=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t2follow_${SLURM_JOB_ID}
mkdir -p "$W"
E_ENS="$E/p_t2_new"     # the NEW d212 3-seed ensemble argmax (the adoption winner)
[ -d "$E_ENS" ] || { echo "FATAL: $E_ENS missing"; exit 4; }
echo "new-T2 ensemble preds: $(ls $E_ENS/*.png 2>/dev/null | wc -l)  gt: $(ls $E/gt_212/*.png 2>/dev/null | wc -l)"

echo "=== (1) FP-SUPPRESS: stale config vs RE-TUNED on the new model ==="
$SEG/python /users/USERNAME/Tiger-SQ-AI/experiments/apply_fp_suppress.py \
   --cfg $NN/t2_ft_ens_fpsuppress_fullres.json --pred "$E_ENS" --out "$W/p_stalefp" 2>&1 | tail -3
$SEG/python /users/USERNAME/Tiger-SQ-AI/experiments/tune_fp_suppress.py \
   --gt "$E/gt_212" --pred "$E_ENS" --task t2 --downscale 0 \
   --out "$W/newfp.json" 2>&1 | tail -5
$SEG/python /users/USERNAME/Tiger-SQ-AI/experiments/apply_fp_suppress.py \
   --cfg "$W/newfp.json" --pred "$E_ENS" --out "$W/p_newfp" 2>&1 | tail -3

echo "=== SCORE: raw vs stale-FP vs retuned-FP (Dice only, same frames) ==="
$SEG/python - "$E/gt_212" "$E_ENS" "$W/p_stalefp" "$W/p_newfp" <<'PY'
import sys, glob, os
import numpy as np
from PIL import Image
sys.path.insert(0, "/users/USERNAME/Tiger-SQ-AI/challenge_repo")
from metrics.classes import CLASSES
from metrics.metrics import dice_per_class
gt_dir = sys.argv[1]; WT = sum(c.weight for c in CLASSES); t3 = [c for c in CLASSES if c.weight == 3]
for lbl, pd in (("raw ensemble", sys.argv[2]), ("stale FP cfg", sys.argv[3]), ("RETUNED FP", sys.argv[4])):
    if not os.path.isdir(pd): print("  %-14s (missing)" % lbl); continue
    wd, tt, n = [], [], 0
    for g in sorted(glob.glob(os.path.join(gt_dir, "*.png"))):
        p = os.path.join(pd, os.path.basename(g))
        if not os.path.isfile(p): continue
        d = dice_per_class(np.array(Image.open(p)), np.array(Image.open(g)), CLASSES)
        wd.append(sum(d[c.label_id]*c.weight for c in CLASSES)/WT)
        tt.append(sum(d[c.label_id] for c in t3)/len(t3)); n += 1
    print("  %-14s wDice %.4f   w3 %.4f   (n=%d)" % (lbl, np.mean(wd), np.mean(tt), n))
print("  >>> ship the FP variant with the best w3; if RETUNED <= raw, ship NO FP on the new model.")
PY
echo "RESULT: workdir=$W"
exit 0
