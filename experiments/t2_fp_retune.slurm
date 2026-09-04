#!/bin/bash
#SBATCH --job-name=tiger_fp_retune
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/users/USERNAME/joblogs/tiger_fp_retune_%j.out
#SBATCH --error=/users/USERNAME/joblogs/tiger_fp_retune_%j.err
#
# Re-tune FP-suppress thresholds ON THE NEW d212 3-seed ensemble.
# The STALE config already helps on the new model (w3 0.6986 -> 0.7372, +0.0386), refuting my
# worry that thresholds tuned on a weaker model would delete true positives. This asks the
# remaining question: does re-tuning on the new preds do BETTER than the stale config?
#
# Two bugs in the first attempt (7710409), both mine:
#   * --val-cases defaulted to center_1_case_14/15, but the eval frames are centres 2/3/4/6/7,
#     so the tuner collected ZERO frames and died on frames[0][0].
#   * --downscale 0 without --no-hd re-enters the O(n^2) Hausdorff the tuner scores with.
# No GPU needed: connected components + Dice only. CPU partition, so it does not compete with
# the GPU queue.
set -uo pipefail
[ -d /users/USERNAME/Tiger-SQ-AI/experiments ] || { echo "[eio-guard] HOME unreadable -> abort"; exit 3; }
SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
E=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/adopt_ens_7705611
W=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t2fp_${SLURM_JOB_ID}
mkdir -p "$W"
CASES=$(ls $E/gt_212/*.png | xargs -n1 basename | sed -E 's/^(center_[0-9]+_case_[0-9]+).*/\1/' | sort -u | tr '\n' ' ')
echo "eval cases: $CASES"

$SEG/python /users/USERNAME/Tiger-SQ-AI/experiments/tune_fp_suppress.py \
  --gt $E/gt_212 --pred $E/p_t2_new --task t2 --downscale 0 --no-hd \
  --val-cases $CASES --out "$W/newfp.json"
echo "TUNE_RC=$?"
[ -f "$W/newfp.json" ] || { echo "no config written -> keep the STALE config (already +0.0386)"; exit 0; }

$SEG/python /users/USERNAME/Tiger-SQ-AI/experiments/apply_fp_suppress.py \
  --cfg "$W/newfp.json" --pred "$E/p_t2_new" --out "$W/p_newfp"

echo "=== FINAL: raw vs stale-FP vs retuned-FP on the new d212 ensemble ==="
$SEG/python - "$E/gt_212" "$E/p_t2_new" /scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t2follow_7710409/p_stalefp "$W/p_newfp" <<'PY'
import sys, glob, os
import numpy as np
from PIL import Image
sys.path.insert(0, "/users/USERNAME/Tiger-SQ-AI/challenge_repo")
from metrics.classes import CLASSES
from metrics.metrics import dice_per_class
gt_dir = sys.argv[1]; WT = sum(c.weight for c in CLASSES); t3 = [c for c in CLASSES if c.weight == 3]
res = {}
for lbl, pd in (("raw ensemble", sys.argv[2]), ("stale FP cfg", sys.argv[3]), ("RETUNED FP", sys.argv[4])):
    if not os.path.isdir(pd): print("  %-14s (missing)" % lbl); continue
    wd, tt, n = [], [], 0
    for g in sorted(glob.glob(os.path.join(gt_dir, "*.png"))):
        p = os.path.join(pd, os.path.basename(g))
        if not os.path.isfile(p): continue
        d = dice_per_class(np.array(Image.open(p)), np.array(Image.open(g)), CLASSES)
        wd.append(sum(d[c.label_id]*c.weight for c in CLASSES)/WT)
        tt.append(sum(d[c.label_id] for c in t3)/len(t3)); n += 1
    res[lbl] = float(np.mean(tt))
    print("  %-14s wDice %.4f   w3 %.4f   (n=%d)" % (lbl, np.mean(wd), np.mean(tt), n))
if "RETUNED FP" in res and "stale FP cfg" in res:
    d = res["RETUNED FP"] - res["stale FP cfg"]
    print("\n  retuned - stale = %+.4f w3" % d)
    print("  >>> %s" % ("SHIP THE RETUNED config" if d > 0.005 else
          "KEEP THE STALE config (retune buys < 0.005; fewer moving parts on a frozen artifact)"))
PY
echo "RESULT: workdir=$W"
exit 0
