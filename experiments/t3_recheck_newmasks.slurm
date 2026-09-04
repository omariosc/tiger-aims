#!/bin/bash
#SBATCH --job-name=tiger_t3_recheck
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=gpu013
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/users/USERNAME/joblogs/tiger_t3_recheck_%j.out
#SBATCH --error=/users/USERNAME/joblogs/tiger_t3_recheck_%j.err
#
# DOES ADOPTING THE NEW T2 BREAK T3?
#
# T3 ships a seg-CONDITIONED head: it reads presence + log-area of 18 boundary structures FROM THE
# FINE MASKS, blended 50/50 with the station co-visibility prior. Adopting the retrain moves FINE
# w3 from 0.3733 to 0.6986 -- so the head's INPUT DISTRIBUTION changes. T3 was settled on
# 2026-09-02 against the OLD masks, and the deployed head (ckpt_segcond_ensmask) was itself trained
# on OLD-model masks. This is a distribution shift, and it can go either way:
#   * BETTER: 2026-06-25 showed cleaner masks lift T3 (+0.046 going single-seed -> ensemble masks).
#   * WORSE:  the head has never seen features this good; train/test mismatch.
# Neither is safe to assume, and T3 is a THIRD of the combined rank.
#
# Scored with the OFFICIAL evaluate_cls on the SAME held-out frames used by the adoption rule.
set -uo pipefail
[ -d /users/USERNAME/Tiger-SQ-AI/experiments ] || { echo "[eio-guard] HOME unreadable -> abort"; exit 3; }
SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
T=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI
E=$T/adopt_ens_7705611
STALEFP=$T/t2follow_7710409/p_stalefp        # new ensemble + the FP config we ship
W=$T/t3recheck_${SLURM_JOB_ID}; mkdir -p "$W"
CKPT=$T/t3/ckpt_segcond_ensmask
for d in "$E/p_t2_inc" "$E/p_t2_new" "$STALEFP" "$CKPT"; do
  [ -e "$d" ] || { echo "FATAL: missing $d"; exit 4; }
done

$SEG/python - "$E" "$STALEFP" "$CKPT" "$W" <<'PY'
import sys, os, glob, json
import numpy as np
from PIL import Image
sys.path.insert(0, "/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, "/users/USERNAME/Tiger-SQ-AI/experiments")
E, STALEFP, CKPT, W = sys.argv[1:5]
import torch
from station_boundary_map import station_boundary_classes          # noqa
from metrics.classes_stations import STATION_ORDER                 # noqa
from metrics.evaluate_cls import evaluate                          # noqa

# feature builder must match training EXACTLY: presence + log-area-fraction of the boundary set
BOUND = sorted({c for v in station_boundary_classes().values() for c in v} | {24})
def feats(png):
    a = np.array(Image.open(png)); tot = float(a.size)
    f = []
    for c in BOUND:
        n = int((a == c).sum())
        f += [1.0 if n > 0 else 0.0, float(np.log1p(n / tot))]
    return f

vis = {}
import csv
for p in ("/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/synapse_v2/lymph_node_station_visibility.csv",
          "/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/synapse/lymph_node_station_visibility.csv"):
    if not os.path.exists(p): continue
    for r in csv.DictReader(open(p)):
        k = (r.get("case_id") or r.get("frame") or "").strip()
        if k: vis.setdefault(k, r)
print("  station GT rows loaded:", len(vis))

sd = torch.load(os.path.join(CKPT, "best.pt"), map_location="cpu")
sd = sd.get("model", sd)
lin = [(k, v) for k, v in sd.items() if v.ndim == 2]
print("  head tensors:", [(k, tuple(v.shape)) for k, v in lin][:4])

def run(pred_dir, label):
    names = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
    X, Y, keep = [], [], []
    for f in names:
        stem = os.path.basename(f)[:-4]
        if stem not in vis: continue
        row = vis[stem]
        try: y = [float(row[s]) for s in STATION_ORDER]
        except KeyError: continue
        X.append(feats(f)); Y.append(y); keep.append(stem)
    if not X:
        print("  %-22s NO frames matched the station GT (stem mismatch) -- cannot score" % label); return None
    X = torch.tensor(X, dtype=torch.float32); Y = np.array(Y)
    m = torch.nn.Sequential()
    try:
        import importlib; tm = importlib.import_module("train_t3_seg_conditioned")
        m = tm.build_head(X.shape[1], len(STATION_ORDER)) if hasattr(tm, "build_head") else None
    except Exception: m = None
    if m is None:
        print("  %-22s could not rebuild the head architecture -- reporting features only" % label); return None
    m.load_state_dict(sd, strict=False); m.eval()
    with torch.no_grad(): P = torch.sigmoid(m(X)).numpy()
    r = evaluate(Y, P)
    print("  %-22s n=%3d  macro-F1 %.4f   AUROC %.4f" % (label, len(keep), r.get("f1", float("nan")), r.get("auroc", float("nan"))))
    return r

print("\n  === deployed seg-cond head, OLD masks vs NEW masks (same frames) ===")
a = run(os.path.join(E, "p_t2_inc"), "OLD masks (incumbent)")
b = run(os.path.join(E, "p_t2_new"), "NEW masks (retrain raw)")
c = run(STALEFP,                     "NEW masks + FP (ship)")
json.dump({"old": a, "new": b, "new_fp": c}, open(os.path.join(W, "t3_recheck.json"), "w"), indent=1, default=str)
print("\n  >>> if NEW >= OLD, T3 is safe under the T2 adoption and ships unchanged.")
print("  >>> if NEW <  OLD, the head must be RETRAINED on new-model masks before the rebuild.")
PY
echo "RESULT: workdir=$W"
exit 0
