#!/usr/bin/env python3
"""Score shipped-vs-v2 segmentation with the OFFICIAL weighted per-image metric.

Uses metrics.metrics.weighted_image_scores unchanged (the same call our validated
score_proxy_with_challenge_metric.py makes), with the challenge's clinical weight tiers.
Majority-vote over the 3 seeds reproduces how the shipped T2 ensemble is formed.
"""
import argparse, sys, glob, os
import numpy as np
from PIL import Image
sys.path.insert(0, "/users/USERNAME/Tiger-SQ-AI/challenge_repo")
from metrics.classes import CLASSES
from metrics.classes_merged import CLASSES_MERGED
from metrics.metrics import weighted_image_scores

ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); A = ap.parse_args()
RAW = "/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/nnUNet_raw"


def load(p): return np.array(Image.open(p))


def score_dir(pred_dir, gt_ds, classes):
    wt = sum(c.weight for c in classes)
    dices, hds = [], []
    for pp in sorted(glob.glob(os.path.join(pred_dir, "*.png"))):
        stem = os.path.basename(pp)
        gp = os.path.join(RAW, gt_ds, "labelsTr", stem)
        if not os.path.exists(gp):
            continue
        s = weighted_image_scores(load(pp), load(gp), classes=classes, weight_total=wt)
        d = s.get("weighted_dice", s.get("dice")); h = s.get("weighted_hd", s.get("hd"))
        if d is not None and np.isfinite(d): dices.append(float(d))
        if h is not None and np.isfinite(h): hds.append(float(h))
    return (float(np.mean(dices)) if dices else float("nan"),
            float(np.mean(hds)) if hds else float("nan"), len(dices))


def vote(dirs, out):
    os.makedirs(out, exist_ok=True)
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dirs[0], "*.png")))
    for nm in names:
        stack = np.stack([load(os.path.join(d, nm)) for d in dirs if os.path.exists(os.path.join(d, nm))])
        if stack.shape[0] == 0: continue
        flat = stack.reshape(stack.shape[0], -1)
        maj = np.apply_along_axis(lambda v: np.bincount(v).argmax(), 0, flat)
        Image.fromarray(maj.reshape(stack.shape[1:]).astype(np.uint8)).save(os.path.join(out, nm))
    return out


rows = []
for tag, d, ds, cls in (("T1 merged  SHIPPED (140 fr)", "old_t1", "Dataset211_TigerT1coarseV2", CLASSES_MERGED),
                        ("T1 merged  V2 RETRAIN (411 fr)", "new_t1", "Dataset211_TigerT1coarseV2", CLASSES_MERGED)):
    p = os.path.join(A.work, d)
    if os.path.isdir(p): rows.append((tag,) + score_dir(p, ds, cls))

for tag, pre in (("T2 fine  SHIPPED 3-seed (140 fr)", "old_t2"), ("T2 fine  V2 RETRAIN 3-seed (411 fr)", "new_t2")):
    dirs = [os.path.join(A.work, pre + s) for s in ("", "_seed1", "_seed2")]
    dirs = [x for x in dirs if os.path.isdir(x) and glob.glob(os.path.join(x, "*.png"))]
    if not dirs: continue
    ens = vote(dirs, os.path.join(A.work, pre + "_ens"))
    rows.append((tag,) + score_dir(ens, "Dataset212_TigerT2fineV2", CLASSES))

print("\n  %-38s %10s %10s %7s" % ("arm", "wDice", "wHD", "frames"))
for t, d, h, n in rows:
    print("  %-38s %10.4f %10.4f %7d" % (t, d, h, n))
for a, b in (("T1 merged  SHIPPED (140 fr)", "T1 merged  V2 RETRAIN (411 fr)"),
             ("T2 fine  SHIPPED 3-seed (140 fr)", "T2 fine  V2 RETRAIN 3-seed (411 fr)")):
    da = next((r for r in rows if r[0] == a), None); db = next((r for r in rows if r[0] == b), None)
    if da and db:
        print("\n  %s -> %s : wDice %+.4f   wHD %+.4f" % (a.split()[1], b.split()[1], db[1] - da[1], db[2] - da[2]))
