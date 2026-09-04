#!/usr/bin/env python3
"""
Wire nnU-Net proxy predictions through the OFFICIAL TIGER SQ-AI challenge scorer.

Goal: prove the challenge's weighted-Dice + normalised-Hausdorff pipeline
(metrics/metrics.py -> evaluate_seg.evaluate) runs end-to-end on REAL model
predictions, and that the w=3 tiny-structure sub-average can be computed.

We DELIBERATELY reuse the challenge code unchanged:
  - metrics.metrics.weighted_image_scores  (weighted Dice + weighted nHD)
  - metrics.evaluate_seg.evaluate          (image->case->dataset aggregation)
We only supply a proxy `classes` registry (CholecSeg8k 13 classes, with a
3-tier w1/w2/w3 weighting mirroring the challenge's tiny-structure scheme)
and an IDENTITY label function (nnU-Net already emits integer-label PNGs, so
no RGB->id conversion is needed for the proxy; on the REAL data this is where
metrics.classes.rgb_mask_to_label_mask plugs in).

The w=3 tier (proxy): the small / clinically-tricky CholecSeg8k structures
Cystic_Duct, Hepatic_Vein, Liver_Ligament, L_hook_Electrocautery  — the
analogue of TIGER's LN / nerves / small arteries tier.

Usage:
  python score_proxy_with_challenge_metric.py --gt <labelsTs_dir> --pred <pred_dir>
Both dirs contain <case>.png single-channel integer-label masks (flat layout).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

# Make the challenge repo importable (metrics is a namespace package under it).
CHALLENGE_REPO = Path("/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, str(CHALLENGE_REPO))

from metrics.classes import ClassDef                       # noqa: E402  (challenge code)
from metrics.metrics import weighted_image_scores, dice_per_class, hausdorff_per_class  # noqa: E402

# Proxy 13-class registry mirroring the challenge weighting scheme (w1/w2/w3).
# rgb is unused here (identity label fn) but kept for ClassDef signature.
PROXY_CLASSES = [
    ClassDef(0,  "background",            (0, 0, 0),       1),
    ClassDef(1,  "Abdominal_Wall",        (1, 1, 1),       1),
    ClassDef(2,  "Liver",                 (2, 2, 2),       1),
    ClassDef(3,  "Grasper",               (3, 3, 3),       1),
    ClassDef(4,  "Connective_Tissue",     (4, 4, 4),       1),
    ClassDef(5,  "Blood",                 (5, 5, 5),       1),
    ClassDef(6,  "Fat",                   (6, 6, 6),       1),
    ClassDef(7,  "GI_Tract",              (7, 7, 7),       2),
    ClassDef(8,  "Gallbladder",           (8, 8, 8),       2),
    # w=3 tier: small / clinically-tricky structures (TIGER tiny-structure analogue)
    ClassDef(9,  "Hepatic_Vein",          (9, 9, 9),       3),
    ClassDef(10, "Liver_Ligament",        (10, 10, 10),    3),
    ClassDef(11, "Cystic_Duct",           (11, 11, 11),    3),
    ClassDef(12, "L_hook_Electrocautery", (12, 12, 12),    3),
]
W3_IDS = [c.label_id for c in PROXY_CLASSES if c.weight == 3]
WEIGHT_TOTAL = sum(c.weight for c in PROXY_CLASSES)


def identity_label(arr: np.ndarray) -> np.ndarray:
    """nnU-Net already writes integer-label masks. The challenge loader calls
    Image.open(...).convert('RGB') first, so collapse RGB back to the single id."""
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=Path, help="dir of GT <case>.png label masks")
    ap.add_argument("--pred", required=True, type=Path, help="dir of pred <case>.png label masks")
    args = ap.parse_args()

    gt_files = sorted(args.gt.glob("*.png"))
    if not gt_files:
        raise SystemExit(f"No GT PNGs in {args.gt}")

    # Per-image weighted Dice/HD via the EXACT challenge primitive.
    img_dice, img_hd = [], []
    # Also accumulate per-class Dice/HD so we can compute the w=3 sub-average.
    perclass_dice = defaultdict(list)
    perclass_hd = defaultdict(list)

    n = 0
    for gtp in gt_files:
        predp = args.pred / gtp.name
        if not predp.exists():
            print(f"  WARN missing pred for {gtp.name}, skip")
            continue
        gt = identity_label(np.array(Image.open(gtp).convert("RGB")))
        pr = identity_label(np.array(Image.open(predp).convert("RGB")))

        s = weighted_image_scores(pr, gt, classes=PROXY_CLASSES, weight_total=WEIGHT_TOTAL)
        img_dice.append(s["dice"]); img_hd.append(s["hd"])

        dpc = dice_per_class(pr, gt, PROXY_CLASSES)
        hpc = hausdorff_per_class(pr, gt, PROXY_CLASSES)
        for cid in dpc:
            perclass_dice[cid].append(dpc[cid])
            perclass_hd[cid].append(hpc[cid])
        n += 1

    final_dice = float(np.mean(img_dice))
    final_hd = float(np.mean(img_hd))
    # w=3 tiny-structure SUB-AVERAGE (the deciding metric tier on the real LB).
    w3_dice = float(np.mean([np.mean(perclass_dice[c]) for c in W3_IDS]))
    w3_hd = float(np.mean([np.mean(perclass_hd[c]) for c in W3_IDS]))

    print("=" * 64)
    print(f"CHALLENGE SCORER on nnU-Net proxy predictions  (n={n} images)")
    print("=" * 64)
    print(f"  weighted Dice (all classes, w1/w2/w3 / {WEIGHT_TOTAL}) : {final_dice:.4f}")
    print(f"  weighted nHD  (all classes, lower better)        : {final_hd:.4f}")
    print(f"  --- w=3 tiny-structure SUB-AVERAGE ---")
    print(f"  w=3 classes: {[PROXY_CLASSES[c].name for c in W3_IDS]}")
    print(f"  w=3 mean Dice : {w3_dice:.4f}")
    print(f"  w=3 mean nHD  : {w3_hd:.4f}")
    print("  --- per-class mean Dice ---")
    for c in PROXY_CLASSES:
        d = np.mean(perclass_dice[c.label_id])
        print(f"    [{c.label_id:2d} w{c.weight}] {c.name:24s} Dice={d:.4f}")


if __name__ == "__main__":
    main()
