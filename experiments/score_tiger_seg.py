#!/usr/bin/env python3
"""
Score nnU-Net val predictions with the OFFICIAL TIGER SQ-AI segmentation metric.

nnU-Net writes integer-label PNGs (ids 0..K-1) for both GT (labelsTr held-out)
and preds. We feed them straight through the unmodified challenge primitives:
  metrics.metrics.weighted_image_scores  (weighted Dice + weighted nHD)
  metrics.metrics.dice_per_class / hausdorff_per_class  (for the w=3 sub-average)

Registry is the REAL challenge one with the official w1/w2/w3 weights:
  --task t1  -> metrics.classes_merged.CLASSES_MERGED (16-class, coarse)
  --task t2  -> metrics.classes.CLASSES             (31-class, fine)

nHD is O(n^2) over all fg pixels -> SLOW at 1080p. --downscale N (default 160)
NEAREST-resamples both masks before HD (Dice is scale-robust; HD ranks preserved)
for fast iteration. Use --downscale 0 for the exact final score.

Usage:
  python score_tiger_seg.py --task t2 --gt <labelsTr_or_gt_dir> --pred <pred_dir> \
         [--downscale 160] [--val-cases center_1_case_14 center_1_case_15]
GT/pred dirs contain <stem>.png integer-label masks (flat layout). If --val-cases
given, only frames whose case is in the set are scored (matches the train split).
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

CHALLENGE_REPO = Path("/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, str(CHALLENGE_REPO))
from metrics.metrics import weighted_image_scores, dice_per_class, hausdorff_per_class


def case_of(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def load_label(p: Path, downscale: int) -> np.ndarray:
    arr = np.array(Image.open(p).convert("RGB"))[:, :, 0].astype(np.uint8)
    if downscale and max(arr.shape) > downscale:
        h, w = arr.shape
        s = downscale / max(h, w)
        nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
        arr = np.array(Image.fromarray(arr).resize((nw, nh), Image.NEAREST))
    return arr


# module-level worker (picklable) for multiprocessing.Pool over frames
_W = {}


def _init_worker(pred_dir, downscale, classes, wt, no_hd=False):
    _W["pred"] = Path(pred_dir); _W["ds"] = downscale
    _W["classes"] = classes; _W["wt"] = wt; _W["no_hd"] = no_hd


def _score_one(gtp_str):
    gtp = Path(gtp_str)
    predp = _W["pred"] / gtp.name
    if not predp.exists():
        return None
    gt = load_label(gtp, _W["ds"]); pr = load_label(predp, _W["ds"])
    dpc = dice_per_class(pr, gt, _W["classes"])
    if _W.get("no_hd"):
        # Dice is exact at full-res & fast; nHD is O(n^2) -> skip it (HD ranks
        # are scale-robust at 160px, validated dossier §0b). CRITICAL: the official
        # weighted_image_scores() ALSO computes nHD internally (calls hausdorff_per_class)
        # -> calling it here would re-trigger the O(n^2) timeout even with --no-hd
        # (the bug that timed out job 6208713). Compute weighted Dice DIRECTLY from
        # dpc, exactly replicating the official w_dice/weight_total. nHD -> NaN.
        wdice = sum(dpc[c.label_id] * c.weight for c in _W["classes"]) / _W["wt"]
        hpc = {c.label_id: float("nan") for c in _W["classes"]}
        return wdice, float("nan"), dpc, hpc
    s = weighted_image_scores(pr, gt, classes=_W["classes"], weight_total=_W["wt"])
    hpc = hausdorff_per_class(pr, gt, _W["classes"])
    return s["dice"], s["hd"], dpc, hpc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["t1", "t2"], required=True)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--downscale", type=int, default=160)
    ap.add_argument("--val-cases", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel frames (full-res nHD is O(n^2); use ~CPUs)")
    ap.add_argument("--no-hd", action="store_true",
                    help="skip O(n^2) nHD; emit exact Dice only (full-res safe). "
                         "HD ranks are scale-robust at 160px (dossier §0b).")
    args = ap.parse_args()

    if args.task == "t1":
        from metrics.classes_merged import CLASSES_MERGED as CLASSES, WEIGHT_TOTAL_MERGED as WT
    else:
        from metrics.classes import CLASSES, WEIGHT_TOTAL as WT
    W3_IDS = [c.label_id for c in CLASSES if c.weight == 3]
    NAME = {c.label_id: c.name for c in CLASSES}

    val_cases = set(args.val_cases) if args.val_cases else None
    gt_files = sorted(args.gt.glob("*.png"))
    if val_cases:
        gt_files = [p for p in gt_files if case_of(p.stem) in val_cases]
    if not gt_files:
        raise SystemExit(f"No GT PNGs to score in {args.gt} (val_cases={val_cases})")

    gt_strs = [str(p) for p in gt_files]
    img_dice, img_hd = [], []
    pdice, phd = defaultdict(list), defaultdict(list)
    n = 0
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers, initializer=_init_worker,
                  initargs=(str(args.pred), args.downscale, CLASSES, WT, args.no_hd)) as pool:
            results = pool.map(_score_one, gt_strs)
    else:
        _init_worker(str(args.pred), args.downscale, CLASSES, WT, args.no_hd)
        results = [_score_one(g) for g in gt_strs]
    for r in results:
        if r is None:
            continue
        d, h, dpc, hpc = r
        img_dice.append(d); img_hd.append(h)
        for cid in dpc:
            pdice[cid].append(dpc[cid]); phd[cid].append(hpc[cid])
        n += 1

    final_dice = float(np.mean(img_dice)); final_hd = float(np.nanmean(img_hd))
    w3_dice = float(np.mean([np.mean(pdice[c]) for c in W3_IDS if pdice[c]]))
    w3_hd = float(np.nanmean([np.nanmean(phd[c]) for c in W3_IDS if phd[c]]))

    print("=" * 64)
    print(f"TIGER {args.task.upper()} challenge score  (n={n} val frames, "
          f"downscale={args.downscale or 'FULL'})")
    print("=" * 64)
    print(f"  weighted Dice (all, /{WT}) : {final_dice:.4f}")
    print(f"  weighted nHD  (lower better): {final_hd:.4f}")
    print(f"  --- w=3 tiny-structure SUB-AVERAGE ({len(W3_IDS)} classes) ---")
    print(f"  w=3 mean Dice : {w3_dice:.4f}   w=3 mean nHD : {w3_hd:.4f}")
    print("  --- per-class mean Dice ---")
    for c in CLASSES:
        d = np.mean(pdice[c.label_id]) if pdice[c.label_id] else float("nan")
        print(f"    [{c.label_id:2d} w{c.weight}] {c.name:32s} Dice={d:.4f}")

    if args.out:
        res = {"task": args.task, "n_val": n, "downscale": args.downscale,
               "weighted_dice": final_dice, "weighted_nhd": final_hd,
               "w3_dice": w3_dice, "w3_nhd": w3_hd,
               "per_class_dice": {NAME[c.label_id]: (float(np.mean(pdice[c.label_id]))
                                  if pdice[c.label_id] else None) for c in CLASSES}}
        args.out.write_text(json.dumps(res, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
