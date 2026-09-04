#!/usr/bin/env python3
"""
Tiger SQ-AI FAILURE-ANALYSIS / DIAGNOSTIC pass for the banked seg models.

Goes beyond score_tiger_seg.py: emits, per task, a worst->best PER-CLASS table of
  Dice | nHD | weight-tier | #frames-present | empty-class hits
split by w1/w2/w3 tier (w3 sub-average is the ranking-deciding stratum), and the
"one-empty catastrophe" detector — for every w3 class, how many val frames have
  GT-empty + a predicted FALSE-POSITIVE blob  -> Dice forced to 0.0  (or
  GT-present + pred-empty -> miss). These are the recoverable / floor strata.

CPU-light: nNHD downscaled (default 160px NEAREST) exactly like the banked scores.
Reuses the OFFICIAL challenge primitives (dice_per_class / hausdorff_per_class).

Usage (seg_env):
  python diagnose_seg.py --task t1 \
      --gt <nnUNet_raw>/Dataset201_TigerT1coarse/labelsTr \
      --pred <nnunet>/predTs/t1 --val-cases center_1_case_14 center_1_case_15 \
      [--downscale 160] [--out t1_diag.json]
  python diagnose_seg.py --task t2 --gt .../Dataset202_TigerT2fine/labelsTr \
      --pred .../predTs/t2_ft ...
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

CHALLENGE_REPO = Path("/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, str(CHALLENGE_REPO))
from metrics.metrics import dice_per_class, hausdorff_per_class  # official


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["t1", "t2"], required=True)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--downscale", type=int, default=160)
    ap.add_argument("--val-cases", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.task == "t1":
        from metrics.classes_merged import CLASSES_MERGED as CLASSES
        ntag = "16-cls MERGED (internal t1)"
    else:
        from metrics.classes import CLASSES
        ntag = "31-cls FINE (internal t2)"
    NAME = {c.label_id: c.name for c in CLASSES}
    WGT = {c.label_id: c.weight for c in CLASSES}

    val_cases = set(args.val_cases) if args.val_cases else None
    gt_files = sorted(args.gt.glob("*.png"))
    if val_cases:
        gt_files = [p for p in gt_files if case_of(p.stem) in val_cases]
    if not gt_files:
        raise SystemExit(f"No GT PNGs in {args.gt} (val_cases={val_cases})")

    pdice, phd = defaultdict(list), defaultdict(list)
    present = defaultdict(int)          # #frames GT has this class
    pred_present = defaultdict(int)     # #frames pred has this class
    one_empty_fp = defaultdict(int)     # GT empty + pred blob -> Dice=0 catastrophe
    one_empty_miss = defaultdict(int)   # GT present + pred empty -> Dice=0 miss
    n = 0
    for gtp in gt_files:
        predp = args.pred / gtp.name
        if not predp.exists():
            print(f"  WARN missing pred {gtp.name}; skip"); continue
        gt = load_label(gtp, args.downscale)
        pr = load_label(predp, args.downscale)
        dpc = dice_per_class(pr, gt, CLASSES)
        hpc = hausdorff_per_class(pr, gt, CLASSES)
        for c in CLASSES:
            cid = c.label_id
            g_has = bool((gt == cid).any())
            p_has = bool((pr == cid).any())
            pdice[cid].append(dpc[cid]); phd[cid].append(hpc[cid])
            if g_has: present[cid] += 1
            if p_has: pred_present[cid] += 1
            if (not g_has) and p_has:   one_empty_fp[cid] += 1
            if g_has and (not p_has):   one_empty_miss[cid] += 1
        n += 1

    rows = []
    for c in CLASSES:
        cid = c.label_id
        d = float(np.mean(pdice[cid])) if pdice[cid] else float("nan")
        h = float(np.mean(phd[cid])) if phd[cid] else float("nan")
        rows.append(dict(id=cid, name=NAME[cid], w=WGT[cid], dice=d, nhd=h,
                         gt_frames=present[cid], pred_frames=pred_present[cid],
                         fp_zero=one_empty_fp[cid], miss_zero=one_empty_miss[cid]))
    rows.sort(key=lambda r: (r["dice"] if r["dice"] == r["dice"] else 9))  # worst->best

    print("=" * 92)
    print(f"DIAGNOSE {args.task.upper()}  {ntag}  (n={n} val frames, downscale="
          f"{args.downscale or 'FULL'})")
    print("=" * 92)
    print(f"{'cls':36s} w  Dice    nHD    GTfr predfr  FP0  MISS0")
    for r in rows:
        flag = ""
        if r["w"] == 3 and r["fp_zero"]:   flag += " <ONE-EMPTY-FP"
        if r["w"] == 3 and r["miss_zero"]: flag += " <MISS"
        print(f"[{r['id']:2d}] {r['name']:31s} {r['w']}  {r['dice']:.3f}  "
              f"{r['nhd']:.3f}  {r['gt_frames']:3d}  {r['pred_frames']:4d}   "
              f"{r['fp_zero']:2d}   {r['miss_zero']:3d}{flag}")

    for tier in (3, 2, 1):
        tr = [r for r in rows if r["w"] == tier and r["dice"] == r["dice"]]
        if not tr: continue
        md = float(np.mean([r["dice"] for r in tr]))
        mh = float(np.mean([r["nhd"] for r in tr]))
        print(f"  --- w={tier} tier ({len(tr)} cls): mean Dice {md:.4f}  mean nHD {mh:.4f}")

    w3 = [r for r in rows if r["w"] == 3]
    n_fp_hit = sum(1 for r in w3 if r["fp_zero"])
    n_miss_hit = sum(1 for r in w3 if r["miss_zero"])
    print(f"  ONE-EMPTY trap: {n_fp_hit}/{len(w3)} w3 classes have >=1 FP-blob-on-absent "
          f"frame (Dice forced 0); {n_miss_hit}/{len(w3)} have >=1 total-miss frame.")

    if args.out:
        args.out.write_text(json.dumps(
            dict(task=args.task, n_val=n, downscale=args.downscale, rows=rows,
                 w3_fp_classes=n_fp_hit, w3_miss_classes=n_miss_hit), indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
