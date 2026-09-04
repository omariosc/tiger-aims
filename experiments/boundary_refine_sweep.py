#!/usr/bin/env python3
"""
TIGER SQ-AI — nHD-TARGETED boundary-refinement post-processing sweep (2026-07-20).

RATIONALE (the untried half of the metric): the score is weighted Dice + normalized
Hausdorff Distance. Every refuted lever (rare-class cascade, SAM2 refiner, synth) targeted
DICE. nHD is the MAX boundary excursion — it is dominated by *outlier specks* far from the
true structure, which barely move Dice. Removing them should cut nHD sharply at ~zero Dice
cost. This sweeps training-free post-processing on CACHED champion predictions and reports
**Dice AND nHD SEPARATELY** (overall weighted + w3 sub-average) so we know which term moved.

Variants (all per-class, argmax id masks):
  baseline        — champion as-is
  rel<X>          — RELATIVE island removal: drop components < X * (largest component of
                    that class in that frame). Targets nHD outliers; never deletes the class.
  abs<F>          — ABSOLUTE island removal: drop components < F * image_px (keeps largest).
  largestcc       — keep ONLY the largest component per class (most aggressive nHD cut).
  close<R>        — binary closing (disk R) per w3 class, writing ONLY onto background
                    (smooths contours / fills gaps without stealing other classes).
  rel<X>+close<R> — combination.
Scoring uses the UNMODIFIED official primitives (metrics.metrics.dice_per_class /
hausdorff_per_class) at 160px (nHD is O(n^2) -> intractable full-res; HD ranks are
scale-robust, dossier 0b/0i). Official semantics: both-empty Dice=1.0/nHD=0.0,
one-empty Dice=0.0/nHD=1.0.

Usage:
  python boundary_refine_sweep.py --task t2 --pred <champion_id_dir> --gt <labelsTr> \
      --val-cases center_1_case_14 center_1_case_15 --out sweep_t2.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy import ndimage

CHALLENGE_REPO = Path("/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, str(CHALLENGE_REPO))
from metrics.metrics import dice_per_class, hausdorff_per_class


def case_of(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def load_ids(p: Path) -> np.ndarray:
    a = np.array(Image.open(p))
    if a.ndim == 3:
        a = a[:, :, 0]
    return a.astype(np.uint8)


def downscale(lab: np.ndarray, m: int) -> np.ndarray:
    if not m or max(lab.shape) <= m:
        return lab
    h, w = lab.shape
    s = m / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    return np.array(Image.fromarray(lab).resize((nw, nh), Image.NEAREST))


# ---------------------------------------------------------------- refiners
def island_filter(ids, class_ids, rel=None, absfrac=None, keep_largest=True):
    """Remove small connected components per class. rel = fraction of that class's
    LARGEST component; absfrac = fraction of total image pixels. Largest kept always."""
    out = ids.copy()
    px = ids.shape[0] * ids.shape[1]
    for c in class_ids:
        m = (ids == c)
        if not m.any():
            continue
        lab, n = ndimage.label(m)
        if n <= 1:
            continue
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        big = sizes.max()
        thr = 0.0
        if rel is not None:
            thr = max(thr, rel * big)
        if absfrac is not None:
            thr = max(thr, absfrac * px)
        keep_idx = int(np.argmax(sizes)) + 1
        for i, s in enumerate(sizes, start=1):
            if s < thr and not (keep_largest and i == keep_idx):
                out[lab == i] = 0
    return out


def largest_cc(ids, class_ids):
    out = ids.copy()
    for c in class_ids:
        m = (ids == c)
        if not m.any():
            continue
        lab, n = ndimage.label(m)
        if n <= 1:
            continue
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        out[m & (lab != keep)] = 0
    return out


def close_bg_only(ids, class_ids, radius):
    """Binary closing per class; new pixels written ONLY where currently background."""
    out = ids.copy()
    r = int(radius)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    se = (yy ** 2 + xx ** 2) <= r * r
    for c in class_ids:
        m = (ids == c)
        if not m.any():
            continue
        closed = ndimage.binary_closing(m, structure=se)
        new = closed & (out == 0)
        out[new] = c
    return out


def apply_variant(ids, name, w3_ids, all_ids):
    if name == "baseline":
        return ids
    cur = ids
    for part in name.split("+"):
        if part.startswith("rel"):
            cur = island_filter(cur, all_ids, rel=float(part[3:]))
        elif part.startswith("abs"):
            cur = island_filter(cur, all_ids, absfrac=float(part[3:]))
        elif part == "largestcc":
            cur = largest_cc(cur, all_ids)
        elif part.startswith("close"):
            cur = close_bg_only(cur, w3_ids, int(part[5:]))
        else:
            raise ValueError(f"unknown variant part {part}")
    return cur


# ---------------------------------------------------------------- worker
_W = {}


def _init(pred_dir, ds, classes, wt, variants, w3_ids, all_ids):
    _W.update(pred=Path(pred_dir), ds=ds, classes=classes, wt=wt,
              variants=variants, w3=w3_ids, allids=all_ids)


def _score_frame(gtp_str):
    gtp = Path(gtp_str)
    pp = _W["pred"] / gtp.name
    if not pp.exists():
        return None
    gt_full = load_ids(gtp)
    pr_full = load_ids(pp)
    gt = downscale(gt_full, _W["ds"])
    res = {}
    for v in _W["variants"]:
        # refine at FULL res (that's how it would ship), then downscale for scoring
        ref = apply_variant(pr_full, v, _W["w3"], _W["allids"])
        pr = downscale(ref, _W["ds"])
        dpc = dice_per_class(pr, gt, _W["classes"])
        hpc = hausdorff_per_class(pr, gt, _W["classes"])
        wd = sum(dpc[c.label_id] * c.weight for c in _W["classes"]) / _W["wt"]
        wh = sum(hpc[c.label_id] * c.weight for c in _W["classes"]) / _W["wt"]
        res[v] = (wd, wh, dpc, hpc)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["t1", "t2"], required=True)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--val-cases", nargs="*", default=None)
    ap.add_argument("--downscale", type=int, default=160)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--w3-only", action="store_true",
                    help="score ONLY the w3 tier. Makes FULL-RES nHD tractable: the O(n^2) "
                         "cost is dominated by huge classes (Background/Lung), while the thin "
                         "w3 classes have few fg pixels. wDice/wnHD then = w3-tier sums.")
    a = ap.parse_args()

    if a.task == "t1":
        from metrics.classes_merged import CLASSES_MERGED as CLASSES, WEIGHT_TOTAL_MERGED as WT
    else:
        from metrics.classes import CLASSES, WEIGHT_TOTAL as WT
    W3 = [c.label_id for c in CLASSES if c.weight == 3]
    ALL = [c.label_id for c in CLASSES if c.label_id != 0]
    NAME = {c.label_id: c.name for c in CLASSES}
    if a.w3_only:
        CLASSES = [c for c in CLASSES if c.weight == 3]
        WT = sum(c.weight for c in CLASSES)

    variants = a.variants or [
        "baseline",
        "rel0.05", "rel0.10", "rel0.20", "rel0.35",
        "abs0.00002", "abs0.0001",
        "largestcc",
        "close2", "close3",
        "rel0.10+close2", "rel0.20+close3",
    ]

    val = set(a.val_cases) if a.val_cases else None
    gts = sorted(a.gt.glob("*.png"))
    if val:
        gts = [p for p in gts if case_of(p.stem) in val]
    if not gts:
        raise SystemExit(f"no GT frames in {a.gt}")

    from multiprocessing import Pool
    with Pool(a.workers, initializer=_init,
              initargs=(str(a.pred), a.downscale, CLASSES, WT, variants, W3, ALL)) as pool:
        out = pool.map(_score_frame, [str(p) for p in gts])

    agg = {v: {"wd": [], "wh": [], "d": defaultdict(list), "h": defaultdict(list)} for v in variants}
    n = 0
    for r in out:
        if r is None:
            continue
        n += 1
        for v, (wd, wh, dpc, hpc) in r.items():
            agg[v]["wd"].append(wd); agg[v]["wh"].append(wh)
            for cid in dpc:
                agg[v]["d"][cid].append(dpc[cid]); agg[v]["h"][cid].append(hpc[cid])

    base = None
    rows = []
    for v in variants:
        A = agg[v]
        wd = float(np.mean(A["wd"])); wh = float(np.nanmean(A["wh"]))
        w3d = float(np.mean([np.mean(A["d"][c]) for c in W3 if A["d"][c]]))
        w3h = float(np.nanmean([np.nanmean(A["h"][c]) for c in W3 if A["h"][c]]))
        if v == "baseline":
            base = (wd, wh, w3d, w3h)
        rows.append((v, wd, wh, w3d, w3h))

    print("=" * 100)
    print(f"TIGER {a.task.upper()} nHD-targeted boundary-refinement sweep  "
          f"(n={n} val frames, {a.downscale}px, pred={a.pred.name})")
    print("=" * 100)
    print(f"{'variant':>16} | {'wDice':>7} {'Δ':>8} | {'wnHD↓':>7} {'Δ':>8} | "
          f"{'w3Dice':>7} {'Δ':>8} | {'w3nHD↓':>7} {'Δ':>8}")
    print("-" * 100)
    for v, wd, wh, w3d, w3h in rows:
        if v == "baseline":
            print(f"{v:>16} | {wd:7.4f} {'—':>8} | {wh:7.4f} {'—':>8} | "
                  f"{w3d:7.4f} {'—':>8} | {w3h:7.4f} {'—':>8}")
        else:
            print(f"{v:>16} | {wd:7.4f} {wd-base[0]:+8.4f} | {wh:7.4f} {wh-base[1]:+8.4f} | "
                  f"{w3d:7.4f} {w3d-base[2]:+8.4f} | {w3h:7.4f} {w3h-base[3]:+8.4f}")
    print("-" * 100)
    print("(nHD: LOWER is better -> a NEGATIVE Δ is an improvement)")

    if a.out:
        a.out.write_text(json.dumps(
            {"task": a.task, "pred": str(a.pred), "n": n, "downscale": a.downscale,
             "baseline": {"wdice": base[0], "wnhd": base[1], "w3dice": base[2], "w3nhd": base[3]},
             "variants": {v: {"wdice": wd, "wnhd": wh, "w3dice": w3d, "w3nhd": w3h}
                          for v, wd, wh, w3d, w3h in rows}}, indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
