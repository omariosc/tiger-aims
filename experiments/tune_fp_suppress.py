#!/usr/bin/env python3
"""
Tune per-class total-area FP-suppression thresholds on val (cases 14,15) to
maximise the OFFICIAL w3 sub-average Dice WITHOUT losing present-frame recall.

Strategy (per task, per w3 class independently — w3 is the deciding tier and the
classes don't overlap so per-class greedy == joint optimum on the sub-average):

  candidate total-area thresholds T in a grid (downscaled-px, matching the 160px
  fast scorer used for the sweep).  For each w3 class c:
    * RECALL GUARD: reject any T that would suppress class c on a frame where
      GT(c) is PRESENT (i.e. T > min present-frame total-area) -> never drop a TP.
      => only T <= min_{present frames} area(c) are admissible. This *cannot*
      reduce present-frame recall by construction.
    * Among admissible T, pick the one maximising the mean per-frame Dice of
      class c over ALL val frames (official dice_per_class).  Larger admissible T
      removes more absent-frame FP blobs (each lifts that frame's class-Dice 0->1).
  A min-blob filter of 2px is applied first to drop single-pixel speckle (also
  guarded the same way).

This is deliberately conservative: by capping T at the smallest present-frame
area we guarantee zero recall loss; the gain comes purely from absent-frame FP
removal. Reported: w3 sub-avg Dice/nHD before->after, present-frame recall delta
(must be 0), #zeroed-classes defused.

Outputs the tuned thresholds to a JSON consumed by the rescore + docker.
"""
from __future__ import annotations
import argparse, sys, json, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

CHALLENGE_REPO = Path("/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, str(CHALLENGE_REPO))
sys.path.insert(0, str(Path(__file__).parent))
from metrics.metrics import dice_per_class, hausdorff_per_class
from fp_suppress import apply_area_filter, _components


def case_of(stem): return stem.rsplit("_", 1)[0]


def load(p, ds):
    a = np.array(Image.open(p).convert("RGB"))[:, :, 0].astype(np.uint8)
    if ds and max(a.shape) > ds:
        h, w = a.shape; s = ds / max(h, w)
        a = np.array(Image.fromarray(a).resize((max(1, round(w*s)), max(1, round(h*s))), Image.NEAREST))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["t1", "t2"], required=True)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--downscale", type=int, default=160)
    ap.add_argument("--val-cases", nargs="*", default=["center_1_case_14", "center_1_case_15"])
    ap.add_argument("--min-blob", type=int, default=2, help="drop CC smaller than this (px)")
    ap.add_argument("--no-hd", action="store_true", help="skip nHD in score (full-res nHD is O(n^2)); Dice only")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.task == "t1":
        from metrics.classes_merged import CLASSES_MERGED as CLASSES, WEIGHT_TOTAL_MERGED as WT
    else:
        from metrics.classes import CLASSES, WEIGHT_TOTAL as WT
    W3 = [c.label_id for c in CLASSES if c.weight == 3]
    NAME = {c.label_id: c.name for c in CLASSES}
    ds = args.downscale
    valset = set(args.val_cases)

    gts = [g for g in sorted(glob.glob(str(args.gt / "*.png"))) if case_of(Path(g).stem) in valset]
    frames = []   # (gt_arr, pred_arr)
    for g in gts:
        p = args.pred / Path(g).name
        if not p.exists():
            print(f"  WARN missing pred {Path(g).name}"); continue
        frames.append((load(g, ds), load(str(p), ds)))
    n = len(frames)

    # --- per-class total-area threshold tuned on the EXACT post-filter prediction.
    # min_blob is applied per-class and ONLY when it never empties a present frame
    # (tiny classes whose TP blobs are <min_blob keep min_blob=0). The total-area
    # threshold is then the largest value that removes the most absent-frame FP while
    # NEVER zeroing a present frame (exact recall guard, evaluated below).
    # We collect, per class, the post-min_blob total area on present vs absent frames.
    def per_class_areas(mb_map):
        pres = defaultdict(list); abst = defaultdict(list)
        for gt, pr in frames:
            pr_f = apply_area_filter(pr, min_blob=mb_map)
            for c in W3:
                ga = bool((gt == c).any()); pa = int((pr_f == c).sum())
                if pa == 0:
                    # a present frame already empty after min_blob = recall loss from min_blob
                    if ga: pres[c]  # (recorded below via guard); skip
                    continue
                (pres if ga else abst)[c].append(pa)
        return pres, abst

    # decide per-class min_blob: use args.min_blob only if it empties no present frame
    min_blob = {}
    for c in W3:
        empties = 0
        for gt, pr in frames:
            if not (gt == c).any():
                continue
            before = int((pr == c).any())
            after = int((apply_area_filter(pr, min_blob={c: args.min_blob}) == c).any())
            if before and not after:
                empties += 1
        min_blob[c] = args.min_blob if empties == 0 else 0

    pres_a, abst_a = per_class_areas(min_blob)
    min_total = {}
    for c in W3:
        pres = sorted(pres_a[c]); abst = sorted(abst_a[c])
        if not abst:
            min_total[c] = 0; continue                 # no FP to suppress -> no-op
        # admissible cap = smallest present-frame post-filter area (strictly keep all TP).
        # entirely-absent class (no GT present) -> safe up to max FP area.
        cap = (pres[0] if pres else (max(abst) + 1))
        cands = [0] + sorted({a + 1 for a in abst if a + 1 <= cap})
        min_total[c] = int(max(cands))

    # --- score before / after with the chosen thresholds
    def score(use_filter):
        pdice, phd = defaultdict(list), defaultdict(list)
        fp_zero = defaultdict(int); recall_hit = defaultdict(int)
        for gt, pr in frames:
            if use_filter:
                pr = apply_area_filter(pr, min_blob=min_blob, min_total=min_total)
            dpc = dice_per_class(pr, gt, CLASSES)
            hpc = None if args.no_hd else hausdorff_per_class(pr, gt, CLASSES)
            for c in W3:
                pdice[c].append(dpc[c])
                if hpc is not None: phd[c].append(hpc[c])
                ga = bool((gt == c).any()); pa = bool((pr == c).any())
                if (not ga) and pa: fp_zero[c] += 1
                if ga and (not pa): recall_hit[c] += 1
        w3d = float(np.mean([np.mean(pdice[c]) for c in W3 if pdice[c]]))
        w3h = float(np.mean([np.mean(phd[c]) for c in W3 if phd[c]])) if any(phd[c] for c in W3) else float("nan")
        nz = sum(1 for c in W3 if fp_zero[c])
        miss = sum(1 for c in W3 if recall_hit[c])
        per = {NAME[c]: dict(dice=float(np.mean(pdice[c])), fp0=fp_zero[c], miss0=recall_hit[c]) for c in W3}
        return w3d, w3h, nz, miss, per

    b_d, b_h, b_nz, b_miss, b_per = score(False)
    a_d, a_h, a_nz, a_miss, a_per = score(True)

    print("=" * 78)
    print(f"FP-SUPPRESS TUNE  {args.task}  (n={n} val, ds={ds}, min_blob={args.min_blob})")
    print("=" * 78)
    print("per-w3 thresholds (total-area, downscaled-px) + Dice before->after:")
    for c in W3:
        print(f"  [{c:2d}] {NAME[c]:30s} T_total={min_total[c]:5d}  "
              f"Dice {b_per[NAME[c]]['dice']:.3f}->{a_per[NAME[c]]['dice']:.3f}  "
              f"FP0 {b_per[NAME[c]]['fp0']:2d}->{a_per[NAME[c]]['fp0']:2d}  "
              f"miss {b_per[NAME[c]]['miss0']}->{a_per[NAME[c]]['miss0']}")
    print(f"w3 sub-avg Dice : {b_d:.4f} -> {a_d:.4f}   (Δ {a_d-b_d:+.4f})")
    print(f"w3 sub-avg nHD  : {b_h:.4f} -> {a_h:.4f}   (Δ {a_h-b_h:+.4f}, lower better)")
    print(f"#w3 classes with FP-zero : {b_nz} -> {a_nz}  (defused {b_nz-a_nz})")
    print(f"#w3 present-frame misses : {b_miss} -> {a_miss}  (recall guard: must be ==)")

    # resolution-invariant fractional thresholds (area / total px) for variable-size
    # docker inputs. Reference px count = the px count frames were tuned at.
    ref_h, ref_w = frames[0][0].shape
    ref_px = float(ref_h * ref_w)
    min_total_frac = {str(k): (v / ref_px) for k, v in min_total.items()}
    min_blob_frac = {str(k): (v / ref_px) for k, v in min_blob.items()}

    args.out.write_text(json.dumps(dict(
        task=args.task, downscale=ds, min_blob_arg=args.min_blob,
        ref_px=ref_px, ref_hw=[ref_h, ref_w],
        min_blob={str(k): v for k, v in min_blob.items()},
        min_total={str(k): v for k, v in min_total.items()},
        min_blob_frac=min_blob_frac, min_total_frac=min_total_frac,
        w3_dice_before=b_d, w3_dice_after=a_d, w3_dice_delta=a_d - b_d,
        w3_nhd_before=b_h, w3_nhd_after=a_h,
        fp_classes_before=b_nz, fp_classes_after=a_nz,
        present_miss_before=b_miss, present_miss_after=a_miss,
        per_class={NAME[c]: dict(id=c, T_total=min_total[c],
                                 dice_before=b_per[NAME[c]]['dice'], dice_after=a_per[NAME[c]]['dice'],
                                 fp0_before=b_per[NAME[c]]['fp0'], fp0_after=a_per[NAME[c]]['fp0'])
                   for c in W3}), indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
