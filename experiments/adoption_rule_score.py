#!/usr/bin/env python3
"""Execute the PRE-REGISTERED Tiger adoption rule.

Compares the INCUMBENT (shipped; trained on 140 frames, ALL from centre 1) against the RETRAIN
(new 40-case / 524-frame / 6-centre release, same recipe) on the SAME frames, restricted to
held-out val cases from centres 2, 3, 4, 6 and 7 — centres the incumbent never trained on.

WHY THE RESTRICTION: our centre-stratified val also holds out 3 centre-1 cases, and the incumbent
may have TRAINED on those. Scoring both on the full val set would hand the incumbent frames it has
already seen. Comparing against the recorded 0.7092 / 0.552 would be worse still — a different
protocol, a different split, and a different scoring resolution.

ADOPT only if the retrain wins here. Ties go to the frozen artifact, which is already built, gated
and submitted. T1 and T2 are decided INDEPENDENTLY.

Scored with the organizers' own `metrics.metrics` — weighted Dice and weighted normalised Hausdorff
across the full class registry, plus the w=3 clinical tier that dominates the ranking.
"""
from __future__ import annotations
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/users/USERNAME/Tiger-SQ-AI/challenge_repo")
from metrics.classes import CLASSES                      # noqa: E402  (31-class, fine / Task 2)
from metrics.classes_merged import CLASSES_MERGED        # noqa: E402  (16-class, merged / Task 1)
from metrics.metrics import weighted_image_scores, dice_per_class  # noqa: E402


def score_dir(gt_dir: str, pred_dir: str, classes) -> tuple:
    """Mean weighted Dice, weighted HD and w=3-tier Dice over every frame present in both dirs."""
    weight_total = sum(c.weight for c in classes)
    tier3 = [c for c in classes if c.weight == 3]
    dice, hd, tier = [], [], []
    for g in sorted(glob.glob(os.path.join(gt_dir, "*.png"))):
        p = os.path.join(pred_dir, os.path.basename(g))
        if not os.path.isfile(p):
            continue
        gt = np.array(Image.open(g))
        pr = np.array(Image.open(p))
        r = weighted_image_scores(pr, gt, classes, weight_total)
        dice.append(r["dice"])
        hd.append(r["hd"])
        d = dice_per_class(pr, gt, classes)
        tier.append(sum(d[c.label_id] for c in tier3) / len(tier3))
    mean = lambda v: float(np.mean(v)) if v else float("nan")   # noqa: E731
    return mean(dice), mean(hd), mean(tier), len(dice)


def main() -> int:
    W = sys.argv[1]
    print("\n" + "=" * 78)
    print("  PRE-REGISTERED ADOPTION RULE — identical frames, centres 2/3/4/6/7 only")
    print("  higher weighted Dice and w3 are better; LOWER weighted HD is better")
    print("=" * 78)

    any_scored = False
    for task, gt, inc, new, classes in (
        ("T1 merged", f"{W}/gt_211", f"{W}/p_t1_inc", f"{W}/p_t1_new", CLASSES_MERGED),
        ("T2 fine",   f"{W}/gt_212", f"{W}/p_t2_inc", f"{W}/p_t2_new", CLASSES),
    ):
        di, hi, ti, n1 = score_dir(gt, inc, classes)
        dn, hn, tn, n2 = score_dir(gt, new, classes)
        if n1 == 0 or n2 == 0:
            print(f"\n  {task}: NOT SCORED — incumbent {n1} frames, retrain {n2} frames.")
            print("     A missing prediction directory means that predict step failed; read the log.")
            continue
        any_scored = True
        if n1 != n2:
            print(f"\n  ⚠ {task}: frame counts differ ({n1} vs {n2}) — NOT a paired comparison.")
        print(f"\n  {task}   ({n1} frames)")
        print(f"    incumbent  140 fr / 1 centre    wDice {di:.4f}   wHD {hi:.4f}   w3 {ti:.4f}")
        print(f"    retrain    524 fr / 6 centres   wDice {dn:.4f}   wHD {hn:.4f}   w3 {tn:.4f}")
        print(f"    delta                           {dn - di:+.4f}          {hn - hi:+.4f}          {tn - ti:+.4f}")
        print(f"    >>> {'ADOPT the retrain' if tn > ti else 'KEEP the frozen artifact (tie or loss)'}")

    print("\n  NOTE: T2 above is SINGLE-SEED vs SINGLE-SEED. The shipped T2 is a 3-seed ensemble,")
    print("        so rerun once seeds 1 and 2 land to decide the ensemble, not one member.")
    print("=" * 78)
    return 0 if any_scored else 9


if __name__ == "__main__":
    sys.exit(main())
