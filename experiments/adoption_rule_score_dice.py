#!/usr/bin/env python3
"""Tiger adoption rule, DICE-ONLY -- the same decision, without the O(n^2) Hausdorff.

WHY A SECOND SCORER RATHER THAN AN EDIT: `adoption_rule_score.py` calls the official
`weighted_image_scores`, which computes `hausdorff_per_class` INTERNALLY. That is O(n^2) over all
foreground pixels at full resolution and is the documented Tiger trap (dossier 0i: a 10 h job that
produced one header line; 6208713 timed out at 4 h for exactly this reason). Job 7703505 finished
all four predictions in ~90 min and then stalled in scoring with a 3 h wall -- the same trap again.

THE FIX IS THE ONE THE DOSSIER ALREADY RECORDS (0l, job 6208713): weighted Dice is
    w_dice = sum(dice_per_class[c.label_id] * c.weight) / weight_total
verbatim from metrics.py, so it can be computed from `dice_per_class` alone and never touch the
Hausdorff path. Verified there to run in <1 s per config where the full call took 4 h.

THIS DOES NOT WEAKEN THE DECISION. The pre-registered adoption rule is on the **w=3 tier Dice**
(`adoption_rule_score.py`: `ADOPT if tn > ti`, where t is the w3 mean). Hausdorff never entered the
rule. It is reported by the original scorer as context only, and is still available for the single
final configuration, which is what 0i prescribes ("full-res reserved ONLY for the FINAL-PICK").
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, "/users/USERNAME/Tiger-SQ-AI/challenge_repo")
from metrics.classes import CLASSES                      # noqa: E402  fine / 31-class
from metrics.classes_merged import CLASSES_MERGED        # noqa: E402  merged / 16-class
from metrics.metrics import dice_per_class               # noqa: E402


def score_dir(gt_dir: str, pred_dir: str, classes) -> tuple:
    weight_total = sum(c.weight for c in classes)
    tier3 = [c for c in classes if c.weight == 3]
    wd, tier, n = [], [], 0
    for g in sorted(glob.glob(os.path.join(gt_dir, "*.png"))):
        p = os.path.join(pred_dir, os.path.basename(g))
        if not os.path.isfile(p):
            continue
        gt = np.array(Image.open(g)); pr = np.array(Image.open(p))
        d = dice_per_class(pr, gt, classes)
        # exact official replica of weighted_image_scores()["dice"], minus the Hausdorff call
        wd.append(sum(d[c.label_id] * c.weight for c in classes) / weight_total)
        tier.append(sum(d[c.label_id] for c in tier3) / len(tier3))
        n += 1
    mean = lambda v: float(np.mean(v)) if v else float("nan")   # noqa: E731
    return mean(wd), mean(tier), n


def main() -> int:
    W = sys.argv[1]
    print("\n" + "=" * 86)
    print("  TIGER PRE-REGISTERED ADOPTION RULE -- identical frames, centres 2/3/4/6/7 only")
    print("  DICE-ONLY (the rule is on the w=3 tier; Hausdorff was never part of it)")
    print("=" * 86)
    any_scored = False
    for task, gt, inc, new, classes in (
        ("T1 merged", f"{W}/gt_211", f"{W}/p_t1_inc", f"{W}/p_t1_new", CLASSES_MERGED),
        ("T2 fine",   f"{W}/gt_212", f"{W}/p_t2_inc", f"{W}/p_t2_new", CLASSES),
    ):
        di, ti, n1 = score_dir(gt, inc, classes)
        dn, tn, n2 = score_dir(gt, new, classes)
        if n1 == 0 or n2 == 0:
            print(f"\n  {task}: NOT SCORED (incumbent {n1} frames, retrain {n2}).")
            continue
        any_scored = True
        if n1 != n2:
            print(f"\n  !! {task}: frame counts differ ({n1} vs {n2}) -- NOT paired.")
        print(f"\n  {task}   ({n1} frames)")
        print(f"    incumbent  140 fr / 1 centre    wDice {di:.4f}    w3 {ti:.4f}")
        print(f"    retrain    524 fr / 6 centres   wDice {dn:.4f}    w3 {tn:.4f}")
        print(f"    delta                           {dn-di:+.4f}           {tn-ti:+.4f}")
        print(f"    >>> {'ADOPT the retrain' if tn > ti else 'KEEP the frozen artifact (tie or loss)'}")
    # The T2 note depends on WHAT IS IN THE WORKDIR: adopt_<jid> holds single-seed predictions,
    # adopt_ens_<jid> holds 3-seed prob-averaged ensembles. Printing the single-seed caveat over
    # an ensemble run would misdescribe the result, so decide it from the path, not from memory.
    if "adopt_ens" in os.path.abspath(W):
        print("\n  T2 above is ENSEMBLE vs ENSEMBLE (3 seeds per side, softmax prob-averaged) --")
        print("  this IS the ship decision, not an ablation.")
    else:
        print("\n  !! T2 above is SINGLE-SEED vs SINGLE-SEED. The SHIPPED T2 is a 3-seed ensemble, so")
        print("     this settles T1 but is only an ablation for T2 -- the ship decision needs")
        print("     ensemble-vs-ensemble, which requires re-predicting with --save_probabilities.")
    print("=" * 86)
    return 0 if any_scored else 9


if __name__ == "__main__":
    sys.exit(main())
