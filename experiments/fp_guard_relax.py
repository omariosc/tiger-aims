#!/usr/bin/env python3
"""
TIGER SQ-AI — is the FP-suppress RECALL GUARD leaving w3 score on the table? (2026-07-20)

The banked FP-suppress thresholds are ALREADY PER-CLASS and w3-only, but they are tuned
under a HARD guard: "never zero a class on a frame where it is present" (zero recall loss).
That guard is only optimal if no threshold has a POSITIVE net payoff. Under the official
metric the two error types cost EXACTLY the same, which makes the trade computable:

  * absent frame with an FP blob   : Dice 0.0  (one-empty).  Remove it -> both-empty -> 1.0  => +1.0
  * present frame we wrongly zero  : Dice d_i (whatever it was) -> one-empty 0.0        => -d_i

So for a per-class min_total threshold T:
    net(T) = [ #absent-frames-removed(T) * 1.0  -  SUM(d_i over present-frames killed(T)) ] / n_frames
and the OPTIMAL T maximises net(T). The banked guard pins T to the largest value with
0 present-frames killed. This script computes net(T) over ALL candidate T per class and
reports the guard-limited vs net-optimal choice, so the "already tuned" claim is tested,
not assumed. (nHD moves in lockstep: one-empty nHD=1.0, both-empty=0.0 — dossier 0g.)

Dice-only => CPU-cheap (no O(n^2) Hausdorff).

Usage:
  python fp_guard_relax.py --task t2 --pred <champion_id_dir> --gt <labelsTr> \
      --val-cases center_1_case_14 center_1_case_15
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

CHALLENGE_REPO = Path("/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, str(CHALLENGE_REPO))


def case_of(stem): return stem.rsplit("_", 1)[0]


def load_ids(p):
    a = np.array(Image.open(p))
    return (a[:, :, 0] if a.ndim == 3 else a).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["t1", "t2"], required=True)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--val-cases", nargs="*", default=None)
    ap.add_argument("--min-blob-frac", type=float, default=9.645061728395062e-07,
                    help="banked min_blob fraction applied before the total-area test")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--loco", action="store_true",
                    help="ALSO report the HONEST leave-one-case-out estimate: fit per-class "
                         "thresholds on one patient, apply to the held-out patient, pool. "
                         "The all-frames 'optimal' number above is ORACLE/test-fit and NOT claimable.")
    a = ap.parse_args()

    if a.task == "t1":
        from metrics.classes_merged import CLASSES_MERGED as CLASSES
    else:
        from metrics.classes import CLASSES
    W3 = [c for c in CLASSES if c.weight == 3]
    NAME = {c.label_id: c.name for c in CLASSES}

    val = set(a.val_cases) if a.val_cases else None
    gts = sorted(a.gt.glob("*.png"))
    if val:
        gts = [p for p in gts if case_of(p.stem) in val]

    # per class: list of (pred_area_after_minblob, gt_present, dice_now)
    data = {c.label_id: [] for c in W3}
    frame_case = []
    for gp in gts:
        frame_case.append(case_of(gp.stem))
        pp = a.pred / gp.name
        if not pp.exists():
            continue
        gt = load_ids(gp); pr = load_ids(pp)
        px = gt.shape[0] * gt.shape[1]
        mb = int(round(a.min_blob_frac * px))
        for c in W3:
            cid = c.label_id
            g = (gt == cid); p = (pr == cid)
            # apply banked min_blob first (drop specks), then measure total area
            if p.any() and mb > 1:
                lab, n = ndimage.label(p)
                if n:
                    sizes = ndimage.sum(p, lab, range(1, n + 1))
                    keep = np.zeros(n + 1, bool)
                    keep[1:] = sizes >= mb
                    p = keep[lab]
            gs, ps = int(g.sum()), int(p.sum())
            if gs == 0 and ps == 0:
                dice = 1.0
            elif gs == 0 or ps == 0:
                dice = 0.0
            else:
                dice = 2 * int((g & p).sum()) / (gs + ps)
            data[cid].append((ps, gs > 0, dice, px))

    n = len(gts)
    print("=" * 108)
    print(f"TIGER {a.task.upper()} — FP-suppress recall-guard relaxation test "
          f"(n={n} frames, pred={a.pred.name})")
    print("=" * 108)
    print(f"{'w3 class':>34} | {'GTfr':>4} {'FPfr':>4} | {'guardT':>9} {'netG':>7} | "
          f"{'optT':>9} {'netOpt':>7} {'kill':>4} | {'gain':>7}")
    print("-" * 108)
    tot_guard = tot_opt = 0.0
    detail = {}
    for c in W3:
        cid = c.label_id
        rows = data[cid]
        if not rows:
            continue
        px = rows[0][3]
        absent_fp = sorted([r[0] for r in rows if (not r[1]) and r[0] > 0])
        present = [(r[0], r[2]) for r in rows if r[1]]
        gtfr = sum(1 for r in rows if r[1]); fpfr = len(absent_fp)
        # candidate thresholds = every observed area boundary
        cands = sorted(set([0] + [x + 1 for x in absent_fp] + [x for x, _ in present]))

        def net(T):
            removed = sum(1 for x in absent_fp if x < T)
            lost = sum(d for x, d in present if x < T and x > 0)
            killed = sum(1 for x, d in present if x < T and x > 0)
            return (removed * 1.0 - lost) / n, removed, killed

        # guard-limited: largest T killing NO present frame
        guardT = 0
        for T in cands:
            if all(not (0 < x < T) for x, _ in present):
                guardT = max(guardT, T)
        gnet, grem, gk = net(guardT)
        # net-optimal over all candidates
        bestT, bnet, bk = guardT, gnet, gk
        for T in cands:
            v, rem, k = net(T)
            if v > bnet + 1e-12:
                bestT, bnet, bk = T, v, k
        tot_guard += gnet; tot_opt += bnet
        gain = bnet - gnet
        flag = " <== RELAX WINS" if gain > 1e-9 else ""
        print(f"{NAME[cid][:34]:>34} | {gtfr:4d} {fpfr:4d} | {guardT:9d} {gnet:+7.4f} | "
              f"{bestT:9d} {bnet:+7.4f} {bk:4d} | {gain:+7.4f}{flag}")
        detail[NAME[cid]] = {"gt_frames": gtfr, "fp_frames": fpfr,
                             "guard_T": guardT, "guard_net": gnet,
                             "opt_T": bestT, "opt_net": bnet, "opt_kills": bk,
                             "gain_over_guard": gain}
    print("-" * 108)
    nw3 = len([c for c in W3 if data[c.label_id]])
    print(f"w3 sub-average Dice gain from FP-suppress: guard-limited {tot_guard/nw3:+.4f}  "
          f"net-optimal {tot_opt/nw3:+.4f}   => RELAXATION HEADROOM = {(tot_opt-tot_guard)/nw3:+.4f}")
    print("(gain is per-class-mean Dice, averaged over the w3 tier = the deciding metric)")
    loco_res = None
    if a.loco:
        cases = sorted(set(frame_case))
        print()
        print("=" * 108)
        print("HONEST LEAVE-ONE-CASE-OUT (fit thresholds on one patient, apply to the held-out "
              f"patient, pool) — {len(cases)} cases")
        print("=" * 108)

        def net_on(rows_subset, T):
            """mean per-frame dice on a subset, if class is zeroed when area < T."""
            out = []
            for ps, present, dice, _px in rows_subset:
                if 0 < ps < T:      # class gets suppressed on this frame
                    out.append(1.0 if not present else 0.0)
                else:
                    out.append(dice)
            return float(np.mean(out)) if out else float("nan")

        base_cls, guard_cls, opt_cls, loco_cls = [], [], [], []
        print(f"{'w3 class':>34} | {'base':>7} {'guardLOCO':>9} {'optLOCO':>8} | {'Δ vs base':>9}")
        print("-" * 108)
        for c in W3:
            cid = c.label_id
            rows = data[cid]
            if not rows:
                continue
            idx_by_case = {cs: [i for i, x in enumerate(frame_case) if x == cs] for cs in cases}
            base_v = float(np.mean([r[2] for r in rows]))
            pooled_guard, pooled_opt = [], []
            for held in cases:
                fit_idx = [i for cs in cases if cs != held for i in idx_by_case[cs]]
                held_idx = idx_by_case[held]
                fit_rows = [rows[i] for i in fit_idx]
                held_rows = [rows[i] for i in held_idx]
                fit_absent = sorted([r[0] for r in fit_rows if (not r[1]) and r[0] > 0])
                fit_present = [(r[0], r[2]) for r in fit_rows if r[1]]
                cands = sorted(set([0] + [x + 1 for x in fit_absent] + [x for x, _ in fit_present]))
                nfit = max(1, len(fit_rows))

                def fitnet(T):
                    rem = sum(1 for x in fit_absent if x < T)
                    lost = sum(d for x, d in fit_present if 0 < x < T)
                    return (rem - lost) / nfit
                gT = 0
                for T in cands:
                    if all(not (0 < x < T) for x, _ in fit_present):
                        gT = max(gT, T)
                oT, oval = gT, fitnet(gT)
                for T in cands:
                    if fitnet(T) > oval + 1e-12:
                        oT, oval = T, fitnet(T)
                pooled_guard += [net_on([r], gT) for r in held_rows]
                pooled_opt += [net_on([r], oT) for r in held_rows]
            g_v = float(np.mean(pooled_guard)); o_v = float(np.mean(pooled_opt))
            base_cls.append(base_v); guard_cls.append(g_v); opt_cls.append(o_v)
            print(f"{NAME[cid][:34]:>34} | {base_v:7.4f} {g_v:9.4f} {o_v:8.4f} | {o_v-base_v:+9.4f}")
        print("-" * 108)
        bw, gw, ow = np.mean(base_cls), np.mean(guard_cls), np.mean(opt_cls)
        print(f"w3 sub-avg Dice: no-suppress {bw:.4f} | guard-LOCO {gw:.4f} ({gw-bw:+.4f}) | "
              f"RELAXED-LOCO {ow:.4f} ({ow-bw:+.4f})")
        print(f">>> HONEST relaxation gain over the guard = {ow-gw:+.4f}   "
              f"(vs the ORACLE {(tot_opt-tot_guard)/nw3:+.4f} above)")
        loco_res = {"w3_base": float(bw), "w3_guard_loco": float(gw), "w3_relaxed_loco": float(ow),
                    "honest_relax_gain": float(ow - gw)}

    if a.out:
        a.out.write_text(json.dumps({"task": a.task, "n": n, "loco": loco_res,
                                     "w3_guard_gain": tot_guard / nw3,
                                     "w3_opt_gain": tot_opt / nw3,
                                     "w3_relax_headroom": (tot_opt - tot_guard) / nw3,
                                     "per_class": detail}, indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
