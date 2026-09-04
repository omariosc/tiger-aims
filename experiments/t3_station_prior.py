#!/usr/bin/env python3
"""
TIGER SQ-AI Task3 — T-1 ACQUISITION-STATION CO-VISIBILITY PRIOR (2026-07-20).

IDEA: every test frame is `<case>_<station>.png` and the published Docker contract
guarantees input filenames arrive UNCHANGED (and metrics/00_README.md mandates the
task3 row key be exactly that stem). So the ANNOTATED station `a` of each frame is
documented, contract-guaranteed input metadata. A station is far from independent of
the acquisition station: P(station s visible | annotated station a) is a 14x14
co-visibility matrix estimable from training cases with ZERO parameters and ZERO GPU.

Estimator (shrinkage FIXED a priori, see PREREGISTRATION.md):
    M[a][s] = (count[a][s] + alpha * pooled[s]) / (n[a] + alpha),  alpha = 2.0
where pooled[s] is the marginal prevalence of station s (the shrinkage target).

Scored with the UNMODIFIED official metrics.evaluate_cls.evaluate
(macro-F1 @ 0.5 + AUROC, station registry weights).

Modes:
  --loco            leave-one-CASE-out over all cases (THE HEADLINE, prior alone)
  --holdout C1 C2   fit on all OTHER cases, evaluate on these (head-to-head vs segcond)
  --segcond CSV     also score the deployed seg-cond preds + the PRE-REGISTERED
                    lambda=0.5 blend on the same rows
  --export-matrix P fit on ALL cases and dump the shipping matrix as JSON
"""
from __future__ import annotations
import argparse, csv, json, sys, tempfile
from pathlib import Path

import numpy as np

REPO = Path("/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, str(REPO))
from metrics.classes_stations import CLASSES_STATIONS, STATION_NAMES
from metrics.evaluate_cls import evaluate as official_evaluate

STATIONS = list(STATION_NAMES)
SIDX = {s: i for i, s in enumerate(STATIONS)}
ALPHA = 2.0          # pre-registered
LAMBDA = 0.5         # pre-registered blend


def load_rows(csv_path: Path):
    """-> list of (case, ann_station, y[14] binary)."""
    out = []
    with open(csv_path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) < 3:
                continue
            case, ann = row[0].strip(), row[1].strip()
            vis = {v.strip() for v in row[2].replace('"', '').split(",") if v.strip()}
            unknown = vis - set(STATIONS)
            if unknown:
                raise SystemExit(f"unknown station token(s) {unknown} in {case}")
            y = np.zeros(len(STATIONS), np.int8)
            for v in vis:
                y[SIDX[v]] = 1
            if ann not in SIDX:
                raise SystemExit(f"unknown annotated-frame token {ann!r} in {case}")
            out.append((case, ann, y))
    return out


def fit_matrix(rows, alpha=ALPHA):
    """M[a][s] with additive shrinkage toward the pooled marginal prevalence."""
    n = len(STATIONS)
    cnt = np.zeros((n, n), np.float64)
    tot = np.zeros(n, np.float64)
    ysum = np.zeros(n, np.float64)
    for _c, ann, y in rows:
        a = SIDX[ann]
        cnt[a] += y
        tot[a] += 1
        ysum += y
    pooled = ysum / max(1, len(rows))          # shrinkage target
    M = (cnt + alpha * pooled[None, :]) / (tot[:, None] + alpha)
    return M, pooled


def predict(M, pooled, rows):
    """-> (stems, P[n,14]) using ONLY the annotated-station token."""
    stems, P = [], []
    for case, ann, _y in rows:
        stems.append(f"{case}_{ann}")
        P.append(M[SIDX[ann]] if ann in SIDX else pooled)
    return stems, np.asarray(P)


def write_csv(path: Path, stems, mat, as_int=False):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id"] + STATIONS)
        for s, v in zip(stems, mat):
            w.writerow([s] + ([int(x) for x in v] if as_int
                              else [f"{float(x):.6f}" for x in v]))


def score(stems, y_true, y_pred, tag):
    """Score with the UNMODIFIED official evaluate_cls."""
    with tempfile.TemporaryDirectory() as td:
        g, p = Path(td) / "gt.csv", Path(td) / "pred.csv"
        write_csv(g, stems, y_true, as_int=True)
        write_csv(p, stems, y_pred)
        res = official_evaluate(g, p, classes=CLASSES_STATIONS)
    print(f"  {tag:<44} macro-F1 = {res['final_f1']:.5f}   AUROC = {res['final_auroc']:.5f}")
    return res


def load_pred_csv(path: Path, stems):
    """Load a prediction CSV, aligned to `stems` (row order irrelevant)."""
    import pandas as pd
    df = pd.read_csv(path)
    key = df.columns[0]
    df = df.set_index(key)
    miss = [s for s in stems if s not in df.index]
    if miss:
        raise SystemExit(f"{path} missing {len(miss)} stems, e.g. {miss[:3]}")
    return df.loc[stems, STATIONS].to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis-csv", type=Path,
                    default=Path("/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/synapse/"
                                 "lymph_node_station_visibility.csv"))
    ap.add_argument("--loco", action="store_true")
    ap.add_argument("--holdout", nargs="*", default=None)
    ap.add_argument("--segcond", type=Path, default=None)
    ap.add_argument("--export-matrix", type=Path, default=None)
    ap.add_argument("--per-centre", action="store_true",
                    help="PART-2 READINESS (run the day part-2 lands): re-estimate the matrix "
                         "PER CENTRE, quantify how far the off-diagonals move between centres, "
                         "and run leave-one-CENTRE-out. If centres disagree materially the prior "
                         "must be shrunk HARDER toward the pooled estimate (alpha up) or dropped.")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    rows = load_rows(a.vis_csv)
    cases = sorted({c for c, _, _ in rows})
    print(f"loaded {len(rows)} frames / {len(cases)} cases   (alpha={ALPHA}, lambda={LAMBDA} pre-registered)")
    results = {}

    if a.loco:
        print("\n=== HEADLINE: leave-one-CASE-out, PRIOR ALONE (all cases) ===")
        stems, P, Y = [], [], []
        for held in cases:
            tr = [r for r in rows if r[0] != held]
            te = [r for r in rows if r[0] == held]
            M, pooled = fit_matrix(tr)
            st, p = predict(M, pooled, te)
            stems += st; P.append(p); Y.append(np.stack([r[2] for r in te]))
        P = np.concatenate(P); Y = np.concatenate(Y)
        results["loco_prior"] = score(stems, Y, P, "LOCO prior-alone (10 folds)")

    if a.holdout:
        held = set(a.holdout)
        print(f"\n=== HEAD-TO-HEAD on {sorted(held)} (matrix fit on the OTHER "
              f"{len(cases)-len(held)} cases only) ===")
        tr = [r for r in rows if r[0] not in held]
        te = [r for r in rows if r[0] in held]
        M, pooled = fit_matrix(tr)
        stems, P = predict(M, pooled, te)
        Y = np.stack([r[2] for r in te])
        results["holdout_prior"] = score(stems, Y, P, "prior alone (fit on other cases)")
        if a.segcond:
            S = load_pred_csv(a.segcond, stems)
            results["holdout_segcond"] = score(stems, Y, S, "deployed seg-conditioned head")
            B = LAMBDA * P + (1.0 - LAMBDA) * S
            results["holdout_blend"] = score(
                stems, Y, B, f"PRE-REGISTERED blend lambda={LAMBDA}")

    if a.per_centre:
        def centre_of(case):           # center_1_case_6 -> center_1
            return case.split("_case_")[0]
        centres = sorted({centre_of(c) for c, _, _ in rows})
        print(f"\n=== PER-CENTRE ANALYSIS ({len(centres)} centre(s): {centres}) ===")
        if len(centres) < 2:
            print("  ONLY ONE CENTRE PRESENT -> centre-transfer of the prior is UNTESTED.")
            print("  The LOCO number above is case-to-case transfer WITHIN one centre and is")
            print("  therefore an OPTIMISTIC estimate of hidden-test (multi-centre) performance.")
            print("  RE-RUN THIS MODE THE DAY PART-2 LANDS.")
        else:
            mats = {}
            for cen in centres:
                sub = [r for r in rows if centre_of(r[0]) == cen]
                mats[cen], _ = fit_matrix(sub)
                print(f"  {cen}: {len(sub)} frames")
            print("\n  pairwise MEAN |ΔP(s|a)| between centre matrices "
                  "(off-diagonal only; large => prior does NOT transfer):")
            n = len(STATIONS)
            offdiag = ~np.eye(n, dtype=bool)
            worst = 0.0
            for i, c1 in enumerate(centres):
                for c2 in centres[i + 1:]:
                    d = np.abs(mats[c1] - mats[c2])[offdiag]
                    worst = max(worst, float(d.mean()))
                    print(f"    {c1} vs {c2}: mean {d.mean():.4f}  max {d.max():.4f}")
            print(f"\n  >>> worst mean off-diagonal drift = {worst:.4f}")
            print("      <0.05 = transfers well (keep alpha) | 0.05-0.15 = SHRINK HARDER "
                  "(raise alpha) | >0.15 = prior is centre-specific, prefer the blend or drop it")
            # leave-one-CENTRE-out (the honest multi-centre estimate)
            print("\n  === LEAVE-ONE-CENTRE-OUT (the number that predicts the hidden test centre) ===")
            stems, P, Y = [], [], []
            for held in centres:
                tr = [r for r in rows if centre_of(r[0]) != held]
                te = [r for r in rows if centre_of(r[0]) == held]
                M, pooled = fit_matrix(tr)
                st, p = predict(M, pooled, te)
                stems += st; P.append(p); Y.append(np.stack([r[2] for r in te]))
            results["locentreo_prior"] = score(
                stems, np.concatenate(Y), np.concatenate(P), "leave-one-CENTRE-out prior")

    if a.export_matrix:
        M, pooled = fit_matrix(rows)
        a.export_matrix.write_text(json.dumps(
            {"stations": STATIONS, "alpha": ALPHA, "lambda": LAMBDA,
             "n_frames": len(rows), "n_cases": len(cases),
             "pooled_prevalence": pooled.tolist(),
             "matrix": M.tolist(),
             "note": "M[a][s] = P(station s visible | annotated-frame station a), "
                     "additive-shrunk toward pooled marginal prevalence."},
            indent=2))
        print(f"\nexported shipping matrix -> {a.export_matrix}")
        print("  P(s|a) diagonal (annotated station visible in its own frame):")
        for i, s in enumerate(STATIONS):
            print(f"    {s:>4}: {M[i][i]:.3f}", end="" if (i + 1) % 5 else "\n")
        print()

    if a.out:
        a.out.write_text(json.dumps(
            {k: {"macro_f1": v["final_f1"], "auroc": v["final_auroc"]}
             for k, v in results.items()}, indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
