#!/usr/bin/env python
"""
T3 per-station decision-threshold calibration (CPU-only, no GPU).

Diagnosis (Tiger CLAUDE.md sec 0 RESIDUAL): several seg-cond T3 stations have
AUROC >> F1@0.5 (11L AUROC 0.944 / F1 0.333; 13L/10R/13R/11R) => the *ranking*
is right but the fixed 0.5 cut is mis-calibrated. Per-station thresholds can
recover ~+0.03-0.06 weighted-F1 at zero GPU cost.

RIGOR: only 2 val cases (28 frames) exist, and NO train-set seg-cond probs are
on disk. So we CANNOT honestly tune per-station thresholds on a separate train
split. Two estimates are reported:
  (A) ORACLE / test-fit: F1-optimal per-station threshold chosen ON the full
      28-frame val set (this is upper-bound, OPTIMISTIC, test-fitting -> NOT a
      claimable gain; reported only as a ceiling).
  (B) LEAVE-ONE-CASE-OUT (LOCO): fit per-station thresholds on one case, apply
      to the held-out case, pool the held-out predictions, then score. This is
      the honest estimate but with n=2 cases it is extremely high-variance.

Scoring uses the OFFICIAL metrics.f1_score primitive and the official station
registry (14 stations, all weight=1 => weighted-F1 == macro-F1). We replicate
evaluate_cls's weighting exactly and assert it reproduces the official
evaluate() at the default 0.5 threshold before reporting any calibrated number.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, str(REPO))
from metrics.metrics import f1_score                      # official F1 primitive
from metrics.classes_stations import CLASSES_STATIONS     # official registry
from metrics.evaluate_cls import evaluate as official_evaluate

CKPT = Path("/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t3/ckpt_segcond_ftpred")
GT_CSV   = CKPT / "val_gt.csv"
PRED_CSV = CKPT / "val_pred.csv"

STATIONS = [c.name for c in CLASSES_STATIONS]
WEIGHTS  = np.array([c.weight for c in CLASSES_STATIONS], dtype=float)

# Candidate thresholds to search (probabilities). Include 0.5 so default is reachable.
GRID = np.round(np.arange(0.05, 0.951, 0.01), 4)


def load():
    gt   = pd.read_csv(GT_CSV,   index_col="case_id").sort_index()
    pred = pd.read_csv(PRED_CSV, index_col="case_id").sort_index()
    assert list(gt.index) == list(pred.index), "case_id mismatch"
    gt_mat   = gt[STATIONS].values.astype(int)
    pred_mat = pred[STATIONS].values.astype(float)
    # LOCO grouping key: center_1_case_<N>  (id = center_1_case_<N>_<station>)
    cases = ["_".join(c.split("_")[:4]) for c in gt.index]
    return gt_mat, pred_mat, np.array(cases), list(gt.index)


def weighted_f1(gt_mat, prob_mat, thr_vec):
    """Replicate evaluate_cls weighting exactly with PER-station thresholds."""
    w_f1 = 0.0
    per = {}
    for i, name in enumerate(STATIONS):
        yt = gt_mat[:, i]
        yp = (prob_mat[:, i] >= thr_vec[i]).astype(int)
        f1 = f1_score(yt, yp)
        w_f1 += f1 * WEIGHTS[i]
        per[name] = f1
    return w_f1 / WEIGHTS.sum(), per


def best_thr_per_station(gt_mat, prob_mat):
    """F1-optimal threshold per station on the GIVEN (fit) split. Ties -> closest to 0.5."""
    thr = np.full(len(STATIONS), 0.5)
    for i in range(len(STATIONS)):
        yt = gt_mat[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            thr[i] = 0.5  # degenerate class on this split -> keep default
            continue
        best_f1, best_t = -1.0, 0.5
        for t in GRID:
            yp = (prob_mat[:, i] >= t).astype(int)
            f1 = f1_score(yt, yp)
            if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
                best_f1, best_t = f1, t
        thr[i] = best_t
    return thr


def main():
    gt_mat, prob_mat, cases, ids = load()
    uniq = sorted(set(cases))
    n_frames = len(ids)

    # --- 0. sanity: reproduce official evaluate() at 0.5 ---
    off = official_evaluate(GT_CSV, PRED_CSV, threshold=0.5, classes=CLASSES_STATIONS)
    thr05 = np.full(len(STATIONS), 0.5)
    my_f1_05, per05 = weighted_f1(gt_mat, prob_mat, thr05)
    assert abs(my_f1_05 - off["final_f1"]) < 1e-9, \
        f"replication mismatch: mine {my_f1_05} vs official {off['final_f1']}"
    print(f"[sanity] official wF1@0.5 = {off['final_f1']:.4f}  AUROC = {off['final_auroc']:.4f}")
    print(f"[sanity] my replicated  wF1@0.5 = {my_f1_05:.4f}  (match OK)")
    print(f"[info] cases = {uniq}, frames = {n_frames}")

    # --- A. ORACLE / test-fit (upper bound, optimistic, NOT claimable) ---
    thr_oracle = best_thr_per_station(gt_mat, prob_mat)
    f1_oracle, per_oracle = weighted_f1(gt_mat, prob_mat, thr_oracle)

    # --- B. LEAVE-ONE-CASE-OUT (honest estimate, n=2) ---
    loco_prob_bin = np.zeros_like(gt_mat)
    loco_thr_used = {c: None for c in uniq}
    for held in uniq:
        fit_mask  = cases != held
        test_mask = cases == held
        thr = best_thr_per_station(gt_mat[fit_mask], prob_mat[fit_mask])
        loco_thr_used[held] = thr.tolist()
        loco_prob_bin[test_mask] = (prob_mat[test_mask] >= thr).astype(int)
    # score pooled LOCO binary predictions with official F1 weighting
    w_f1_loco = 0.0
    per_loco = {}
    for i, name in enumerate(STATIONS):
        f1 = f1_score(gt_mat[:, i], loco_prob_bin[:, i])
        w_f1_loco += f1 * WEIGHTS[i]
        per_loco[name] = f1
    f1_loco = w_f1_loco / WEIGHTS.sum()

    # mean LOCO threshold per station (for reporting a single deployable vector)
    loco_thr_mat = np.array([loco_thr_used[c] for c in uniq])
    loco_thr_mean = loco_thr_mat.mean(axis=0)

    # --- report ---
    print("\n=== per-station: thr(default 0.5) / F1@0.5 | oracle thr / F1 | LOCO F1 | AUROC ===")
    aurocs = {n: off["class_scores"][n]["auroc"] for n in STATIONS}
    for i, n in enumerate(STATIONS):
        print(f"  {n:>4}: 0.50/F1={per05[n]:.3f} | "
              f"orc thr={thr_oracle[i]:.2f}/F1={per_oracle[n]:.3f} | "
              f"LOCO F1={per_loco[n]:.3f} | AUROC={aurocs[n]:.3f}")

    print("\n=== HEADLINE weighted-F1 (== macro-F1, all w=1) ===")
    print(f"  default 0.5         : {my_f1_05:.4f}")
    print(f"  LOCO (honest, n=2)  : {f1_loco:.4f}   (Δ {f1_loco - my_f1_05:+.4f})")
    print(f"  ORACLE/test-fit (UB): {f1_oracle:.4f}   (Δ {f1_oracle - my_f1_05:+.4f})  [NOT claimable]")
    print(f"  AUROC (unchanged)   : {off['final_auroc']:.4f}")

    improved_loco = [n for n in STATIONS if per_loco[n] > per05[n] + 1e-9]
    worsened_loco = [n for n in STATIONS if per_loco[n] < per05[n] - 1e-9]
    print(f"\n  LOCO improved stations : {improved_loco}")
    print(f"  LOCO worsened stations : {worsened_loco}")

    out = {
        "default_thr": thr05.tolist(),
        "wf1_default": my_f1_05,
        "auroc": off["final_auroc"],
        "oracle_thr_per_station": dict(zip(STATIONS, thr_oracle.tolist())),
        "wf1_oracle_testfit_UB": f1_oracle,
        "loco_thr_per_case": loco_thr_used,
        "loco_thr_mean_per_station": dict(zip(STATIONS, loco_thr_mean.tolist())),
        "wf1_loco_honest": f1_loco,
        "per_station_f1": {n: {"f1_0.5": per05[n], "f1_oracle": per_oracle[n],
                               "f1_loco": per_loco[n], "auroc": aurocs[n]} for n in STATIONS},
        "loco_improved": improved_loco,
        "loco_worsened": worsened_loco,
        "n_cases": len(uniq), "n_frames": n_frames,
    }
    outp = CKPT / "threshold_calibration.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n[written] {outp}")


if __name__ == "__main__":
    main()
