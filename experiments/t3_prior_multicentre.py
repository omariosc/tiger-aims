#!/usr/bin/env python3
"""TIGER T3 — does the SHIPPED co-visibility prior survive contact with new centres?

`t3_station_prior.py` is a VALIDATED INSTRUMENT and is NOT modified. This is a new file.

CONTEXT. The shipped matrix (`covis_matrix.json`, 2026-07-27) was fit on 10 cases that all came
from CENTRE 1. The 2026-08-31 release adds 30 cases across centres 2, 3, 4, 6 and 7. The
pre-registered drift rule in the older script has now FIRED: worst mean off-diagonal drift between
centre matrices is 0.3737, and the rule says ">0.15 = prior is centre-specific, prefer the blend or
drop it".

PRE-REGISTERED QUESTIONS, fixed before any number below was computed:
  Q1  Does the SHIPPED (centre-1) matrix beat the pooled marginal prevalence on centres it has
      never seen (2,3,4,6,7)?  If NOT, the shipped prior is centre-specific and the blend weight
      should come down.
  Q2  Does a matrix re-fit on all 40 cases beat the shipped one on held-out CENTRES?
  Q3  Does raising the shrinkage alpha help, as the drift rule predicts it should?
DECISION RULE, also fixed in advance: ship the configuration with the best leave-one-CENTRE-out
macro-F1, since the hidden test is multi-centre; ties go to the LOWER-variance (higher alpha) option.
"""
from __future__ import annotations
import json, sys, re
from pathlib import Path
import numpy as np

REPO = Path("/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/users/USERNAME/Tiger-SQ-AI/experiments")
from metrics.classes_stations import STATION_NAMES
from t3_station_prior import load_rows, fit_matrix, predict, write_csv, STATIONS

NEW = Path("/mnt/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/synapse_v2/lymph_node_station_visibility.csv")
SHIPPED = Path("/users/USERNAME/Tiger-SQ-AI/docker/covis_matrix.json")


def centre_of(case): return re.match(r"(center_\d+)", case).group(1)


# Import the VALIDATED scorer rather than reimplementing it. My first attempt rewrote it and
# dropped `classes=CLASSES_STATIONS`, so evaluate_cls fell back to the SEGMENTATION registry and
# matched zero station columns. Importing removes that whole class of error.
from t3_station_prior import score as _score


def score(stems, Y, P, tag):
    r = _score(stems, Y, P, tag)
    return r["final_f1"], r["final_auroc"]


rows = load_rows(NEW)
cases = sorted({c for c, _, _ in rows})
centres = sorted({centre_of(c) for c in cases})
print("loaded %d frames / %d cases / %d centres %s\n" % (len(rows), len(cases), len(centres), centres))

# ---- Q1: the SHIPPED centre-1 matrix, applied to centres it has never seen -------------------
sh = json.load(open(SHIPPED))
M_ship = np.array(sh["matrix"] if "matrix" in sh else sh["M"], float)
pooled_ship = np.array(sh.get("pooled", sh.get("marginal")), float)
unseen = [r for r in rows if centre_of(r[0]) != "center_1"]
stems, P = predict(M_ship, pooled_ship, unseen)
Y = np.stack([r[2] for r in unseen])
print("=== Q1: SHIPPED centre-1 matrix on the %d frames from centres 2,3,4,6,7 ===" % len(unseen))
f1_ship, _ = score(stems, Y, P, "SHIPPED matrix (fit on centre 1 only)")
# baseline: pooled prevalence of the UNSEEN centres' own training data is not available at test
# time, so the honest baseline is the shipped pooled marginal, broadcast to every row
P_base = np.tile(pooled_ship, (len(unseen), 1))
f1_base, _ = score(stems, Y, P_base, "baseline: shipped marginal prevalence (no co-visibility)")
print("  --> co-visibility structure is worth %+.5f macro-F1 on unseen centres\n" % (f1_ship - f1_base))

# ---- Q2/Q3: leave-one-CENTRE-out, refit on 40 cases, alpha swept ------------------------------
print("=== Q2/Q3: leave-one-CENTRE-out on the new 40-case release, alpha swept ===")
best = None
for alpha in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
    stems, P, Y = [], [], []
    for held in centres:
        tr = [r for r in rows if centre_of(r[0]) != held]
        te = [r for r in rows if centre_of(r[0]) == held]
        M, pooled = fit_matrix(tr, alpha=alpha)
        st, p = predict(M, pooled, te)
        stems += st; P.append(p); Y.append(np.stack([r[2] for r in te]))
    P = np.concatenate(P); Y = np.concatenate(Y)
    f1, au = score(stems, Y, P, "LOCO-centre refit on 40 cases, alpha=%-5g" % alpha)
    if best is None or f1 > best[1]: best = (alpha, f1, au)
print("\n  >>> best alpha = %g  (macro-F1 %.5f, AUROC %.5f)" % best)
print("  >>> shipped alpha = 2.0")
