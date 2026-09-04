#!/usr/bin/env python3
"""TIGER T3 — find a prior configuration that is not DOMINATED on either ranked metric.

Task 3 is ranked by macro-F1 AND AUROC. The 31 Aug head-to-head found the refit matrix gains
+0.090 macro-F1 but loses -0.102 AUROC against the shipped one, so neither dominates and the
"rebuild" question was left open. This tries four principled configurations on the SAME rows.

All arms are evaluated on the 298 frames from centres 2,3,4,6,7 - centres the SHIPPED matrix has
never seen, so arm A is honest there. Arms that refit exclude the held-out centre.

PRE-REGISTERED DECISION, fixed before any number below existed: choose the arm with the highest
MEAN of (macro-F1, AUROC), and report BOTH metrics for every arm so a dominated winner is visible.
Ties go to the SHIPPED arm, because changing a frozen artifact needs a positive reason.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, "/users/USERNAME/Tiger-SQ-AI/experiments")
from t3_station_prior import load_rows, fit_matrix, predict, score, STATIONS, SIDX, ALPHA

NEW = Path("/mnt/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/synapse_v2/lymph_node_station_visibility.csv")
SHIPPED = Path("/users/USERNAME/Tiger-SQ-AI/docker/covis_matrix.json")
HELD = ["center_2", "center_3", "center_4", "center_6", "center_7"]
def centre_of(c): return re.match(r"(center_\d+)", c).group(1)


def fit_toward(rows, target, alpha=ALPHA):
    """Same estimator, but each row shrinks toward the corresponding row of `target`."""
    n = len(STATIONS)
    cnt = np.zeros((n, n)); tot = np.zeros(n); ysum = np.zeros(n)
    for _c, ann, y in rows:
        a = SIDX[ann]; cnt[a] += y; tot[a] += 1; ysum += y
    pooled = ysum / max(1, len(rows))
    M = (cnt + alpha * target) / (tot[:, None] + alpha)
    return M, pooled


rows = load_rows(NEW)
sh = json.load(open(SHIPPED))
M_ship = np.array(sh["matrix"] if "matrix" in sh else sh["M"], float)
pooled_ship = np.array(sh.get("pooled", sh.get("marginal")), float)
target = [r for r in rows if centre_of(r[0]) in HELD]
stems_ref = [f"{c}_{a}" for c, a, _ in target]
Y = np.stack([r[2] for r in target])
print("all arms scored on the SAME %d frames from %s\n" % (len(target), HELD))

def loco_predict(fitter):
    """Fit excluding each held-out centre, predict it, return rows aligned to stems_ref."""
    out = {}
    for C in HELD:
        tr = [r for r in rows if centre_of(r[0]) != C]
        te = [r for r in rows if centre_of(r[0]) == C]
        M, pooled = fitter(tr)
        st, p = predict(M, pooled, te)
        out.update(dict(zip(st, p)))
    return np.stack([out[s] for s in stems_ref])

arms = {}
arms["A shipped (centre-1, frozen artifact)"] = predict(M_ship, pooled_ship, target)[1]
arms["B refit on 40 cases (pooled target)"]   = loco_predict(lambda tr: fit_matrix(tr))
arms["C mean of A and B"]                     = 0.5 * arms["A shipped (centre-1, frozen artifact)"] + 0.5 * arms["B refit on 40 cases (pooled target)"]
arms["D refit, shrunk toward SHIPPED"]        = loco_predict(lambda tr: fit_toward(tr, M_ship))

res = {}
for tag, P in arms.items():
    r = score(stems_ref, Y, P, tag)
    res[tag] = (r["final_f1"], r["final_auroc"])

print("\n  %-42s %9s %9s %9s" % ("arm", "macroF1", "AUROC", "mean"))
for tag, (f1, au) in res.items():
    print("  %-42s %9.5f %9.5f %9.5f" % (tag, f1, au, (f1 + au) / 2))
best = max(res.items(), key=lambda kv: (kv[1][0] + kv[1][1]) / 2)
print("\n  >>> PRE-REGISTERED WINNER (max mean of the two ranked metrics): %s" % best[0])
a_f1, a_au = res["A shipped (centre-1, frozen artifact)"]
b_f1, b_au = best[1]
print("      vs shipped: macro-F1 %+.5f   AUROC %+.5f" % (b_f1 - a_f1, b_au - a_au))
dominated = [t for t, (f, u) in res.items() if f <= a_f1 and u <= a_au and t != "A shipped (centre-1, frozen artifact)"]
if dominated: print("      arms DOMINATED by the shipped matrix (worse on both):", dominated)
