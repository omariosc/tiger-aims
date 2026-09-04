#!/usr/bin/env python3
"""TIGER T3 — SHIPPED centre-1 matrix vs a matrix refit on the new 40-case release,
scored on THE SAME ROWS: the 298 frames from centres 2,3,4,6,7.

Why a separate script: my first pass compared the shipped matrix (evaluated on centres 2-7) with a
leave-one-centre-out refit (evaluated on all six centres). Different row sets, so the difference was
not attributable to the matrix. This fixes that - both arms are scored on the identical 298 rows.

Arm A  SHIPPED  covis_matrix.json, fit on 10 cases from centre 1 only (what the container ships).
Arm B  REFIT    for each held-out centre C in {2,3,4,6,7}, fit on every case NOT in C (so centre 1
                plus the other new centres) and predict C. Centre C is never in its own training
                data, so this is honest for a hidden centre.

PRE-REGISTERED DECISION: rebuild the container's matrix only if B beats A on these rows.
"""
from __future__ import annotations
import json, sys, re
from pathlib import Path
import numpy as np
sys.path.insert(0, "/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, "/users/USERNAME/Tiger-SQ-AI/experiments")
from t3_station_prior import load_rows, fit_matrix, predict, score

NEW = Path("/mnt/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/synapse_v2/lymph_node_station_visibility.csv")
SHIPPED = Path("/users/USERNAME/Tiger-SQ-AI/docker/covis_matrix.json")
def centre_of(c): return re.match(r"(center_\d+)", c).group(1)

rows = load_rows(NEW)
held_centres = ["center_2", "center_3", "center_4", "center_6", "center_7"]
target = [r for r in rows if centre_of(r[0]) in held_centres]
print("evaluating both arms on the SAME %d frames from %s\n" % (len(target), held_centres))

sh = json.load(open(SHIPPED))
M_ship = np.array(sh["matrix"] if "matrix" in sh else sh["M"], float)
pooled_ship = np.array(sh.get("pooled", sh.get("marginal")), float)
stems_a, Pa = predict(M_ship, pooled_ship, target)
Ya = np.stack([r[2] for r in target])
ra = score(stems_a, Ya, Pa, "A SHIPPED (fit: 10 cases, centre 1 only)")

stems_b, Pb, Yb = [], [], []
for C in held_centres:
    tr = [r for r in rows if centre_of(r[0]) != C]
    te = [r for r in rows if centre_of(r[0]) == C]
    M, pooled = fit_matrix(tr)              # alpha=2.0, the swept optimum
    st, p = predict(M, pooled, te)
    stems_b += st; Pb.append(p); Yb.append(np.stack([r[2] for r in te]))
Pb = np.concatenate(Pb); Yb = np.concatenate(Yb)
assert stems_a == stems_b or sorted(stems_a) == sorted(stems_b), "row sets differ - not comparable"
# realign B to A's row order so the comparison is strictly paired
idx = {s: i for i, s in enumerate(stems_b)}
order = [idx[s] for s in stems_a]
Pb, Yb = Pb[order], Yb[order]
assert (Yb == Ya).all(), "ground truth mismatch after realignment"
rb = score(stems_a, Ya, Pb, "B REFIT   (fit: 40 cases, held-out centre excluded)")

d_f1 = rb["final_f1"] - ra["final_f1"]; d_au = rb["final_auroc"] - ra["final_auroc"]
print("\n  DELTA (B - A) on identical rows:  macro-F1 %+.5f   AUROC %+.5f" % (d_f1, d_au))
print("  DECISION: %s" % ("REBUILD the matrix from the 40-case release"
                          if d_f1 > 0 else "KEEP the shipped matrix"))
