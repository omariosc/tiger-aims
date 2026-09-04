#!/usr/bin/env python3
"""Does adopting the new T2 break T3?

T3 ships a seg-CONDITIONED head reading presence + log-percent-area of 18 boundary structures FROM
THE FINE MASKS, blended 50/50 with the station co-visibility prior. Adopting the retrain moves FINE
w3 from 0.3733 to 0.6986, so the head's INPUT DISTRIBUTION changes -- and the deployed head was
itself trained on OLD-model masks. T3 is a third of the combined rank, so this is measured, not
assumed. It can go either way: cleaner masks lifted T3 by +0.046 on 2026-06-25, but a head has never
seen features this good either.

*** EVERYTHING IS REUSED FROM train_t3_seg_conditioned.py RATHER THAN REIMPLEMENTED. A first draft
*** of this script guessed the feature builder and got three things wrong that would still have
*** produced plausible numbers: the wrong boundary-id set, log-area as a FRACTION where training
*** uses PERCENT (log1p(c/tot*100)), and no mu/sd normalisation at all -- although the checkpoint
*** stores feat_ids, stations, mu and sd precisely so callers need not guess.
"""
import sys, os, glob, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, "/users/USERNAME/Tiger-SQ-AI/experiments")
import train_t3_seg_conditioned as tm                       # noqa: E402
# Use the VALIDATED scorer, not the raw evaluate(): official evaluate_cls takes CSV PATHS
# (gt_csv, pred_csv), not arrays, and needs classes=CLASSES_STATIONS -- omitting that silently
# falls back to the SEGMENTATION registry and matches zero station columns (logged 2026-08-31).
from t3_station_prior import score as official_score        # noqa: E402

# The module's DATA constant points at the OLD 140-frame single-centre release. Our eval frames are
# centres 2/3/4/6/7 from the 40-case release, whose visibility rows live in synapse_v2 -- without
# this override load_labels() silently returns no matching stems.
V2 = Path("/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/synapse_v2")
if (V2 / "lymph_node_station_visibility.csv").exists():
    tm.DATA = V2
print("  visibility CSV:", tm.DATA / "lymph_node_station_visibility.csv")

CKPT = Path("/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t3/ckpt_segcond_ensmask/best.pt")
ck = torch.load(CKPT, map_location="cpu")
feat_ids, stations = ck["feat_ids"], ck["stations"]
mu = torch.as_tensor(ck["mu"]).float(); sd = torch.as_tensor(ck["sd"]).float()
head = tm.Head(len(feat_ids) * 2, len(stations))
head.load_state_dict(ck["head"]); head.eval()
print("  head %d->%d, feat_ids n=%d, mu/sd %s" % (len(feat_ids)*2, len(stations), len(feat_ids), tuple(mu.shape)))
assert list(feat_ids) == list(tm.FEAT_IDS), "checkpoint feat_ids disagree with the module's FEAT_IDS"

labels = tm.load_labels()
print("  visibility rows: %d" % len(labels))

def run(pred_dir, label):
    X, Y, stems = [], [], []
    for f in sorted(glob.glob(os.path.join(pred_dir, "*.png"))):
        stem = os.path.basename(f)[:-4]
        if stem not in labels:
            continue
        X.append(tm.seg_features(tm.load_pred_label(f)))
        Y.append(labels[stem]); stems.append(stem)
    if not X:
        print("  %-26s NO frames matched the station GT" % label); return None
    Xt = (torch.tensor(np.array(X), dtype=torch.float32) - mu) / sd
    with torch.no_grad():
        P = torch.sigmoid(head(Xt)).numpy()
    r = official_score(stems, np.array(Y), P, label)
    return {"n": len(stems), "f1": float(r["final_f1"]), "auroc": float(r["final_auroc"])}

E = "/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/adopt_ens_7705611"
FP = "/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t2follow_7710409/p_stalefp"
print("\n  === deployed seg-cond head: OLD masks vs NEW masks, identical frames ===")
out = {}
out["old"]    = run(os.path.join(E, "p_t2_inc"), "OLD masks (incumbent ens)")
out["new"]    = run(os.path.join(E, "p_t2_new"), "NEW masks (retrain ens)")
out["new_fp"] = run(FP,                          "NEW masks + FP (ship config)")
if out["old"] and out["new_fp"]:
    d = out["new_fp"]["f1"] - out["old"]["f1"]
    print("\n  ship-config minus incumbent: %+.4f macro-F1" % d)
    print("  >>> %s" % ("T3 is SAFE under the T2 adoption -- ships unchanged" if d >= -0.005 else
          "T3 DEGRADES -- retrain the seg-cond head on new-model masks before rebuilding"))
print("\n  NOTE: this is the seg-cond HALF only. Shipped T3 is a 50/50 blend with the co-visibility")
print("  prior, which is unaffected by the mask change, so the blended effect is roughly half this.")
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/t3_recheck.json", "w"), indent=1)
