#!/usr/bin/env python3
"""
Tiger SQ-AI — test-time FALSE-POSITIVE suppression / per-class calibration.

The dominant w3 killer (diagnose_seg.py) is the one-empty-Dice=0 catastrophe:
a hallucinated FP blob of a w3 class on a frame where that class is ABSENT in GT
forces that frame's class-Dice to 0 (and nHD to 1.0). Because nnU-Net writes
ARGMAX integer-label masks (no per-pixel probabilities on disk for the banked
preds), the training-free lever here is a per-class CONNECTED-COMPONENT AREA
filter on the argmax mask:

  For class c on a frame, look at its predicted pixels:
    * (a) drop connected components whose own area < min_blob[c]   (kills speckle)
    * (b) if the class's TOTAL remaining area < min_total[c], drop the WHOLE class
          prediction on that frame                                  (kills absent-FP)
  Dropped pixels are relabelled to BACKGROUND (id 0). Background is present on
  every frame with huge area, so the added pixels are negligible to its Dice.

Empirically (val cases 14,15, 160px) for several w3 classes the absent-frame FP
blobs are much SMALLER than present-frame TP masks (e.g. t2 Left_Inf_Pulmonary_Lig
TP-area>=116 vs FP-area<=28; Right_Inf_Pulmonary_Lig TP>=61 vs FP-med 6), so a
total-area threshold cleanly separates them and recovers the whole class.

This module also supports an optional per-class MIN-CONFIDENCE threshold when a
softmax probability map IS available (the Docker path can `save_probabilities`):
drop the class on a frame if its max foreground probability < min_conf[c].

The same `apply_area_filter` is wired into both the val scorer/sweep here and
docker/inference.py for the submission.
"""
from __future__ import annotations
import numpy as np

try:
    from scipy import ndimage
    _HAVE_SCIPY = True
except Exception:                                        # pragma: no cover
    _HAVE_SCIPY = False


def _components(mask: np.ndarray):
    """Return (labelled, sizes_dict {comp_id: area}) for a binary mask."""
    if _HAVE_SCIPY:
        lbl, n = ndimage.label(mask)
        if n == 0:
            return lbl, {}
        sizes = ndimage.sum(np.ones_like(lbl, dtype=np.int64), lbl, range(1, n + 1))
        return lbl, {i + 1: int(sizes[i]) for i in range(n)}
    # fallback: treat the whole mask as one component (total-area filter still works)
    lbl = mask.astype(np.int32)
    return lbl, ({1: int(mask.sum())} if mask.any() else {})


def apply_area_filter(label_arr: np.ndarray,
                      min_blob: dict | None = None,
                      min_total: dict | None = None,
                      bg_id: int = 0) -> np.ndarray:
    """
    Per-class connected-component area filter on an ARGMAX integer-label mask.

    Args:
      label_arr : (H,W) int mask of class ids (the argmax prediction).
      min_blob  : {class_id: min connected-component area}  (drop smaller blobs).
      min_total : {class_id: min total class area after blob filter}
                  (drop the WHOLE class on this frame if below).
      bg_id     : id to relabel suppressed pixels to (default background 0).

    Returns a NEW filtered label array (does not mutate input).
    Only classes present in min_blob / min_total are touched; everything else
    is passed through unchanged.
    """
    min_blob = min_blob or {}
    min_total = min_total or {}
    out = label_arr.copy()
    touched = set(min_blob) | set(min_total)
    for cid in touched:
        if cid == bg_id:
            continue
        mask = (out == cid)
        if not mask.any():
            continue
        mb = min_blob.get(cid, 0)
        if mb and _HAVE_SCIPY:
            lbl, sizes = _components(mask)
            for comp_id, area in sizes.items():
                if area < mb:
                    out[lbl == comp_id] = bg_id
            mask = (out == cid)
        mt = min_total.get(cid, 0)
        if mt and int(mask.sum()) < mt:
            out[mask] = bg_id
    return out


def apply_conf_filter(label_arr: np.ndarray,
                      prob: np.ndarray,
                      min_conf: dict | None = None,
                      bg_id: int = 0) -> np.ndarray:
    """
    Optional per-class min-confidence gate (Docker path, when probabilities exist).
    `prob` is (C,H,W) softmax. If a class's max prob over its argmax region
    < min_conf[c], drop the class on this frame.  Returned mask is argmax-based.
    """
    min_conf = min_conf or {}
    out = label_arr.copy()
    for cid, thr in min_conf.items():
        if cid == bg_id or thr <= 0:
            continue
        mask = (out == cid)
        if not mask.any():
            continue
        if prob[cid][mask].max() < thr:
            out[mask] = bg_id
    return out
