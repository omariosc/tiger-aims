#!/usr/bin/env python3
"""
Apply tuned FP-suppression thresholds to a dir of nnU-Net argmax-id PNGs and
write the filtered masks to a NEW dir (banked preds are never overwritten).

Uses the per-class min_blob + min_total (absolute px, at the resolution the
thresholds were tuned) from a tune_fp_suppress.py JSON. Pred PNGs are full-res
mode-L id masks; thresholds in the *_fullres.json are full-res px so they apply
directly. (The frac fields support variable-size docker inputs.)

Usage:
  python apply_fp_suppress.py --cfg <..._fullres.json> --pred <in_dir> --out <out_dir>
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from fp_suppress import apply_area_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--use-frac", action="store_true",
                    help="scale thresholds by frame px (for variable-size inputs)")
    args = ap.parse_args()

    cfg = json.loads(args.cfg.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(args.pred.glob("*.png")):
        ids = np.array(Image.open(p))
        if ids.ndim == 3:
            ids = ids[:, :, 0]
        if args.use_frac:
            px = float(ids.shape[0] * ids.shape[1])
            mb = {int(k): int(round(v * px)) for k, v in cfg["min_blob_frac"].items()}
            mt = {int(k): int(round(v * px)) for k, v in cfg["min_total_frac"].items()}
        else:
            mb = {int(k): int(v) for k, v in cfg["min_blob"].items()}
            mt = {int(k): int(v) for k, v in cfg["min_total"].items()}
        out = apply_area_filter(ids, min_blob=mb, min_total=mt)
        Image.fromarray(out.astype(np.uint8), mode="L").save(args.out / p.name)
        n += 1
    print(f"wrote {n} filtered masks -> {args.out}")


if __name__ == "__main__":
    main()
