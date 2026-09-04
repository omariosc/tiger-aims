#!/usr/bin/env python
"""Cross-dataset ARGMAX-LEVEL label graft (NOT prob-averaging).

The champion ensemble lives on Dataset202 (896x1536, 9-stage); the synth model on
Dataset203 (512x512, 8-stage). Their saved softmax npz are DIFFERENT shapes -> cannot
prob-average. Instead fuse at the argmax-mask level: keep the champion's prediction
everywhere, and ONLY graft the synth model's prediction for a small set of data-floor
rare classes (--graft-ids) at pixels where the champion predicts background (id 0).

Both pred dirs are <stem>.png integer-id masks. If shapes differ per stem, the synth
mask is nearest-neighbour resized to the champion's shape before grafting.

Usage:
  python fuse_argmax_graft.py --champ DIR_CHAMP --synth DIR_SYNTH \
      --graft-ids 8 --out OUTDIR
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def load_ids(p: Path) -> np.ndarray:
    a = np.array(Image.open(p))
    if a.ndim == 3:
        a = a[:, :, 0]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champ", required=True, type=Path, help="champion id-PNG dir (kept everywhere)")
    ap.add_argument("--synth", required=True, type=Path, help="synth id-PNG dir (donor for graft-ids)")
    ap.add_argument("--graft-ids", nargs="+", type=int, required=True,
                    help="class ids to graft from synth onto champion")
    ap.add_argument("--mode", choices=["fill", "replace"], default="fill",
                    help="fill (default): graft synth gid only where champ==0 (gap-fill). "
                         "replace: per-gid, first clear champ's own version of gid (champ==gid -> 0), "
                         "then write synth's gid everywhere synth==gid (the win-class region comes "
                         "entirely from synth; everything else stays champ).")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    champ_files = sorted(args.champ.glob("*.png"))
    assert champ_files, f"no PNGs in champ dir {args.champ}"
    graft = set(args.graft_ids)
    n_grafted_px = 0
    n_cleared_px = 0
    for cf in champ_files:
        sf = args.synth / cf.name
        assert sf.exists(), f"synth missing {sf.name}"
        champ = load_ids(cf)
        synth = load_ids(sf)
        if synth.shape != champ.shape:
            synth_img = Image.fromarray(synth.astype(np.uint8), mode="L").resize(
                (champ.shape[1], champ.shape[0]), Image.NEAREST)
            synth = np.array(synth_img)
        out = champ.copy()
        if args.mode == "fill":
            # graft: where champ==background(0) AND synth in graft-ids -> take synth label
            bg = (champ == 0)
            for gid in graft:
                sel = bg & (synth == gid)
                out[sel] = gid
                n_grafted_px += int(sel.sum())
        else:  # replace
            # per-gid: FIRST remove champ's own version of gid (champ==gid -> background 0),
            # THEN write synth's gid everywhere synth==gid. Net: the win-class region comes
            # entirely from synth; all other champ labels untouched.
            for gid in graft:
                clr = (out == gid)
                out[clr] = 0
                n_cleared_px += int(clr.sum())
                sel = (synth == gid)
                out[sel] = gid
                n_grafted_px += int(sel.sum())
        Image.fromarray(out.astype(np.uint8), mode="L").save(args.out / cf.name)
    print(f"[graft mode={args.mode}] {len(champ_files)} frames; ids {sorted(graft)}; "
          f"{n_grafted_px} synth px written, {n_cleared_px} champ px cleared -> {args.out}")


if __name__ == "__main__":
    main()
