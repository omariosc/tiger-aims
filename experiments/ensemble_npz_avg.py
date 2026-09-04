#!/usr/bin/env python
"""Softmax-prob-average an arbitrary set of nnUNetv2 prediction dirs (each holding
<stem>.npz with key 'probabilities', shape (C,1,H,W)) -> argmax -> uint8 id PNG.

Replicates run_nnunet_ensemble() from tiger_sandbox/opt/tiger/inference.py:
  acc = sum(prob over members) / n ; ids = acc.argmax(0).squeeze().astype(uint8)

All members MUST share the same spatial shape (same Dataset / plans) -> use this
ONLY within a single dataset (e.g. the 3 synth seeds on Dataset203). Cross-dataset
fusion (Dataset202 vs 203) is shape-incompatible -> use fuse_argmax_graft.py instead.

Usage:
  python ensemble_npz_avg.py --members DIR1 DIR2 DIR3 --out OUTDIR
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, type=Path,
                    help="prediction dirs each containing <stem>.npz (probabilities)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    member_dirs = [Path(m) for m in args.members]
    for md in member_dirs:
        npzs = list(md.glob("*.npz"))
        assert npzs, f"member dir has NO .npz files: {md}"
    args.out.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in member_dirs[0].glob("*.npz"))
    assert stems, f"no .npz in first member {member_dirs[0]}"
    print(f"[ens] {len(member_dirs)} members x {len(stems)} stems -> {args.out}")
    for stem in stems:
        acc = None
        for md in member_dirs:
            f = md / f"{stem}.npz"
            assert f.exists(), f"MISSING npz {f}"
            prob = np.load(f)["probabilities"].astype(np.float32)
            acc = prob if acc is None else acc + prob
        acc /= len(member_dirs)
        ids = acc.argmax(axis=0).squeeze().astype(np.uint8)  # (C,1,H,W)->(H,W)
        Image.fromarray(ids, mode="L").save(args.out / f"{stem}.png")
    print(f"[ens] wrote {len(stems)} id masks -> {args.out}")


if __name__ == "__main__":
    main()
