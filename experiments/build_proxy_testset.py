#!/usr/bin/env python3
"""
Build a held-out test set (imagesTs / labelsTs) of CholecSeg8k frames that were
NOT used in the 30-frames-per-video training subsample. These are genuinely
unseen frames -> honest proxy for prediction + scoring.

imagesTs: <case>_0000/0001/0002.png  (R/G/B channels, nnU-Net predict input)
labelsTs: <case>.png                 (uint8 label ids, GT for the scorer)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cholecseg8k_classmap import watershed_to_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="dataset dir (has imagesTr)")
    ap.add_argument("--train-frames-per-video", type=int, default=30)
    ap.add_argument("--test-per-video", type=int, default=4)
    ap.add_argument("--videos", nargs="*", default=["video01", "video25", "video55"],
                    help="which videos to draw the held-out test frames from")
    args = ap.parse_args()

    img_ts = args.out / "imagesTs"
    lbl_ts = args.out / "labelsTs"
    img_ts.mkdir(parents=True, exist_ok=True)
    lbl_ts.mkdir(parents=True, exist_ok=True)

    n = 0
    for vname in args.videos:
        vid = args.src / vname
        frame_imgs = sorted(vid.glob("*/*_endo.png"))
        # frames used in training (linspace subsample)
        if len(frame_imgs) > args.train_frames_per_video:
            train_idx = set(np.linspace(0, len(frame_imgs) - 1,
                                        args.train_frames_per_video).astype(int).tolist())
        else:
            train_idx = set(range(len(frame_imgs)))
        held = [i for i in range(len(frame_imgs)) if i not in train_idx]
        if not held:
            continue
        pick = np.linspace(0, len(held) - 1, min(args.test_per_video, len(held))).astype(int)
        for k in pick:
            i = held[int(k)]
            fimg = frame_imgs[i]
            stem = fimg.name.replace("_endo.png", "")
            wmask = fimg.with_name(fimg.name.replace("_endo.png", "_endo_watershed_mask.png"))
            if not wmask.exists():
                continue
            case = f"{vname}_{stem}"
            rgb = np.array(Image.open(fimg).convert("RGB"))
            lab = watershed_to_label(np.array(Image.open(wmask).convert("L")))
            for ch in range(3):
                Image.fromarray(rgb[:, :, ch], mode="L").save(img_ts / f"{case}_{ch:04d}.png")
            Image.fromarray(lab.astype(np.uint8), mode="L").save(lbl_ts / f"{case}.png")
            n += 1
    print(f"Wrote {n} held-out test cases to {img_ts} / {lbl_ts}")


if __name__ == "__main__":
    main()
