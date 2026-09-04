#!/usr/bin/env python3
"""
Convert CholecSeg8k -> nnU-Net v2 raw format (2D RGB semantic segmentation).

This mirrors the pipeline TIGER SQ-AI will need:
  - read RGB frame  -> 3-channel nnU-Net input (_0000/_0001/_0002 grayscale PNGs)
  - read mask       -> single-channel integer label PNG (ids 0..K-1, contiguous)

For CholecSeg8k the label source is the '*_endo_watershed_mask.png' grayscale
code mask (see cholecseg8k_classmap.py). For the REAL challenge the analogous
step is RGB-COLOUR -> label_id via metrics/classes.py:rgb_mask_to_label_mask.

nnU-Net v2 layout produced:
  $nnUNet_raw/Dataset<ID>_<name>/
    imagesTr/<case>_0000.png  (R)
    imagesTr/<case>_0001.png  (G)
    imagesTr/<case>_0002.png  (B)
    labelsTr/<case>.png       (uint8 label ids)
    dataset.json

Case ids encode the source video so nnU-Net's later splits can respect it,
but here we let nnU-Net auto-CV (the proxy goal is a working checkpoint, not
patient-disjoint rigour — that lever applies to the real 50-case data).

Usage:
  python convert_cholecseg8k_to_nnunet.py \
      --src   /scratch/.../data/proxy/extracted/CholecSeg8k \
      --raw   /scratch/.../nnunet/nnUNet_raw \
      --dataset-id 101 --dataset-name CholecSeg8k \
      --max-frames-per-video 60      # subsample for a FAST proxy run
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cholecseg8k_classmap import watershed_to_label, ID_TO_NAME, NUM_CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--dataset-id", type=int, default=101)
    ap.add_argument("--dataset-name", default="CholecSeg8k")
    ap.add_argument("--max-frames-per-video", type=int, default=0,
                    help="0 = use all frames; else subsample evenly per video")
    ap.add_argument("--max-videos", type=int, default=0,
                    help="0 = all videos; else cap number of videos")
    args = ap.parse_args()

    ds_dir = args.raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    img_tr = ds_dir / "imagesTr"
    lbl_tr = ds_dir / "labelsTr"
    img_tr.mkdir(parents=True, exist_ok=True)
    lbl_tr.mkdir(parents=True, exist_ok=True)

    videos = sorted(d for d in args.src.iterdir() if d.is_dir() and d.name.startswith("video"))
    if args.max_videos:
        videos = videos[: args.max_videos]

    n_written = 0
    seen_ids = set()
    for vid in videos:
        # each video/ has frame-folders; gather all frames
        frame_imgs = sorted(vid.glob("*/*_endo.png"))
        if args.max_frames_per_video and len(frame_imgs) > args.max_frames_per_video:
            idx = np.linspace(0, len(frame_imgs) - 1, args.max_frames_per_video).astype(int)
            frame_imgs = [frame_imgs[i] for i in idx]

        for fimg in frame_imgs:
            stem = fimg.name.replace("_endo.png", "")          # frame_2095
            wmask = fimg.with_name(fimg.name.replace("_endo.png", "_endo_watershed_mask.png"))
            if not wmask.exists():
                continue
            case = f"{vid.name}_{stem}"                          # video26_frame_2095

            rgb = np.array(Image.open(fimg).convert("RGB"))      # (H,W,3)
            gray = np.array(Image.open(wmask).convert("L"))      # (H,W) codes
            lab = watershed_to_label(gray)                       # (H,W) ids 0..12
            seen_ids.update(np.unique(lab).tolist())

            # 3 channel PNGs (nnU-Net reads each modality separately)
            for ch in range(3):
                Image.fromarray(rgb[:, :, ch], mode="L").save(
                    img_tr / f"{case}_{ch:04d}.png")
            Image.fromarray(lab.astype(np.uint8), mode="L").save(
                lbl_tr / f"{case}.png")
            n_written += 1

    labels = {name: lid for lid, name in ID_TO_NAME.items()}
    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": labels,
        "numTraining": n_written,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    print(f"Wrote {n_written} cases to {ds_dir}")
    print(f"Label ids present in data: {sorted(seen_ids)}")
    print(f"dataset.json labels ({NUM_CLASSES}): {labels}")


if __name__ == "__main__":
    main()
