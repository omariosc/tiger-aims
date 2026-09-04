#!/usr/bin/env python3
"""
Convert the REAL TIGER SQ-AI segmentation data -> nnU-Net v2 raw format.

Source: /scratch/.../data/synapse/{images, masks_coarse, masks_fine}
Two nnU-Net datasets are produced (challenge task numbering, per README/memory):
  T1  = COARSE / merged   (16 classes, ids 0..15)  -> Dataset201_TigerT1coarse
  T2  = FINE   / full     (31 classes, ids 0..30)  -> Dataset202_TigerT2fine

We reuse the OFFICIAL challenge RGB->label_id converters unchanged so the
colour map is guaranteed identical to the scorer:
  metrics.classes_merged.rgb_mask_to_label_mask     (T1, 16-class)
  metrics.classes.rgb_mask_to_label_mask            (T2, 31-class)

nnU-Net v2 layout produced per dataset:
  $nnUNet_raw/Dataset<ID>_<name>/
    imagesTr/<case>_0000.png  (R)   imagesTr/<case>_0001.png (G)  _0002.png (B)
    labelsTr/<case>.png       (uint8 label ids)
    dataset.json              (lowercase 'background' key; NaturalImage2DIO)
    splits_final.json         (CASE-disjoint train/val, written into preprocessed later)

Case id = full frame stem, e.g. center_1_case_6_6R  (so nnU-Net keeps frames
addressable and a case-level split can hold out whole patients).

Sanity checks performed and printed:
  - every image has matching fine+coarse mask
  - RGB->id round-trip: count of unknown (255) pixels per frame (should be ~0)
  - class id histogram present across the dataset
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

CHALLENGE_REPO = Path("/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, str(CHALLENGE_REPO))

from metrics.classes import CLASSES, rgb_mask_to_label_mask as rgb_to_id_fine          # 31-class
from metrics.classes_merged import CLASSES_MERGED, rgb_mask_to_label_mask as rgb_to_id_coarse  # 16-class

DATA = Path("/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/synapse")


def case_of(stem: str) -> str:
    """center_1_case_6_6R -> center_1_case_6  (the patient/case for splitting)."""
    # drop the trailing _<station> token (station is 6L/6R/.../8/9)
    parts = stem.rsplit("_", 1)
    return parts[0]


def build_dataset(task: str, ds_id: int, ds_name: str, mask_dir: Path,
                  rgb_to_id, classes, raw_root: Path, val_cases: set[str]):
    ds_dir = raw_root / f"Dataset{ds_id:03d}_{ds_name}"
    img_tr = ds_dir / "imagesTr"; lbl_tr = ds_dir / "labelsTr"
    img_tr.mkdir(parents=True, exist_ok=True); lbl_tr.mkdir(parents=True, exist_ok=True)

    imgs = sorted((DATA / "images").glob("*.png"))
    id_hist = Counter(); unknown_total = 0; unknown_frames = 0
    n = 0; train_ids, val_ids = [], []
    for ip in imgs:
        stem = ip.stem
        mp = mask_dir / ip.name
        if not mp.exists():
            print(f"  WARN no mask for {ip.name}; skip"); continue
        rgb_img = np.array(Image.open(ip).convert("RGB"))
        rgb_msk = np.array(Image.open(mp).convert("RGB"))
        lab = rgb_to_id(rgb_msk)                       # (H,W) uint8 ids, 255=unknown
        unk = int((lab == 255).sum())
        if unk:
            unknown_total += unk; unknown_frames += 1
            # map unknown -> background (0) so nnU-Net never sees an out-of-range id
            lab = lab.copy(); lab[lab == 255] = 0
        id_hist.update(np.unique(lab).tolist())

        for ch in range(3):
            Image.fromarray(rgb_img[:, :, ch], mode="L").save(img_tr / f"{stem}_{ch:04d}.png")
        Image.fromarray(lab.astype(np.uint8), mode="L").save(lbl_tr / f"{stem}.png")

        (val_ids if case_of(stem) in val_cases else train_ids).append(stem)
        n += 1

    # dataset.json — lowercase 'background' (nnU-Net LabelManager requirement)
    labels = {}
    for c in classes:
        key = "background" if c.label_id == 0 else c.name
        labels[key] = c.label_id
    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": labels,
        "numTraining": n,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    print(f"[{task}] {ds_dir.name}: wrote {n} frames "
          f"({len(train_ids)} train / {len(val_ids)} val frames)")
    print(f"[{task}] class ids present: {sorted(id_hist)} "
          f"(of {len(classes)} defined; missing="
          f"{sorted(set(c.label_id for c in classes) - set(id_hist))})")
    print(f"[{task}] unknown-colour pixels: {unknown_total} across {unknown_frames} frames "
          f"(mapped->background); round-trip {'OK' if unknown_total==0 else 'has stray colours'}")
    return ds_dir, train_ids, val_ids


def write_split(preproc_root: Path, ds_id: int, ds_name: str, train_ids, val_ids):
    """nnU-Net reads splits_final.json from the PREPROCESSED dir. Write a single
    fold-0 with our case-disjoint split. (Must run after plan_and_preprocess
    creates the dir; we create the dir if absent.)"""
    d = preproc_root / f"Dataset{ds_id:03d}_{ds_name}"
    d.mkdir(parents=True, exist_ok=True)
    splits = [{"train": sorted(train_ids), "val": sorted(val_ids)}]
    (d / "splits_final.json").write_text(json.dumps(splits, indent=2))
    print(f"  wrote case-disjoint splits_final.json (fold 0) to {d.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path,
                    default=Path("/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/nnUNet_raw"))
    ap.add_argument("--preproc", type=Path,
                    default=Path("/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/nnUNet_preprocessed"))
    ap.add_argument("--val-cases", nargs="+",
                    default=["center_1_case_14", "center_1_case_15"],
                    help="cases held out for validation (case-disjoint)")
    args = ap.parse_args()

    val_cases = set(args.val_cases)
    all_cases = sorted({case_of(p.stem) for p in (DATA / "images").glob("*.png")})
    print(f"Cases ({len(all_cases)}): {all_cases}")
    print(f"VAL cases (held out): {sorted(val_cases)}")
    print(f"TRAIN cases: {sorted(set(all_cases) - val_cases)}")
    assert val_cases <= set(all_cases), f"val cases not in data: {val_cases - set(all_cases)}"

    # T1 = coarse (16), T2 = fine (31)
    ds1, tr1, va1 = build_dataset("T1-coarse", 201, "TigerT1coarse",
                                  DATA / "masks_coarse", rgb_to_id_coarse, CLASSES_MERGED,
                                  args.raw, val_cases)
    write_split(args.preproc, 201, "TigerT1coarse", tr1, va1)

    ds2, tr2, va2 = build_dataset("T2-fine", 202, "TigerT2fine",
                                  DATA / "masks_fine", rgb_to_id_fine, CLASSES,
                                  args.raw, val_cases)
    write_split(args.preproc, 202, "TigerT2fine", tr2, va2)

    print("DONE conversion.")


if __name__ == "__main__":
    main()
