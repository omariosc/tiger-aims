#!/usr/bin/env python3
"""Convert the 2026-08-31 TIGER release (40 cases / 524 frames / 6 centres) -> nnU-Net v2 raw.

`convert_tiger_to_nnunet.py` is left UNMODIFIED (it built the shipped 201/202 datasets). This is a
new file writing NEW dataset ids so the old ones stay intact:
    T1 coarse/merged (16 cls) -> Dataset211_TigerT1coarseV2
    T2 fine/full     (31 cls) -> Dataset212_TigerT2fineV2

★ FIXES A REAL LEAKAGE BUG IN THE OLD case_of().
  It derived the case by `stem.rsplit("_", 1)[0]`, i.e. "drop the trailing station token". The new
  release publishes two frames as EXTRA frames with a non-standard stem:
      center_3_case_3_na_station_1.png      center_7_case_3_na_station_1.png
  On those, rsplit yields `center_3_case_3_na_station` - a PHANTOM case. The real
  `center_3_case_3` could then appear in BOTH train and val, which is patient leakage.
  Here the case is matched explicitly as `^center_<n>_case_<m>`.

SPLIT: case-disjoint AND centre-stratified. The hidden test is multi-centre, so holding out whole
cases spread across centres is the honest protocol; a random frame split would leak patients and a
single-centre val would not measure what the test measures.

The OFFICIAL colour->id converters are reused unchanged, so the mapping matches the scorer.
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

CHALLENGE_REPO = Path("/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, str(CHALLENGE_REPO))
from metrics.classes import CLASSES, rgb_mask_to_label_mask as rgb_to_id_fine
from metrics.classes_merged import CLASSES_MERGED, rgb_mask_to_label_mask as rgb_to_id_coarse

CASE_RE = re.compile(r"^(center_\d+_case_\d+)")


def case_of(stem: str) -> str:
    m = CASE_RE.match(stem)
    if not m:
        raise ValueError("unparseable frame stem: %r" % stem)
    return m.group(1)


def centre_of(case: str) -> str:
    return case.split("_case_")[0]


def build(task, ds_id, ds_name, data_root: Path, mask_sub, rgb_to_id, classes,
          raw_root: Path, val_cases: set):
    ds = raw_root / ("Dataset%03d_%s" % (ds_id, ds_name))
    img_tr, lbl_tr = ds / "imagesTr", ds / "labelsTr"
    img_tr.mkdir(parents=True, exist_ok=True); lbl_tr.mkdir(parents=True, exist_ok=True)
    imgs = sorted((data_root / "images").glob("*.png"))
    hist = Counter(); unk_total = unk_frames = 0; n = 0
    train_ids, val_ids = [], []
    for ip in imgs:
        stem = ip.stem
        mp = data_root / mask_sub / ip.name
        if not mp.exists():
            print("  WARN no %s mask for %s; skip" % (mask_sub, ip.name)); continue
        rgb_i = np.array(Image.open(ip).convert("RGB"))
        lab = rgb_to_id(np.array(Image.open(mp).convert("RGB")))
        u = int((lab == 255).sum())
        if u:
            unk_total += u; unk_frames += 1
            lab = lab.copy(); lab[lab == 255] = 0
        for c in range(3):
            Image.fromarray(rgb_i[:, :, c]).save(img_tr / ("%s_%04d.png" % (stem, c)))
        Image.fromarray(lab.astype(np.uint8)).save(lbl_tr / ("%s.png" % stem))
        hist.update(np.unique(lab).tolist()); n += 1
        (val_ids if case_of(stem) in val_cases else train_ids).append(stem)
    # NOTE: the attribute is `label_id`, NOT `id`. An earlier version used
    # getattr(c, "id", None), which returned None for every class and silently wrote a
    # dataset.json declaring ONLY background - nnU-Net then rejected every mask with
    # "Unexpected labels found ... Expected: [0]". Same rule as the original converter.
    labels = {}
    for c in classes:
        labels["background" if c.label_id == 0 else c.name] = int(c.label_id)
    json.dump({"channel_names": {"0": "R", "1": "G", "2": "B"},
               "labels": labels, "numTraining": n, "file_ending": ".png",
               "overwrite_image_reader_writer": "NaturalImage2DIO"},
              open(ds / "dataset.json", "w"), indent=2)
    print("  %-10s %-26s frames=%-4d train=%-4d val=%-3d  ids_seen=%d  unknown_px=%d in %d frames"
          % (task, ds.name, n, len(train_ids), len(val_ids), len(hist), unk_total, unk_frames))
    return train_ids, val_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path("/mnt/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/synapse_v2"))
    ap.add_argument("--raw", type=Path,
                    default=Path("/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/nnUNet_raw"))
    ap.add_argument("--preproc", type=Path,
                    default=Path("/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/nnUNet_preprocessed"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    stems = [p.stem for p in sorted((a.data / "images").glob("*.png"))]
    cases = sorted({case_of(s) for s in stems})
    by_centre = defaultdict(list)
    for c in cases:
        by_centre[centre_of(c)].append(c)
    rng = np.random.default_rng(a.seed)
    val = set()
    for cen in sorted(by_centre):
        cs = sorted(by_centre[cen])
        k = max(1, int(round(len(cs) * a.val_frac)))
        val.update(rng.permutation(cs)[:k].tolist())
    print("frames=%d cases=%d centres=%d" % (len(stems), len(cases), len(by_centre)))
    for cen in sorted(by_centre):
        held = sorted(set(by_centre[cen]) & val)
        print("  %-10s %2d cases, val: %s" % (cen, len(by_centre[cen]), held))
    print("VAL cases (%d):" % len(val), sorted(val))

    tr1, va1 = build("T1-coarse", 211, "TigerT1coarseV2", a.data, "masks_coarse",
                     rgb_to_id_coarse, CLASSES_MERGED, a.raw, val)
    tr2, va2 = build("T2-fine", 212, "TigerT2fineV2", a.data, "masks_fine",
                     rgb_to_id_fine, CLASSES, a.raw, val)
    for ds_id, name, tr, va in ((211, "TigerT1coarseV2", tr1, va1), (212, "TigerT2fineV2", tr2, va2)):
        d = a.preproc / ("Dataset%03d_%s" % (ds_id, name)); d.mkdir(parents=True, exist_ok=True)
        json.dump([{"train": tr, "val": va}], open(d / "splits_final.json", "w"), indent=2)
    # leakage assertion: no case may appear on both sides
    for tr, va in ((tr1, va1), (tr2, va2)):
        assert not ({case_of(s) for s in tr} & {case_of(s) for s in va}), "CASE LEAK across split"
    print("OK: splits are case-disjoint and centre-stratified")


if __name__ == "__main__":
    main()
