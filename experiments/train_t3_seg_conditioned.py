#!/usr/bin/env python3
"""
TIGER SQ-AI Task 3 — SEG-CONDITIONED LN-station visibility head (lever P2.5).

Protocol rule: a station is "in frame" IFF its boundary structures are visible.
=> predict the 14-station multi-hot from the T2 (31-class FINE) segmentation,
not from raw appearance. Per-frame seg features:
  - presence (0/1) + log-area-fraction of EACH of the 18 boundary-relevant fine
    classes used across the station map (+ shared Lymph_Node) -> compact, dense.
A small MLP maps these features -> 14 station logits (BCE + pos_weight).

Variants (--mode):
  seg     : features from GT fine masks (masks_fine)  [upper-bound of the idea]
  hybrid  : seg features  +  frozen DINOv3 image features  [seg + appearance]
  image   : DINOv3 only (== the 5991701 baseline; for an in-script control)
For val, --val-pred-dir <nnUNet T2 predTs> uses PREDICTED masks for the val frames
(realistic deployment) while train still uses GT masks; falls back to GT if absent.

Case-disjoint split (same held-out cases as seg). Scored with the OFFICIAL
metrics.evaluate_cls.evaluate (weighted-F1@0.5 + AUROC). Outputs to --outdir:
  metrics.json, val_pred.csv, val_gt.csv, best.pt
"""
from __future__ import annotations
import argparse, json, sys, csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from PIL import Image

CHALLENGE_REPO = Path("/users/USERNAME/Tiger-SQ-AI/challenge_repo")
sys.path.insert(0, str(CHALLENGE_REPO))
sys.path.insert(0, str(Path(__file__).parent))
from metrics.classes_stations import CLASSES_STATIONS, STATION_NAMES
from metrics.classes import rgb_mask_to_label_mask as rgb_to_id_fine
from station_boundary_map import STATION_BOUNDARY_FINE, SHARED_FINE

DATA = Path("/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/synapse")
STATIONS = STATION_NAMES
S2I = {s: i for i, s in enumerate(STATIONS)}

# Build the ordered set of fine-class ids used as features (union over stations + shared).
FEAT_IDS = sorted(set([i for v in STATION_BOUNDARY_FINE.values() for i in v] + SHARED_FINE))
FID2COL = {fid: j for j, fid in enumerate(FEAT_IDS)}


def case_of(stem):
    return stem.rsplit("_", 1)[0]


def load_labels():
    df = pd.read_csv(DATA / "lymph_node_station_visibility.csv")
    df.columns = [c.strip() for c in df.columns]
    frame_col = [c for c in df.columns if "annotated" in c.lower()][0]
    vis_col = [c for c in df.columns if "visible" in c.lower()][0]
    out = {}
    for _, row in df.iterrows():
        stem = f"{str(row['case']).strip()}_{str(row[frame_col]).strip()}"
        vec = np.zeros(len(STATIONS), dtype=np.float32)
        for tok in str(row[vis_col]).split(","):
            tok = tok.strip()
            if tok in S2I:
                vec[S2I[tok]] = 1.0
        out[stem] = vec
    return out


def seg_features(label_arr):
    """presence + log-area-fraction for each FEAT_ID fine class -> 2*len(FEAT_IDS)."""
    h, w = label_arr.shape
    tot = float(h * w)
    pres = np.zeros(len(FEAT_IDS), dtype=np.float32)
    area = np.zeros(len(FEAT_IDS), dtype=np.float32)
    ids, counts = np.unique(label_arr, return_counts=True)
    for i, c in zip(ids, counts):
        if int(i) in FID2COL:
            j = FID2COL[int(i)]
            pres[j] = 1.0
            area[j] = np.log1p(c / tot * 100.0)  # log of percent-area
    return np.concatenate([pres, area])


def load_fine_label(path, downscale=384):
    rgb = np.array(Image.open(path).convert("RGB"))
    lab = rgb_to_id_fine(rgb)
    lab[lab == 255] = 0
    if downscale and max(lab.shape) > downscale:
        h, w = lab.shape
        s = downscale / max(h, w)
        lab = np.array(Image.fromarray(lab.astype(np.uint8)).resize(
            (max(1, int(w * s)), max(1, int(h * s))), Image.NEAREST))
    return lab.astype(np.uint8)


def load_pred_label(path, downscale=384):
    """nnU-Net pred PNG = integer-id in channel 0 (already label ids)."""
    arr = np.array(Image.open(path).convert("RGB"))[:, :, 0].astype(np.uint8)
    if downscale and max(arr.shape) > downscale:
        h, w = arr.shape
        s = downscale / max(h, w)
        arr = np.array(Image.fromarray(arr).resize(
            (max(1, int(w * s)), max(1, int(h * s))), Image.NEAREST))
    return arr


def build_seg_feats(stems, val_set, val_pred_dir):
    feats = {}
    for stem in stems:
        use_pred = (val_pred_dir is not None and stem in val_set
                    and (val_pred_dir / f"{stem}.png").exists())
        if use_pred:
            lab = load_pred_label(val_pred_dir / f"{stem}.png")
        else:
            lab = load_fine_label(DATA / "masks_fine" / f"{stem}.png")
        feats[stem] = seg_features(lab)
    return feats


@torch.no_grad()
def build_img_feats(stems, device, backbone_name):
    import timm
    model = timm.create_model(backbone_name, pretrained=True, num_classes=0).eval().to(device)
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg, is_training=False)
    feats = {}
    for stem in stems:
        img = Image.open(DATA / "images" / f"{stem}.png").convert("RGB")
        feats[stem] = model(tf(img).unsqueeze(0).to(device)).squeeze(0).float().cpu().numpy()
    return feats


class Head(nn.Module):
    def __init__(self, d_in, n_out=14, hidden=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
            nn.Dropout(p), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["seg", "hybrid", "image"], default="seg")
    ap.add_argument("--backbone", default="vit_base_patch16_dinov3.lvd1689m")
    ap.add_argument("--val-cases", nargs="+",
                    default=["center_1_case_14", "center_1_case_15"])
    ap.add_argument("--val-pred-dir", type=Path, default=None,
                    help="nnU-Net T2 predTs dir -> use PREDICTED masks for val frames")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path,
                    default=Path("/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t3/ckpt_segcond"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} mode={args.mode} feat_ids={FEAT_IDS}")

    labels = load_labels()
    stems = sorted(labels.keys())
    val_cases = set(args.val_cases)
    val_set = {s for s in stems if case_of(s) in val_cases}
    tr = [s for s in stems if s not in val_set]
    va = [s for s in stems if s in val_set]
    print(f"frames {len(stems)} | train {len(tr)} | val {len(va)} | "
          f"val_pred_dir={'yes' if args.val_pred_dir else 'GT-masks'}")

    parts = []
    if args.mode in ("seg", "hybrid"):
        sf = build_seg_feats(stems, val_set, args.val_pred_dir)
        parts.append(sf)
        print(f"seg feat dim = {len(next(iter(sf.values())))}")
    if args.mode in ("hybrid", "image"):
        imf = build_img_feats(stems, device, args.backbone)
        parts.append(imf)
        print(f"img feat dim = {len(next(iter(imf.values())))}")

    def vec(stem):
        return np.concatenate([p[stem] for p in parts]).astype(np.float32)

    Xtr = torch.tensor(np.stack([vec(s) for s in tr])).to(device)
    Ytr = torch.tensor(np.stack([labels[s] for s in tr])).to(device)
    Xva = torch.tensor(np.stack([vec(s) for s in va])).to(device)
    d = Xtr.shape[1]
    print(f"input dim = {d}")

    # standardise (seg features have very different scales)
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd

    pos = Ytr.sum(0).clamp(min=1); neg = (Ytr.shape[0] - pos).clamp(min=1)
    pos_weight = (neg / pos).clamp(max=20.0)
    head = Head(d).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    head.train()
    for ep in range(args.epochs):
        opt.zero_grad()
        loss = lossf(head(Xtr), Ytr)
        loss.backward(); opt.step()
        if (ep + 1) % 100 == 0 or ep == 0:
            print(f"  ep {ep+1:3d}  loss {loss.item():.4f}")

    head.eval()
    with torch.no_grad():
        prob_va = torch.sigmoid(head(Xva)).cpu().numpy()

    pred_csv = args.outdir / "val_pred.csv"
    gt_csv = args.outdir / "val_gt.csv"
    with open(pred_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case_id"] + STATIONS)
        for s, p in zip(va, prob_va):
            w.writerow([s] + [f"{v:.6f}" for v in p])
    with open(gt_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case_id"] + STATIONS)
        for s in va:
            w.writerow([s] + [int(v) for v in labels[s]])

    from metrics.evaluate_cls import evaluate as eval_cls
    res = eval_cls(gt_csv, pred_csv, classes=CLASSES_STATIONS)
    out = {"mode": args.mode, "final_f1": res["final_f1"], "final_auroc": res["final_auroc"],
           "n_val": len(va), "val_cases": sorted(val_cases),
           "used_pred_masks": bool(args.val_pred_dir),
           "per_station": {k: {"f1": v["f1"], "auroc": v["auroc"]}
                           for k, v in res["class_scores"].items()}}
    (args.outdir / "metrics.json").write_text(json.dumps(out, indent=2))
    torch.save({"head": head.state_dict(), "mode": args.mode, "feat_ids": FEAT_IDS,
                "stations": STATIONS, "mu": mu.cpu(), "sd": sd.cpu()}, args.outdir / "best.pt")
    print("=" * 56)
    print(f"T3 [{args.mode}] weighted F1@0.5 = {out['final_f1']:.4f}  "
          f"AUROC = {out['final_auroc']:.4f}  (val n={len(va)})")


if __name__ == "__main__":
    main()
