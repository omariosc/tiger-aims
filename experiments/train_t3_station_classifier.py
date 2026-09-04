#!/usr/bin/env python3
"""
TIGER SQ-AI Task 3 — lymph-node-station visibility (multi-label, 14 stations).

Frame -> 14 independent sigmoid outputs. Backbone from the open FM pool
(default: DINOv3 ViT-B/16, timm `vit_base_patch16_dinov3.lvd1689m` in hf_cache;
public LVD-1689M pretrain -> compliant). Backbone frozen by default (140 frames
is tiny -> freeze to avoid overfit), linear/MLP head trained with BCE.

Labels: lymph_node_station_visibility.csv, multi-hot over the OFFICIAL station
order from metrics.classes_stations.CLASSES_STATIONS:
  [6L,6R,7L,7R,8,9,10L,10R,11L,11R,12L,12R,13L,13R]

Case-disjoint split (same held-out cases as the seg tasks). Metric mirrors the
challenge scorer: weighted-macro F1@0.5 + AUROC (metrics.evaluate_cls.evaluate),
which we invoke directly on the val-prediction CSV we emit (challenge schema).

Outputs (predictable paths under --outdir):
  best.pt                  — head weights + config
  val_pred.csv             — challenge-schema probs (case_id,6L,...,13R) for val frames
  val_gt.csv               — matching binary GT (for evaluate_cls)
  metrics.json             — final weighted F1 + AUROC (via official evaluate_cls)
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
from metrics.classes_stations import CLASSES_STATIONS, STATION_NAMES  # official order

DATA = Path("/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/synapse")
STATIONS = STATION_NAMES                       # 14 names in official column order
S2I = {s: i for i, s in enumerate(STATIONS)}


def case_of(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def load_labels():
    """Return {frame_stem: multihot(14)} from the visibility CSV.
    Frame stem = <case>_<annotated frame> e.g. center_1_case_6_6R."""
    df = pd.read_csv(DATA / "lymph_node_station_visibility.csv")
    df.columns = [c.strip() for c in df.columns]
    frame_col = [c for c in df.columns if "annotated" in c.lower()][0]
    vis_col = [c for c in df.columns if "visible" in c.lower()][0]
    out = {}
    for _, row in df.iterrows():
        case = str(row["case"]).strip()
        frame = str(row[frame_col]).strip()
        stem = f"{case}_{frame}"
        vec = np.zeros(len(STATIONS), dtype=np.float32)
        for tok in str(row[vis_col]).split(","):
            tok = tok.strip()
            if tok in S2I:
                vec[S2I[tok]] = 1.0
        out[stem] = vec
    return out


def build_backbone(name: str, device):
    import timm
    model = timm.create_model(name, pretrained=True, num_classes=0)  # feature extractor
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    cfg = timm.data.resolve_model_data_config(model)
    tf = timm.data.create_transform(**cfg, is_training=False)
    feat_dim = model.num_features
    return model, tf, feat_dim


class Head(nn.Module):
    def __init__(self, d_in, n_out=14, hidden=512, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
            nn.Dropout(p), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def embed_all(stems, backbone, tf, device):
    feats = {}
    for stem in stems:
        img = Image.open(DATA / "images" / f"{stem}.png").convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        feats[stem] = backbone(x).squeeze(0).float().cpu()
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_base_patch16_dinov3.lvd1689m")
    ap.add_argument("--val-cases", nargs="+",
                    default=["center_1_case_14", "center_1_case_15"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path,
                    default=Path("/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t3/ckpt"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} backbone={args.backbone}")

    labels = load_labels()
    stems = sorted(labels.keys())
    val_cases = set(args.val_cases)
    tr = [s for s in stems if case_of(s) not in val_cases]
    va = [s for s in stems if case_of(s) in val_cases]
    print(f"frames: {len(stems)} | train {len(tr)} | val {len(va)} "
          f"(val cases {sorted(val_cases)})")
    print(f"positive rate per station (train): "
          f"{dict(zip(STATIONS, np.stack([labels[s] for s in tr]).mean(0).round(2)))}")

    backbone, tf, d = build_backbone(args.backbone, device)
    print(f"feat_dim={d}")
    feats = embed_all(stems, backbone, tf, device)

    Xtr = torch.stack([feats[s] for s in tr]).to(device)
    Ytr = torch.tensor(np.stack([labels[s] for s in tr])).to(device)
    Xva = torch.stack([feats[s] for s in va]).to(device)

    # pos_weight for class imbalance (some stations rare)
    pos = Ytr.sum(0).clamp(min=1); neg = (Ytr.shape[0] - pos).clamp(min=1)
    pos_weight = (neg / pos).clamp(max=20.0)
    head = Head(d).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    head.train()
    for ep in range(args.epochs):
        opt.zero_grad()
        logit = head(Xtr)
        loss = lossf(logit, Ytr)
        loss.backward(); opt.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  ep {ep+1:3d}  loss {loss.item():.4f}")

    head.eval()
    with torch.no_grad():
        prob_va = torch.sigmoid(head(Xva)).cpu().numpy()

    # write challenge-schema CSVs (case_id + 14 station columns in official order)
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

    # score with the OFFICIAL evaluate_cls
    from metrics.evaluate_cls import evaluate as eval_cls
    res = eval_cls(gt_csv, pred_csv, classes=CLASSES_STATIONS)
    out = {"final_f1": res["final_f1"], "final_auroc": res["final_auroc"],
           "n_val": len(va), "val_cases": sorted(val_cases),
           "per_station": {k: {"f1": v["f1"], "auroc": v["auroc"]}
                           for k, v in res["class_scores"].items()}}
    (args.outdir / "metrics.json").write_text(json.dumps(out, indent=2))
    torch.save({"head": head.state_dict(), "backbone": args.backbone,
                "stations": STATIONS}, args.outdir / "best.pt")
    print("=" * 56)
    print(f"T3 weighted F1@0.5 = {out['final_f1']:.4f}  "
          f"AUROC = {out['final_auroc']:.4f}  (val n={len(va)})")
    print(f"wrote {pred_csv}, {gt_csv}, metrics.json, best.pt")


if __name__ == "__main__":
    main()
