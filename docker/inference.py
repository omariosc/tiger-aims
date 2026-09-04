#!/usr/bin/env python3
"""
TIGER SQ-AI 2026 — full 3-task inference for the Docker submission.

Champion stack (CONFIRMED 2026-06-25, job 6291940):
  * task1 (MERGED 15/16-cls) <- nnU-Net Dataset201_TigerT1coarse, TopK-Tversky (w3 0.552)
  * task2 (FINE 30/31-cls)   <- Dataset202 3-seed Focal-Tversky ENSEMBLE
                                + replace{8} CT205-synth graft + FP   (w3 0.7092)
  * task3 (14 stations)      <- seg-conditioned MLP head (ckpt_segcond_ensmask, F1 0.6276)
                                on the FINE (task2) masks

FINE replace{8} graft (post-process baked here, §0m.4):
  champion FINE ensemble argmax = the 3-seed FT ensemble on Dataset202 (RAW, pre-FP).
  CT205 synth ensemble argmax   = the 3-seed FT ensemble on Dataset205_TigerT2fineSynthCT
                                  (the anatomy-aware diffusion-synth model; class-8
                                  Right_Inf_Pulmonary_Lig is the ONE class it beats champ on).
  replace{8}: clear the champion's own id-8 pixels, then write the synth's id-8 wherever
  the synth predicts it (synth NN-resized to the champ shape if planes differ). FP-suppress
  is applied AFTER the graft (exactly as the scored champion was produced). Env-gated by
  TIGER_FINE_GRAFT (default on); set TIGER_FINE_GRAFT=0 to fall back to the plain ensemble.

  !!! TASK MAPPING — CORRECTED 2026-07-19 vs the PUBLISHED contract !!!
  The OFFICIAL submission `task1/` = MERGED ("Simplified", 15/16-cls) and
  `task2/` = FINE ("Full", 30/31-cls).  The organizers FLIPPED task1<->task2 after
  our 2026-06-15 clone; verified against Docker-Instructions wiki (639935), Tasks wiki
  (639931), and metrics/01_evaluate_challenge.py @origin/main (task1 scored with
  CLASSES_MERGED, task2 with the fine registry).  Our internal nnU-Net datasets are
  Dataset201 = coarse/MERGED and Dataset202 = FINE.  This script now maps them
  correctly:  task1 <- Dataset201(merged),  task2 <- Dataset202(fine).

I/O (per official Docker-Instructions wiki 639935 / metrics/00_README.md @origin/main):
  INPUT_DIR  : flat RGB PNG frames  <case_id>_<station>.png  (e.g. center_1_case_6_6L.png)
               NATIVE resolution, VARIABLE between centres -> do NOT assume a fixed size.
  OUTPUT_DIR/
     task1/<stem>.png   exact-RGB MERGED masks (15/16-cls, lossless PNG, official colours)
     task2/<stem>.png   exact-RGB FINE masks   (30/31-cls, lossless PNG, official colours)
     task3.csv          case_id,6L,6R,7L,7R,8,9,10L,10R,11L,11R,12L,12R,13L,13R  (probabilities)

Run:  python inference.py --input <frames_dir> --output <out_dir>
Env (defaults match the in-container layout):
  TIGER_RESULTS  = nnU-Net results root (contains the two Dataset dirs)
  TIGER_T3_CKPT  = seg-cond T3 head best.pt
  TIGER_REPO     = challenge_repo (for the official class registries)
"""
from __future__ import annotations
import argparse, os, sys, csv, json, shutil, tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ----------------------------------------------------------------------------- config
RESULTS = Path(os.environ.get("TIGER_RESULTS",
                              "/opt/tiger/nnUNet_results"))
T3_CKPT = Path(os.environ.get("TIGER_T3_CKPT",
                              "/opt/tiger/t3_head/best.pt"))
REPO = Path(os.environ.get("TIGER_REPO", "/opt/tiger/challenge_repo"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

# Official registries (id -> name/rgb/weight) -> the canonical colour LUTs.
from metrics.classes import CLASSES as FINE_CLASSES            # 31 fine
from metrics.classes_merged import CLASSES_MERGED              # 16 merged
from metrics.classes_stations import CLASSES_STATIONS, STATION_NAMES
from station_boundary_map import STATION_BOUNDARY_FINE, SHARED_FINE
from fp_suppress import apply_area_filter

# Test-time FP-suppression configs (tuned on val cases 14,15, FULL-RES; lever
# CONFIRMED 2026-06-16: w3 sub-avg +0.157 task1(fine) / +0.086 task2(merged),
# zero present-frame recall loss). Fractional thresholds -> resolution-invariant.
# FINAL stack (2026-06-19): FP configs tuned ON the winning preds (ensemble / TopK).
FP_CFG_FINE   = Path(os.environ.get("TIGER_FP_FINE",
                     str(Path(__file__).parent / "t2_ft_ens_fpsuppress_fullres.json")))
FP_CFG_MERGED = Path(os.environ.get("TIGER_FP_MERGED",
                     str(Path(__file__).parent / "t1_topk_fpsuppress_fullres.json")))


def load_fp_cfg(p):
    if not Path(p).exists():
        print(f"[tiger] WARN FP-suppress cfg {p} missing -> no suppression")
        return None
    return json.loads(Path(p).read_text())


def apply_fp_suppression(id_dir: Path, cfg):
    """Per-class area-filter the argmax id PNGs in place (resolution-invariant fracs)."""
    if cfg is None:
        return
    nfilt = 0
    for p in sorted(id_dir.glob("*.png")):
        ids = np.array(Image.open(p))
        if ids.ndim == 3:
            ids = ids[:, :, 0]
        px = float(ids.shape[0] * ids.shape[1])
        mb = {int(k): int(round(v * px)) for k, v in cfg["min_blob_frac"].items()}
        mt = {int(k): int(round(v * px)) for k, v in cfg["min_total_frac"].items()}
        out = apply_area_filter(ids, min_blob=mb, min_total=mt)
        if not np.array_equal(out, ids):
            nfilt += 1
        Image.fromarray(out.astype(np.uint8), mode="L").save(p)
    print(f"[tiger] FP-suppress: filtered {nfilt} masks in {id_dir.name}")

# ---------------------------------------------------------------------------------------------
# ★ stack.env IS THE SINGLE SOURCE OF TRUTH FOR *WHICH MODELS THIS IMAGE CONTAINS*.
# The staging step writes it, the build assertions verify against it, and inference now READS it.
# WHY (build 7720071, caught by the contract gate and by nothing else): the image shipped the
# adopted Dataset212 weights and every build assertion passed -- but the Dockerfile sets only
# TIGER_RESULTS/T3_CKPT/REPO, never TIGER_FINE_DATASET. So inference fell back to its hardcoded
# "Dataset202_TigerT2fine" default and died with
#     FileNotFoundError: .../Dataset202_TigerT2fine/.../dataset.json
# i.e. RIGHT WEIGHTS, WRONG POINTER. With the old stack the defaults happened to match, so this
# latent gap never showed. Reading stack.env closes it for good: weights, assertions and runtime
# now all derive from ONE declaration.
# Precedence: an explicit environment variable still wins (so a run can be overridden for testing),
# then stack.env, then the historical hardcoded default.
_STACK = {}
try:
    _sp = Path(os.environ.get("TIGER_STACK_ENV", "/opt/tiger/stack.env"))
    if _sp.is_file():
        for _line in _sp.read_text().splitlines():
            if "=" in _line and not _line.lstrip().startswith("#"):
                _k, _v = _line.split("=", 1)
                _STACK[_k.strip()] = _v.strip()
        print(f"[tiger] stack.env: " + " ".join(
            f"{k}={_STACK[k]}" for k in ("FINE_DATASET", "MERGED_DATASET", "FINE_GRAFT")
            if k in _STACK), flush=True)
except Exception as _e:  # never let provenance reading break inference
    print(f"[tiger] WARNING: could not read stack.env ({_e!r}); using env/defaults", flush=True)

def _cfg(env_key, stack_key, default):
    """explicit env > stack.env > hardcoded default"""
    v = os.environ.get(env_key)
    if v is not None and v != "":
        return v
    return _STACK.get(stack_key, default)
# ---------------------------------------------------------------------------------------------

# nnU-Net dataset dirs (INTERNAL names; mapped to the OFFICIAL task on output).
# Trainer dirs are env-overridable so the Docker can bank the confirmed winners
# (FINAL stack, 2026-06-19, ranked at 160px §0i, deciding metric = w3 sub-avg Dice):
#   official task1 FINE   = 3-seed Focal-Tversky ENSEMBLE  (w3 0.701, +0.107 vs single)
#   official task2 MERGED = TopK-Tversky single model      (w3 0.552, +0.175 vs banked)
FINE_DATASET   = _cfg("TIGER_FINE_DATASET", "FINE_DATASET", "Dataset202_TigerT2fine")
FINE_TRAINER   = _cfg("TIGER_FINE_TRAINER", "FINE_TRAINER",
                      "nnUNetTrainer_250epochs_NoMirror_FocalTversky__nnUNetPlans__2d")
# Comma-separated list of FINE trainers to softmax-prob-average (3-seed ensemble).
# If set (default = the confirmed 3-seed ens), FINE uses ensemble inference.
FINE_ENSEMBLE_TRAINERS = _cfg(
    "TIGER_FINE_ENSEMBLE_TRAINERS", "FINE_ENSEMBLE_TRAINERS",
    "nnUNetTrainer_250epochs_NoMirror_FocalTversky__nnUNetPlans__2d,"
    "nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed1__nnUNetPlans__2d,"
    "nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed2__nnUNetPlans__2d")
MERGED_DATASET = _cfg("TIGER_MERGED_DATASET", "MERGED_DATASET", "Dataset201_TigerT1coarse")
MERGED_TRAINER = _cfg("TIGER_MERGED_TRAINER", "MERGED_TRAINER",
                      "nnUNetTrainer_250epochs_NoMirror_TopKTversky__nnUNetPlans__2d")

# ---- FINE replace{8} CT205-synth graft (champion 2026-06-25, §0m.4) -----------
# After the champion FINE ensemble argmax, run the CT205 diffusion-synth ensemble
# and REPLACE-graft class 8 (Right_Inf_Pulmonary_Lig) — the one class where synth
# beats the champion (synth 0.818 > champ 0.736). The graft is applied to the RAW
# (pre-FP) ensemble masks; FP-suppress runs afterwards. Env-gated (default ON).
FINE_GRAFT          = _cfg("TIGER_FINE_GRAFT", "FINE_GRAFT", "1") not in ("0", "", "false", "False")
FINE_GRAFT_IDS      = [int(x) for x in _cfg("TIGER_FINE_GRAFT_IDS", "FINE_GRAFT_IDS", "8").split(",") if x.strip()]
SYNTH_DATASET       = _cfg("TIGER_SYNTH_DATASET", "SYNTH_DATASET", "Dataset205_TigerT2fineSynthCT")
SYNTH_ENSEMBLE_TRAINERS = os.environ.get(
    "TIGER_SYNTH_ENSEMBLE_TRAINERS",
    "nnUNetTrainer_250epochs_NoMirror_FocalTversky__nnUNetPlans__2d,"
    "nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed1__nnUNetPlans__2d,"
    "nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed2__nnUNetPlans__2d")

STATIONS = STATION_NAMES
FEAT_IDS = sorted(set([i for v in STATION_BOUNDARY_FINE.values() for i in v] + SHARED_FINE))
FID2COL = {fid: j for j, fid in enumerate(FEAT_IDS)}

# ---- T-1 acquisition-station CO-VISIBILITY PRIOR (2026-07-20) -----------------
# Every test frame is `<case>_<station>.png` and the published Docker contract
# GUARANTEES input filenames arrive unchanged (metrics/00_README.md also mandates the
# task3 row key be exactly that stem) => the annotated station is documented,
# contract-guaranteed input metadata. M[a][s] = P(s visible | annotated station a),
# additive-shrunk toward the pooled marginal prevalence (alpha=2.0, fit on all 10
# training cases). Blend is the PRE-REGISTERED lambda=0.5 (see PREREGISTRATION.md);
# TIGER_T3_PRIOR_MODE=prior|blend|off, TIGER_T3_PRIOR_LAMBDA override for ablation.
# FAIL-CLOSED: any station token we cannot parse EXACTLY -> pure seg-conditioned output.
T3_PRIOR_CFG = Path(os.environ.get("TIGER_T3_PRIOR",
                    str(Path(__file__).parent / "covis_matrix.json")))
T3_PRIOR_MODE = os.environ.get("TIGER_T3_PRIOR_MODE", "blend").strip().lower()
T3_PRIOR_LAMBDA = float(os.environ.get("TIGER_T3_PRIOR_LAMBDA", "0.5"))


def load_station_prior(p: Path):
    """Load the co-visibility matrix. Returns None if absent/malformed (=> prior OFF)."""
    try:
        d = json.loads(Path(p).read_text())
        st = list(d["stations"])
        if st != list(STATIONS):
            print(f"[tiger][T3-PRIOR] station order mismatch in {p} -> PRIOR DISABLED")
            return None
        M = np.asarray(d["matrix"], dtype=np.float64)
        if M.shape != (len(STATIONS), len(STATIONS)) or not np.isfinite(M).all():
            print(f"[tiger][T3-PRIOR] bad matrix shape/values in {p} -> PRIOR DISABLED")
            return None
        print(f"[tiger][T3-PRIOR] loaded {M.shape[0]}x{M.shape[1]} co-visibility matrix "
              f"from {p.name} (mode={T3_PRIOR_MODE}, lambda={T3_PRIOR_LAMBDA})")
        return {s: M[i] for i, s in enumerate(STATIONS)}
    except Exception as e:
        print(f"[tiger][T3-PRIOR] could not load {p} ({e}) -> PRIOR DISABLED")
        return None


# STRICT station token parse. The stem must end in `_<station>` where <station> is
# EXACTLY one of the 14 official tokens (case-sensitive). Anything else fails closed.
_STATION_SET = set(STATIONS)


def parse_station(stem: str):
    """`center_1_case_3_6R` -> '6R'. Returns None on ANY deviation (fail-closed)."""
    if not stem or "_" not in stem:
        return None
    tok = stem.rsplit("_", 1)[1]
    return tok if tok in _STATION_SET else None


# ----------------------------------------------------------------------------- helpers
def id_to_rgb_lut(classes):
    lut = np.zeros((256, 3), np.uint8)
    for c in classes:
        lut[c.label_id] = c.rgb
    return lut

FINE_LUT = id_to_rgb_lut(FINE_CLASSES)
MERGED_LUT = id_to_rgb_lut(CLASSES_MERGED)


def rgb_to_grayscale_channels(input_dir: Path, work: Path):
    """Convert each RGB frame -> nnU-Net 3x grayscale (_0000/_0001/_0002) PNGs."""
    work.mkdir(parents=True, exist_ok=True)
    stems = []
    for p in sorted(input_dir.glob("*.png")):
        stem = p.stem
        stems.append(stem)
        rgb = np.array(Image.open(p).convert("RGB"))
        for ch in range(3):
            Image.fromarray(rgb[:, :, ch], mode="L").save(work / f"{stem}_{ch:04d}.png")
    return stems


def run_nnunet(dataset, trainer, in_dir: Path, out_dir: Path):
    """Predict integer-id masks with the banked nnU-Net model (single fold 0, no TTA)."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,  # flip-safe
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False, allow_tqdm=False)
    model_folder = str(RESULTS / dataset / trainer)
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(0,),
                                                    checkpoint_name="checkpoint_final.pth")
    predictor.predict_from_files(str(in_dir), str(out_dir),
                                 save_probabilities=False, overwrite=True,
                                 num_processes_preprocessing=2,
                                 num_processes_segmentation_export=2)


def run_nnunet_ensemble(dataset, trainers, in_dir: Path, out_dir: Path, work: Path):
    """3-seed nnU-Net ensemble: predict each trainer WITH probabilities, softmax-
    average the per-class probs, argmax -> integer-id PNG (matches nnUNetv2_ensemble).
    Falls back to single-model run_nnunet if only one trainer is given."""
    trainers = [t for t in trainers if t.strip()]
    if len(trainers) == 1:
        run_nnunet(dataset, trainers[0], in_dir, out_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    member_dirs = []
    for k, tr in enumerate(trainers):
        md = work / f"ens_member_{k}"
        md.mkdir(parents=True, exist_ok=True)
        print(f"[tiger]   ensemble member {k}: {tr}")
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=False,  # flip-safe
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(
            str(RESULTS / dataset / tr), use_folds=(0,),
            checkpoint_name="checkpoint_final.pth")
        predictor.predict_from_files(str(in_dir), str(md),
                                     save_probabilities=True, overwrite=True,
                                     num_processes_preprocessing=2,
                                     num_processes_segmentation_export=2)
        member_dirs.append(md)
    # average softmax probs across members, argmax -> id PNG
    stems = sorted(p.stem for p in member_dirs[0].glob("*.npz"))
    for stem in stems:
        acc = None
        for md in member_dirs:
            prob = np.load(md / f"{stem}.npz")["probabilities"].astype(np.float32)
            acc = prob if acc is None else acc + prob
        acc /= len(member_dirs)
        ids = acc.argmax(axis=0).squeeze().astype(np.uint8)  # (C,1,H,W)->(H,W)
        Image.fromarray(ids, mode="L").save(out_dir / f"{stem}.png")
    print(f"[tiger]   ensemble: averaged {len(trainers)} members -> {len(stems)} id masks")


def replace_graft(champ_dir: Path, synth_dir: Path, graft_ids, out_dir: Path):
    """REPLACE-mode argmax graft (exact replica of fuse_argmax_graft.py --mode replace).
    Per gid: clear the champion's own gid pixels (champ==gid -> 0), then write the
    synth's gid everywhere synth==gid. The synth mask is NN-resized to the champ shape
    if planes differ (cross-dataset: champ Dataset202 896x1536 vs synth Dataset205 512x512).
    All non-grafted champ labels untouched. Writes <stem>.png id masks to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    graft = set(graft_ids)
    n_written = n_cleared = 0
    for cf in sorted(champ_dir.glob("*.png")):
        sf = synth_dir / cf.name
        champ = np.array(Image.open(cf))
        if champ.ndim == 3:
            champ = champ[:, :, 0]
        if not sf.exists():
            # no synth donor for this frame -> keep champ unchanged
            Image.fromarray(champ.astype(np.uint8), mode="L").save(out_dir / cf.name)
            continue
        synth = np.array(Image.open(sf))
        if synth.ndim == 3:
            synth = synth[:, :, 0]
        if synth.shape != champ.shape:
            synth = np.array(Image.fromarray(synth.astype(np.uint8), mode="L").resize(
                (champ.shape[1], champ.shape[0]), Image.NEAREST))
        out = champ.copy()
        for gid in graft:
            clr = (out == gid)
            out[clr] = 0
            n_cleared += int(clr.sum())
            sel = (synth == gid)
            out[sel] = gid
            n_written += int(sel.sum())
        Image.fromarray(out.astype(np.uint8), mode="L").save(out_dir / cf.name)
    print(f"[tiger]   replace-graft ids {sorted(graft)}: "
          f"{n_written} synth px written, {n_cleared} champ px cleared -> {out_dir.name}")


def ids_to_rgb_png(id_dir: Path, out_dir: Path, lut: np.ndarray):
    """nnU-Net writes <stem>.png integer-id (mode L). Map -> exact-RGB lossless PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(id_dir.glob("*.png")):
        ids = np.array(Image.open(p))
        if ids.ndim == 3:
            ids = ids[:, :, 0]
        rgb = lut[ids]
        Image.fromarray(rgb, mode="RGB").save(out_dir / p.name)  # PNG = lossless


def seg_features(label_arr, downscale=384):
    if downscale and max(label_arr.shape) > downscale:
        h, w = label_arr.shape
        s = downscale / max(h, w)
        label_arr = np.array(Image.fromarray(label_arr.astype(np.uint8)).resize(
            (max(1, int(w * s)), max(1, int(h * s))), Image.NEAREST))
    h, w = label_arr.shape
    tot = float(h * w)
    pres = np.zeros(len(FEAT_IDS), np.float32)
    area = np.zeros(len(FEAT_IDS), np.float32)
    ids, counts = np.unique(label_arr, return_counts=True)
    for i, c in zip(ids, counts):
        if int(i) in FID2COL:
            j = FID2COL[int(i)]
            pres[j] = 1.0
            area[j] = np.log1p(c / tot * 100.0)
    return np.concatenate([pres, area])


class Head(nn.Module):
    def __init__(self, d_in, n_out=14, hidden=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
            nn.Dropout(p), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def predict_t3(fine_id_dir: Path, stems, out_csv: Path):
    """Seg-conditioned T3 (+ optional acquisition-station co-visibility prior).

    Per frame the path taken is LOGGED explicitly:
      PRIOR+SEG  = station token parsed, blended with the seg-cond head
      PRIOR-ONLY = station token parsed, prior used alone (mode=prior)
      SEG-ONLY   = FALLBACK: token unparseable / prior unavailable -> seg-cond only
    The fallback never crashes and never silently degrades: it is always announced.
    """
    ck = torch.load(T3_CKPT, map_location="cpu")
    mu, sd = ck["mu"], ck["sd"]
    d_in = mu.shape[1]
    head = Head(d_in)
    head.load_state_dict(ck["head"])
    head.eval()

    prior = None if T3_PRIOR_MODE == "off" else load_station_prior(T3_PRIOR_CFG)
    lam = min(1.0, max(0.0, T3_PRIOR_LAMBDA))
    n_prior = n_fallback = 0

    rows = []
    for stem in stems:
        idp = fine_id_dir / f"{stem}.png"
        ids = np.array(Image.open(idp))
        if ids.ndim == 3:
            ids = ids[:, :, 0]
        feat = torch.tensor(seg_features(ids)).unsqueeze(0).float()
        feat = (feat - mu) / sd
        with torch.no_grad():
            seg_prob = torch.sigmoid(head(feat)).squeeze(0).numpy().astype(np.float64)

        station = parse_station(stem) if prior is not None else None
        if station is None:
            prob, path = seg_prob, "SEG-ONLY(fallback)"
            n_fallback += 1
        else:
            p_prior = prior[station]
            if T3_PRIOR_MODE == "prior":
                prob, path = p_prior, f"PRIOR-ONLY[{station}]"
            else:
                prob = lam * p_prior + (1.0 - lam) * seg_prob
                path = f"PRIOR+SEG[{station},lam={lam}]"
            n_prior += 1
        prob = np.clip(prob, 0.0, 1.0)
        print(f"[tiger][T3] {stem}: {path}")
        rows.append((stem, prob))

    print(f"[tiger][T3] station-prior used on {n_prior}/{len(stems)} frames; "
          f"seg-only fallback on {n_fallback}/{len(stems)}")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id"] + STATIONS)
        for stem, prob in rows:
            w.writerow([stem] + [f"{v:.6f}" for v in prob])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path(os.environ.get("TIGER_INPUT", "/input")))
    ap.add_argument("--output", type=Path,
                    default=Path(os.environ.get("TIGER_OUTPUT", "/output")))
    ap.add_argument("--skip-seg", action="store_true",
                    help="(smoke) skip nnU-Net; expect pre-made id masks in --fine-id/--merged-id")
    ap.add_argument("--fine-id", type=Path, default=None)
    ap.add_argument("--merged-id", type=Path, default=None)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    task1_dir = args.output / "task1"   # MERGED (official task1 = "Simplified", 15/16 cls)
    task2_dir = args.output / "task2"   # FINE   (official task2 = "Full", 30/31 cls)
    task3_csv = args.output / "task3.csv"

    work = Path(tempfile.mkdtemp(prefix="tiger_"))
    print(f"[tiger] input={args.input} output={args.output} work={work}")

    if args.skip_seg:
        fine_id_dir = args.fine_id
        merged_id_dir = args.merged_id
        stems = sorted(p.stem for p in fine_id_dir.glob("*.png"))
    else:
        gray = work / "gray"
        stems = rgb_to_grayscale_channels(args.input, gray)
        print(f"[tiger] {len(stems)} frames -> grayscale channels")
        fine_id_dir = work / "fine_ids"
        merged_id_dir = work / "merged_ids"
        fine_trainers = [t for t in FINE_ENSEMBLE_TRAINERS.split(",") if t.strip()]
        print(f"[tiger] nnU-Net FINE (task1) — {len(fine_trainers)}-seed ensemble ...")
        run_nnunet_ensemble(FINE_DATASET, fine_trainers, gray, fine_id_dir, work)
        print(f"[tiger] nnU-Net MERGED (task2) — {MERGED_TRAINER.split('__')[0]} ...")
        run_nnunet(MERGED_DATASET, MERGED_TRAINER, gray, merged_id_dir)

        # FINE replace{8} CT205-synth graft (champion, §0m.4): run the synth ensemble,
        # REPLACE-graft the win class(es) onto the champion ensemble (RAW masks, pre-FP),
        # then continue with FP-suppress on the grafted masks. Env-gated (default ON).
        if FINE_GRAFT and FINE_GRAFT_IDS:
            synth_trainers = [t for t in SYNTH_ENSEMBLE_TRAINERS.split(",") if t.strip()]
            print(f"[tiger] FINE graft — CT205 synth {len(synth_trainers)}-seed ensemble "
                  f"({SYNTH_DATASET}) for replace{{{','.join(map(str, FINE_GRAFT_IDS))}}} ...")
            synth_id_dir = work / "synth_ids"
            synth_work = work / "synth_ens"
            run_nnunet_ensemble(SYNTH_DATASET, synth_trainers, gray, synth_id_dir, synth_work)
            grafted_dir = work / "fine_ids_grafted"
            replace_graft(fine_id_dir, synth_id_dir, FINE_GRAFT_IDS, grafted_dir)
            fine_id_dir = grafted_dir   # downstream FP-suppress / id->RGB / T3 use the grafted masks
        else:
            print("[tiger] FINE graft DISABLED (TIGER_FINE_GRAFT=0) — plain ensemble")

    # Test-time FP-suppression on the argmax id masks (before id->RGB and before T3,
    # so the seg-cond T3 head sees the cleaned fine masks too).
    print("[tiger] FP-suppress task1 (fine) + task2 (merged)")
    apply_fp_suppression(fine_id_dir,   load_fp_cfg(FP_CFG_FINE))
    apply_fp_suppression(merged_id_dir, load_fp_cfg(FP_CFG_MERGED))

    print("[tiger] id->RGB  task1 (MERGED, 15/16-cls)")
    ids_to_rgb_png(merged_id_dir, task1_dir, MERGED_LUT)
    print("[tiger] id->RGB  task2 (FINE, 30/31-cls)")
    ids_to_rgb_png(fine_id_dir, task2_dir, FINE_LUT)
    print("[tiger] seg-conditioned T3 (on FINE masks) -> task3.csv")
    predict_t3(fine_id_dir, stems, task3_csv)

    shutil.rmtree(work, ignore_errors=True)
    print(f"[tiger] DONE. task1={len(list(task1_dir.glob('*.png')))} "
          f"task2={len(list(task2_dir.glob('*.png')))} "
          f"task3 rows={sum(1 for _ in open(task3_csv))-1}")


if __name__ == "__main__":
    main()
