#!/bin/bash
#SBATCH --job-name=tiger_dockerbuild
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%j.err
# AUTONOMOUS LADDER step B: actually BUILD the submission container with apptainer
# (podman rootless is unavailable on this cluster: no subuid range) and RUN it
# end-to-end on a few val frames so it is BUILD-VERIFIED submit-ready.
# Reads step A's decision lines to bank the 1000-ep model per task if CONFIRMED.
set -e
module load apptainer/1.3.6
export APPTAINER_CACHEDIR=/scratch/USERNAME/apptainer_cache
export APPTAINER_TMPDIR=/scratch/USERNAME/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
RES=/users/USERNAME/AUTONOMOUS_RESULTS.md
CTX=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/docker_ctx
# Build to a NEW sif/sandbox; the LIVE tiger.sif is only overwritten after the
# smoke verification passes (safety-net: tiger_prev.sif = the previous champion).
SIF=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/tiger_new.sif
LIVE_SIF=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/tiger.sif
SANDBOX=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/tiger_sandbox_new
NN=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet
D=$(date +%F)
log(){ echo "[Tiger][$SLURM_JOB_ID] $1" >> $RES; echo ">>> $1"; }

# ---- (0) CHAMPION stack (CONFIRMED 2026-06-25 job 6291940, deciding metric = w3 sub-avg @160):
#   FINE   (task1) = 3-seed FT ENSEMBLE + replace{8} CT205-synth graft + FP (w3 0.7092, +0.008)
#   MERGED (task2) = TopK-Tversky single model      (t1_topk_fp,   w3 0.552, +0.175 vs banked)
#   T3            = seg-cond head on ENSEMBLE masks (ckpt_segcond_ensmask, F1 0.6276, +0.046)
export FINE_ENS_TRS="nnUNetTrainer_250epochs_NoMirror_FocalTversky__nnUNetPlans__2d,nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed1__nnUNetPlans__2d,nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed2__nnUNetPlans__2d"
export FINE_TR0=nnUNetTrainer_250epochs_NoMirror_FocalTversky__nnUNetPlans__2d
export MERGED_TR=nnUNetTrainer_250epochs_NoMirror_TopKTversky__nnUNetPlans__2d
# CT205 diffusion-synth ensemble for the replace{8} graft (same trainer names, Dataset205)
export SYNTH_ENS_TRS="nnUNetTrainer_250epochs_NoMirror_FocalTversky__nnUNetPlans__2d,nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed1__nnUNetPlans__2d,nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed2__nnUNetPlans__2d"
export FINE_GRAFT_IDS=8
export FP_FINE_CFG=$NN/t2_ft_ens_fpsuppress_fullres.json
export FP_MERGED_CFG=$NN/t1_topk_fpsuppress_fullres.json
export T3_CKPT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t3/ckpt_segcond_ensmask/best.pt
log "$D docker: CHAMPION stack = 3-seed FT ens + replace{8} graft (FINE/task1) + TopK (MERGED/task2) + seg-cond-ensmask T3"

echo "===== (1) STAGE build context ====="
bash /users/USERNAME/Tiger-SQ-AI/docker/stage_build_context.sh $CTX

echo "===== (2) APPTAINER BUILD (sandbox, then .sif) ====="
cd $CTX
rm -rf $SANDBOX $SIF
# sandbox build is the most reliable on lustre (squashfuse .sif mount can be flaky);
# build sandbox first, run-verify from it, then pack to a portable .sif for submission.
apptainer build --ignore-fakeroot-command --sandbox $SANDBOX $CTX/tiger.def
echo "sandbox built:"; du -sh $SANDBOX

echo "===== (3) RUN container end-to-end on val frames ====="
IN=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/docker_smoke_in;  rm -rf $IN;  mkdir -p $IN
OUT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/docker_smoke_out; rm -rf $OUT; mkdir -p $OUT
SRCIMG=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/synapse/images
for f in center_1_case_14_9 center_1_case_14_8 center_1_case_15_7L center_1_case_15_10R; do
  [ -f $SRCIMG/${f}.png ] && cp $SRCIMG/${f}.png $IN/
done
echo "input frames:"; ls $IN
# pass the chosen trainer dirs into the container env so inference.py loads them
# (FINE = 3-seed ensemble via TIGER_FINE_ENSEMBLE_TRAINERS; MERGED = single TopK)
apptainer exec --nv \
  --env TIGER_FINE_TRAINER=$FINE_TR0 \
  --env TIGER_FINE_ENSEMBLE_TRAINERS=$FINE_ENS_TRS \
  --env TIGER_MERGED_TRAINER=$MERGED_TR \
  --env TIGER_FINE_GRAFT=1 \
  --env TIGER_FINE_GRAFT_IDS=$FINE_GRAFT_IDS \
  --env TIGER_SYNTH_DATASET=Dataset205_TigerT2fineSynthCT \
  --env TIGER_SYNTH_ENSEMBLE_TRAINERS=$SYNTH_ENS_TRS \
  -B $IN:/input -B $OUT:/output $SANDBOX \
  python /opt/tiger/inference.py --input /input --output /output

# also run WITHOUT the graft (TIGER_FINE_GRAFT=0) to a side dir so the verify step can
# prove the FINE/task1 output actually reflects the replace{8} graft (must differ).
OUTNG=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/docker_smoke_out_nograft; rm -rf $OUTNG; mkdir -p $OUTNG
apptainer exec --nv \
  --env TIGER_FINE_TRAINER=$FINE_TR0 \
  --env TIGER_FINE_ENSEMBLE_TRAINERS=$FINE_ENS_TRS \
  --env TIGER_MERGED_TRAINER=$MERGED_TR \
  --env TIGER_FINE_GRAFT=0 \
  -B $IN:/input -B $OUTNG:/output $SANDBOX \
  python /opt/tiger/inference.py --input /input --output /output

echo "===== (4) VERIFY outputs ====="
PASS=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/.tiger_build_pass
rm -f $PASS
SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
$SEG/python - <<PY >> $RES 2>/tmp/verify_err || cat /tmp/verify_err
import sys, csv, datetime, os
import numpy as np
from pathlib import Path
sys.path.insert(0,"/users/USERNAME/Tiger-SQ-AI/challenge_repo")
from metrics.classes import rgb_mask_to_label_mask as fine2i, CLASSES as FINE_CLASSES
from metrics.classes_merged import rgb_mask_to_label_mask as merged2i
from metrics.classes_stations import STATION_NAMES
from PIL import Image
OUT=Path("$OUT"); D=datetime.date.today().isoformat(); JID=os.environ.get("SLURM_JOB_ID","?")
ok=True; msgs=[]
t1=sorted((OUT/"task1").glob("*.png")); t2=sorted((OUT/"task2").glob("*.png"))
unk1=sum(int((fine2i(np.array(Image.open(p).convert("RGB")))==255).sum()) for p in t1)
unk2=sum(int((merged2i(np.array(Image.open(p).convert("RGB")))==255).sum()) for p in t2)
if unk1: ok=False; msgs.append(f"task1 unknown px={unk1}")
if unk2: ok=False; msgs.append(f"task2 unknown px={unk2}")
# task3.csv schema (14 stations)
exp=["case_id"]+list(STATION_NAMES)
rows=list(csv.reader(open(OUT/"task3.csv")))
hdr=rows[0]; nrow=len(rows)-1
if hdr!=exp: ok=False; msgs.append(f"task3 header mismatch")
badp=0
for r in rows[1:]:
  for v in r[1:]:
    fv=float(v)
    if fv<0 or fv>1: badp+=1
if badp: ok=False; msgs.append(f"task3 {badp} probs out of [0,1]")
# (graft proof) task1 (FINE) must differ from the no-graft run -> the replace{8} graft is live
OUTNG=Path("$OUTNG")
graft_px=0
for p in t1:
    ng=OUTNG/"task1"/p.name
    if ng.exists():
        a=np.array(Image.open(p).convert("RGB")); b=np.array(Image.open(ng).convert("RGB"))
        graft_px+=int((a!=b).any(-1).sum())
if graft_px==0:
    ok=False; msgs.append("FINE task1 identical with/without graft (graft NOT live)")
status="BUILD-VERIFIED" if ok else "FAILED"
extra="" if ok else " ["+"; ".join(msgs)+"]"
print(f"[Tiger][{JID}] {D} CHAMPION container run task1={len(t1)} task2={len(t2)} task3_rows={nrow} unk1={unk1} unk2={unk2} graft_diff_px={graft_px} → {status} (submit-ready){extra}")
if ok: Path("$PASS").write_text("ok")
PY

echo "===== (5) GUARDED PROMOTE: pack .sif only if verify PASSED, then promote to live ====="
if [ -f $PASS ]; then
  apptainer build --force $SIF $SANDBOX || echo "(sif pack failed; sandbox still usable)"
  if [ -f $SIF ]; then
    echo "new sif:"; du -sh $SIF
    # tiger_prev.sif is the previous champion (copied by the orchestrator before this run);
    # promote the verified new build to the live tiger.sif submission artifact.
    mv -f $SIF $LIVE_SIF
    echo "promoted -> $LIVE_SIF"; du -sh $LIVE_SIF
    log "$D CHAMPION apptainer .sif packed + promoted to $LIVE_SIF (FINE ens+replace{8}+FP, T3 ensmask) → ready to push to Synapse"
  else
    log "$D BUILD FAILED to pack .sif — live tiger.sif UNCHANGED, tiger_prev.sif safety-net intact"
  fi
else
  log "$D SMOKE VERIFY FAILED — did NOT pack/promote; live tiger.sif UNCHANGED, tiger_prev.sif safety-net intact"
fi
echo "=== ALL DONE build_and_verify ==="
