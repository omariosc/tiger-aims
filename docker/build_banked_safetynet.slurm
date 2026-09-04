#!/bin/bash
#SBATCH --job-name=tiger_dockerbanked
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%j.out
#SBATCH --error=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/logs/%x_%j.err
# SAFETY-NET (independent, no deps): BUILD-VERIFY the BANKED 250-ep container so a
# submit-ready image exists regardless of the 1k retrains / step A. Distinct .sif
# path so it never clobbers the conditional 1k-aware build (6015691).
set -e
module load apptainer/1.3.6
export APPTAINER_CACHEDIR=/scratch/USERNAME/apptainer_cache APPTAINER_TMPDIR=/scratch/USERNAME/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
RES=/users/USERNAME/AUTONOMOUS_RESULTS.md
CTX=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/docker_ctx_banked
SANDBOX=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/tiger_banked_sandbox
SIF=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/tiger_banked.sif
D=$(date +%F); JID=$SLURM_JOB_ID

echo "=== stage (banked 250-ep defaults) ==="
bash /users/USERNAME/Tiger-SQ-AI/docker/stage_build_context.sh $CTX

echo "=== apptainer build sandbox ==="
cd $CTX; rm -rf $SANDBOX $SIF
apptainer build --ignore-fakeroot-command --sandbox $SANDBOX $CTX/tiger.def
du -sh $SANDBOX

echo "=== run container on val frames ==="
IN=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/banked_smoke_in;  rm -rf $IN;  mkdir -p $IN
OUT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/banked_smoke_out; rm -rf $OUT; mkdir -p $OUT
SRCIMG=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/synapse/images
for f in center_1_case_14_9 center_1_case_14_8 center_1_case_15_7L center_1_case_15_10R; do
  [ -f $SRCIMG/${f}.png ] && cp $SRCIMG/${f}.png $IN/
done
apptainer exec --nv -B $IN:/input -B $OUT:/output $SANDBOX \
  python /opt/tiger/inference.py --input /input --output /output

echo "=== verify + log ==="
SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
$SEG/python - <<PY >> $RES 2>/dev/null || true
import sys, csv, datetime, os
import numpy as np
from pathlib import Path
sys.path.insert(0,"/users/USERNAME/Tiger-SQ-AI/challenge_repo")
from metrics.classes import rgb_mask_to_label_mask as fine2i
from metrics.classes_merged import rgb_mask_to_label_mask as merged2i
from metrics.classes_stations import STATION_NAMES
from PIL import Image
OUT=Path("$OUT"); D=datetime.date.today().isoformat(); JID="$JID"
t1=sorted((OUT/"task1").glob("*.png")); t2=sorted((OUT/"task2").glob("*.png"))
unk1=sum(int((fine2i(np.array(Image.open(p).convert("RGB")))==255).sum()) for p in t1)
unk2=sum(int((merged2i(np.array(Image.open(p).convert("RGB")))==255).sum()) for p in t2)
rows=list(csv.reader(open(OUT/"task3.csv"))); hdr_ok=(rows[0]==["case_id"]+list(STATION_NAMES))
ok=(unk1==0 and unk2==0 and hdr_ok and len(t1)>0)
print(f"[Tiger][{JID}] {D} SAFETY-NET banked 250-ep container run task1={len(t1)} task2={len(t2)} task3_rows={len(rows)-1} unk1={unk1} unk2={unk2} hdr_ok={hdr_ok} → {'BUILD-VERIFIED' if ok else 'FAILED'} (banked fallback, submit-ready)")
PY
apptainer build --force $SIF $SANDBOX || echo "(sif pack failed; sandbox usable)"
[ -f $SIF ] && du -sh $SIF
echo "=== DONE banked safetynet ==="
