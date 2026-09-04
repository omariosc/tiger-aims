#!/bin/bash
#SBATCH --job-name=tiger_buildv2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/selftest_0719/buildv2_%j.out
set -e
# TIGER SQ-AI 2026 — REBUILD the submission container with the task1<->task2 FLIP FIX
# (2026-07-19) and verify the ACTUAL .sif entrypoint on native + novel-resolution frames.
# Mirrors the proven build_and_verify.slurm recipe (module apptainer + sandbox build + exec).
# Build context = docker_ctx_v2 (fixed inference.py + updated challenge_repo, pre-staged).
module load apptainer/1.3.6
export APPTAINER_CACHEDIR=/scratch/USERNAME/apptainer_cache
export APPTAINER_TMPDIR=/scratch/USERNAME/apptainer_tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
ROOT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI
CTX=$ROOT/docker_ctx_v2
BASE=$ROOT/selftest_0719
SANDBOX=$ROOT/tiger_sandbox_v2
SIF=$ROOT/tiger_v2.sif
PY=/scratch/USERNAME/miccai-2026/seg_env/bin/python
REPO=/users/USERNAME/Tiger-SQ-AI/challenge_repo
export MKL_THREADING_LAYER=GNU

echo "############ (1) apptainer sandbox build (fixed context) ############"
cd $CTX
rm -rf $SANDBOX $SIF
apptainer build --ignore-fakeroot-command --sandbox $SANDBOX $CTX/tiger.def
echo "sandbox built:"; du -sh $SANDBOX

echo "############ (2) run the ACTUAL container on 6 native + 6 novel-res frames ############"
rm -rf $BASE/sif_in_native $BASE/sif_out_native $BASE/sif_out_varres
mkdir -p $BASE/sif_in_native $BASE/sif_out_native $BASE/sif_out_varres
ls $BASE/input_native/*.png | head -6 | xargs -I{} cp {} $BASE/sif_in_native/
echo "--- container NATIVE (champion defaults baked in inference.py) ---"
apptainer exec --nv -B $BASE/sif_in_native:/input -B $BASE/sif_out_native:/output $SANDBOX \
    python /opt/tiger/inference.py --input /input --output /output
echo "--- container VARRES (novel resolutions) ---"
apptainer exec --nv -B $BASE/input_varres:/input -B $BASE/sif_out_varres:/output $SANDBOX \
    python /opt/tiger/inference.py --input /input --output /output

echo "############ (2b) T-1 ADVERSARIAL FILENAME TEST *INSIDE* THE REAL CONTAINER ############"
# Pre-registered KILL: the strict station parse must FAIL-CLOSED on deliberately renamed
# files and the seg-conditioned fallback must demonstrably fire (announced per frame).
rm -rf $BASE/adv_in $BASE/adv_out; mkdir -p $BASE/adv_in $BASE/adv_out
i=0
for f in $(ls $BASE/input_native/*.png | head -6); do
  i=$((i+1)); n=$(basename $f .png)
  if [ $i -le 3 ]; then cp $f $BASE/adv_in/$n.png
  elif [ $i -eq 4 ]; then cp $f "$BASE/adv_in/frame_0004.png"
  elif [ $i -eq 5 ]; then cp $f "$BASE/adv_in/${n}_copy.png"
  else cp $f "$BASE/adv_in/$(echo $n | tr 'A-Z' 'a-z').png"; fi
done
echo "adversarial inputs:"; ls $BASE/adv_in
apptainer exec --nv -B $BASE/adv_in:/input -B $BASE/adv_out:/output $SANDBOX \
    python /opt/tiger/inference.py --input /input --output /output 2>&1 | tee $BASE/adv_run.log | grep -E "T3"
NPRIOR=$(grep -c "PRIOR+SEG\[" $BASE/adv_run.log || true)
NFALL=$(grep -c "SEG-ONLY(fallback)" $BASE/adv_run.log || true)
echo ">>> in-container: prior path on $NPRIOR frames, fallback on $NFALL frames (expect 3 and 3)"
if [ "$NPRIOR" = "3" ] && [ "$NFALL" = "3" ]; then echo ">>> T-1 ADVERSARIAL TEST: PASS (fail-closed + fallback fires)"; ADV_OK=1
else echo ">>> T-1 ADVERSARIAL TEST: FAIL — delivery path KILLED"; ADV_OK=0; fi

echo "############ (3) verify the container output (routing / size / palette / task3) ############"
$PY /users/USERNAME/Tiger-SQ-AI/experiments/selftest_verify.py \
    --repo $REPO --gt-root $ROOT/data/synapse \
    --native-out $BASE/sif_out_native --native-in $BASE/sif_in_native \
    --varres-out $BASE/sif_out_varres --varres-in $BASE/input_varres \
    --verdict $BASE/sif_verdict.json && SIF_OK=1 || SIF_OK=0

echo "############ (4) pack portable tiger_v2.sif IF verify passed AND adversarial test passed ############"
if [ "$SIF_OK" = "1" ] && [ "$ADV_OK" = "1" ]; then
  apptainer build --force $SIF $SANDBOX && echo "packed: $(du -sh $SIF)" || echo "(sif pack failed; sandbox usable)"
  echo ">>> tiger_v2.sif is the FIXED submission artifact — supersedes tiger.sif (which had the task1/2 FLIP)."
  echo ">>> Promote with:  mv -f $SIF $ROOT/tiger.sif   (after human review)"
else
  echo ">>> VERIFY FAILED — did NOT pack .sif. Inspect $BASE/sif_verdict.json"
fi
echo "############ BUILD+VERIFY DONE (SIF_OK=$SIF_OK) ############"
