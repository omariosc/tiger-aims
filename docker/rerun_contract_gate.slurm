#!/bin/bash
#SBATCH --job-name=tiger_gate_rerun
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/submission/gate_rerun_%j.log
# Re-runs ONLY the CPU contract gate against the ALREADY-BUILT archive, after the
# 2026-07-28 fix to the task3.csv header check (it rejected a correct CRLF header).
set -u
export TMPDIR=${TMPDIR:-/tmp}
ST=$TMPDIR/tigergate_$SLURM_JOB_ID; mkdir -p "$ST"
TAR=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/submission/tigersqai_AIMS_task123.tar.gz
echo "=== archive under test ==="; ls -l "$TAR"; sha256sum "$TAR" | tee "$ST/sha"
diff <(awk '{print $1}' "$ST/sha") <(awk '{print $1}' "${TAR}.sha256") >/dev/null 2>&1 \
  && echo "  sha256 matches the recorded .sha256" || echo "  !! sha256 MISMATCH vs recorded"
# contract_test.sh needs IN=<dir of RGB PNG frames>; build a 2-frame set from the val cases.
CIN=$ST/in; COUT=$ST/out; mkdir -p "$CIN" "$COUT"
SRCF=$(ls /scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/**/imagesTs/*.png \
          /scratch/USERNAME/miccai-2026/Tiger-SQ-AI/selftest_0719/input_native/*.png 2>/dev/null | head -2)
[ -n "$SRCF" ] || { echo "!! no source frames found for the gate"; exit 1; }
for f in $SRCF; do cp -f "$f" "$CIN/"; done
echo "=== gate input frames ==="; ls -l "$CIN"
TARGZ="$TAR" IMAGE=localhost/tigersqai_aims:v1 IN="$CIN" OUT="$COUT" GPU=0 ST="$ST/cgate" \
  bash /users/USERNAME/Tiger-SQ-AI/docker/contract_test.sh
RC=$?
echo "GATE_RERUN_RC=$RC  (0=PASS 1=FAIL 2=static-pass/infra-blocked)"
[ $RC -eq 0 ] && echo ">>> CONTRACT GATE PASS — archive conforms" || echo ">>> still failing, inspect above"
rm -rf "$ST" 2>/dev/null
