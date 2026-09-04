#!/bin/bash
#SBATCH --job-name=tiger_gate_v2
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/submission/gate_v2_%j.log
#
# Re-run the contract gate on the ALREADY-BUILT v2 archive (the adopted T2 stack).
# Build 7720071 passed every build assertion with the correct derived stack
#   expected_ckpt_set_derived: 4 ckpts, graft=OFF, fine=Dataset212_TigerT2fineV2
#   graft_off_implies_no_synth_weights::Dataset205: ABSENT as expected
# then reported "do NOT submit" because of
#   Error: OCI runtime error: crun: clone: No space left on device
# which on AIRE means user.max_net_namespaces=0, NOT disk. contract_test.sh now probes for that,
# falls back to --network host with a logged divergence, and classifies it INFRA_BLOCKED (exit 2)
# instead of branding a correct archive unsubmittable.
set -uo pipefail
[ -d /users/USERNAME/Tiger-SQ-AI/docker ] || { echo "[eio-guard] HOME unreadable -> abort"; exit 3; }
export TMPDIR=${TMPDIR:-/tmp}
SUB=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/submission
TARGZ=$SUB/tigersqai_AIMS_task123_v2.tar.gz
IMAGE=localhost/tigersqai_aims:v2
[ -f "$TARGZ" ] || { echo "FATAL: v2 archive missing"; exit 4; }
echo "=== JOB $SLURM_JOB_ID on $(hostname) ==="
echo "  max_net_namespaces = $(cat /proc/sys/user/max_net_namespaces 2>/dev/null)"
echo "  TMPDIR free        = $(df -h $TMPDIR | tail -1 | awk '{print $4}')  <- if large, crun's ENOSPC is the NAMESPACE limit"
echo "  archive            = $(stat -c%s "$TARGZ") bytes  (v1 was 6305959909; smaller = graft weights dropped)"
CIN=$TMPDIR/gin_$SLURM_JOB_ID; COUT=$TMPDIR/gout_$SLURM_JOB_ID; mkdir -p "$CIN" "$COUT"
# two real frames from the v2 data release
SRC=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/data/synapse_v2/images
ls "$SRC"/*.png 2>/dev/null | head -2 | xargs -I{} cp {} "$CIN"/ 2>/dev/null
echo "  input frames: $(ls $CIN | wc -l)"
TARGZ="$TARGZ" IMAGE="$IMAGE" IN="$CIN" OUT="$COUT" GPU=0 ST=$TMPDIR/cgate_$SLURM_JOB_ID \
  bash /users/USERNAME/Tiger-SQ-AI/docker/contract_test.sh
RC=$?
echo "CONTRACT_GATE_RC=$RC"
case $RC in
  0) echo "RESULT: v2 archive CONTRACT GATE PASS -- adopted-T2 archive is submittable";;
  2) echo "RESULT: v2 archive STATIC PASS / DYNAMIC INFRA_BLOCKED on $(hostname) -- NOT a defect";;
  *) echo "RESULT: v2 archive CONTRACT GATE FAILED (rc=$RC) -- genuine defect";;
esac
exit $RC
