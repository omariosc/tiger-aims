#!/bin/bash
#SBATCH --job-name=tiger_dockerarchive
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/submission/build_%j.log
# =============================================================================
# TIGER SQ-AI 2026 — close the PACKAGING GAP (contract §7).
#
# The contract wants a DOCKER-ARCHIVE:
#     docker save <image> | gzip -c > tigersqai_<team>_task<list>.tar.gz
# Our verified artifact was an APPTAINER .sif, which cannot be `docker save`d.
# This job rebuilds the SAME build context with rootless buildah/podman and emits
#     tigersqai_AIMS_task123.tar.gz     (we enter task1 + task2 + task3)
#
# NO GPU IS NEEDED TO BUILD. Routed to the `nodes` partition on purpose: a CLiMB
# container build finished there in 32 min while its GPU twin was still pending
# behind a 76-deep queue. The GPU-dependent numeric verification is a separate job.
#
# Proven on-cluster rootless recipe (iMED/MB/CLiMB):
#   buildah unshare | --isolation=oci | --network=host | vfs + ignore_chown_errors
#   (no /etc/subuid range for this user) | FULLY-QUALIFIED docker.io/... base
#   (short-name resolution is enforced and cannot prompt without a TTY)
#   | --layers=false so vfs holds ONE working copy, not one per instruction.
#
# Model / weights / inference logic are NOT touched. This is pure delivery.
# =============================================================================
set +e
ROOT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI
SRC=/users/USERNAME/Tiger-SQ-AI/docker
CTX_IN=$ROOT/docker_ctx_v2          # the VERIFIED context (tiger_v2.sif provenance)
CTX=$ROOT/docker_ctx_v3             # mode-fixed copy we actually build from
SUB=$ROOT/submission
TEAM=AIMS
TASKS=123
IMAGE=localhost/tigersqai_aims:v1
TARGZ=$SUB/tigersqai_${TEAM}_task${TASKS}.tar.gz
PY=/scratch/USERNAME/miccai-2026/seg_env/bin/python
mkdir -p "$SUB"

echo "=== JOB $SLURM_JOB_ID on $(hostname) $(date -Is) ==="
export TMPDIR=${TMPDIR:-/tmp}
ST=$TMPDIR/tigerbuild_$SLURM_JOB_ID; mkdir -p "$ST/root" "$ST/runroot"
df -h "$TMPDIR" | tail -1

# ---------------------------------------------------------------------------
# (1) STAGE the build context, with the mode defect fixed HOST-SIDE.
#     Our umask is 077, so every staged file is 0600 and `COPY` preserves that.
#     Fixing it here (rather than with an in-image `chmod -R`) avoids an overlayfs
#     copy-up of the 2.8 GB weight tree (ORENA: 17.7 GB -> 35 GB from exactly that).
# ---------------------------------------------------------------------------
echo "############ (1) stage mode-fixed build context ############"
rm -rf "$CTX"
rsync -a "$CTX_IN"/ "$CTX"/ || { echo "FATAL: rsync of $CTX_IN failed"; exit 1; }
# refresh the recipe + build-time assertions from the repo (weights come from CTX_IN)
cp -f "$SRC/Dockerfile" "$SRC/build_checks.py" "$CTX/" || { echo "FATAL: cp recipe"; exit 1; }
# assert the shipped SOURCES are identical to the repo copies (no silent drift)
for f in inference.py station_boundary_map.py covis_matrix.json; do
  if ! cmp -s "$SRC/$f" "$CTX/$f"; then echo "FATAL: $f differs between $SRC and the context"; exit 1; fi
done
cmp -s /users/USERNAME/Tiger-SQ-AI/experiments/fp_suppress.py "$CTX/fp_suppress.py" \
  || { echo "FATAL: fp_suppress.py differs from experiments/"; exit 1; }
chmod -R a+rX "$CTX"
BAD=$(find "$CTX" ! -perm -o+r | wc -l); BADD=$(find "$CTX" -type d ! -perm -o+x | wc -l)
echo "context modes: $BAD not-world-readable, $BADD dirs not-traversable (want 0/0)"
[ "$BAD" -eq 0 ] && [ "$BADD" -eq 0 ] || { echo "FATAL: context modes not fixed"; exit 1; }
echo "context: $(du -sh "$CTX" | cut -f1), $(find "$CTX" -type f | wc -l) files"
ls -l "$CTX" | head -20

# ---------------------------------------------------------------------------
# (2) BUILD (rootless buildah, no apt anywhere -> dodges the dpkg/setgroups wall)
# ---------------------------------------------------------------------------
echo "############ (2) buildah bud -> $IMAGE ############"
STORE="--root $ST/root --runroot $ST/runroot --storage-driver=vfs"
OPTS="--isolation=oci --network=host --layers=false --storage-opt vfs.ignore_chown_errors=true"
SENTINEL=$ST/status; rm -f "$SENTINEL"
cat > "$ST/inner.sh" <<INNER
set -x
cd "$CTX"
buildah $STORE bud $OPTS -t "$IMAGE" -f Dockerfile .
rc=\$?; echo "INNER_BUILD_RC=\$rc"
if [ \$rc -eq 0 ] && buildah $STORE inspect --type image "$IMAGE" >/dev/null 2>&1; then
  echo OK > "$SENTINEL"
else
  echo FAIL > "$SENTINEL"
fi
INNER
time buildah unshare bash "$ST/inner.sh"
if [ "$(cat "$SENTINEL" 2>/dev/null)" != OK ]; then
  echo "FATAL: image build failed (sentinel=$(cat "$SENTINEL" 2>/dev/null))"; exit 1
fi
echo "=== build OK ==="
buildah unshare podman $STORE images 2>&1 | head -5

# ---------------------------------------------------------------------------
# (3) THE CONTRACT §7 COMMAND, verbatim in shape:  docker save <image> | gzip -c > f
#     (podman save defaults to --format docker-archive; stated explicitly.)
# ---------------------------------------------------------------------------
echo "############ (3) docker save | gzip -> $TARGZ ############"
rm -f "$TARGZ" "$TARGZ.tmp"
GZ=gzip; command -v pigz >/dev/null && GZ="pigz -p ${SLURM_CPUS_PER_TASK:-8}"
buildah unshare bash -c \
  "podman $STORE save --format docker-archive '$IMAGE' | $GZ -c > '$TARGZ.tmp'"
SAVE_RC=$?
echo "SAVE_RC=$SAVE_RC"
if [ $SAVE_RC -ne 0 ] || [ ! -s "$TARGZ.tmp" ]; then
  echo "FATAL: docker save|gzip produced nothing"; exit 1
fi
mv -f "$TARGZ.tmp" "$TARGZ"
# Lesson 4.11 (CLiMB nearly pushed a stale tar): print the artifact's identity.
echo "--- ARTIFACT ---"
ls -l --time-style=full-iso "$TARGZ"
echo "bytes          : $(stat -c%s "$TARGZ")"
echo "human          : $(du -h "$TARGZ" | cut -f1)"
echo -n "sha256         : "; sha256sum "$TARGZ" | cut -d' ' -f1
sha256sum "$TARGZ" > "$TARGZ.sha256"
echo "--- image size in store ---"
buildah unshare podman $STORE images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>&1 | head -3
# prove it really is a docker-archive (manifest.json at the tar root, not an OCI dir)
echo "--- archive manifest ---"
$GZ -dc "$TARGZ" 2>/dev/null | tar -xOf - manifest.json 2>/dev/null | head -c 500; echo

# ---------------------------------------------------------------------------
# (4) CONTRACT GATE, CPU MODE — the exact published command (minus --gpus, which
#     needs a GPU node), run against the ARCHIVE WE WOULD UPLOAD, on 2 frames
#     down-scaled to a resolution the model has never seen (the contract warns
#     resolutions differ between centres). Proves: the archive loads, ENTRYPOINT+CMD
#     are self-sufficient with NO arguments, /opt/tiger is world-readable, and the
#     output tree is exactly task1/ task2/ task3.csv and nothing else.
# ---------------------------------------------------------------------------
echo "############ (4) CPU-mode contract gate on the ARCHIVE ############"
CIN=$ST/cin; COUT=$ST/cout; rm -rf "$CIN" "$COUT"; mkdir -p "$CIN"
i=0
for f in $(ls "$ROOT"/data/synapse/images/center_1_case_15_*.png 2>/dev/null | head -2); do
  i=$((i+1))
  $PY - "$f" "$CIN/$(basename "$f")" <<'PYEOF'
import sys
from PIL import Image
im = Image.open(sys.argv[1]).convert("RGB")
w, h = im.size
s = 640.0 / max(w, h)
im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.BILINEAR).save(sys.argv[2])
PYEOF
done
echo "CPU-gate input: $(ls -1 "$CIN" | wc -l) frames"; $PY -c "
import glob,sys
from PIL import Image
for p in sorted(glob.glob('$CIN/*.png')): print('  ', p.split('/')[-1], Image.open(p).size)"

TARGZ="$TARGZ" IMAGE="$IMAGE" IN="$CIN" OUT="$COUT" GPU=0 ST=$ST/cgate \
  bash "$SRC/contract_test.sh"
CPU_GATE_RC=$?
echo "CPU_CONTRACT_GATE_RC=$CPU_GATE_RC  (0=PASS 1=FAIL 2=static-pass/infra-blocked)"

# keep the CPU-gate output tree for inspection
cp -r "$COUT" "$SUB/cpu_gate_out_$SLURM_JOB_ID" 2>/dev/null

echo "############ SUMMARY ############"
echo "archive     : $TARGZ"
echo "sha256      : $(cut -d' ' -f1 "$TARGZ.sha256")"
echo "size        : $(du -h "$TARGZ" | cut -f1)"
echo "cpu gate rc : $CPU_GATE_RC"
if [ "$CPU_GATE_RC" = 0 ]; then
  echo ">>> PACKAGING GAP CLOSED: docker-archive built + contract command verified (CPU)."
  echo ">>> Numeric/GPU verification = the companion job (verify_docker_archive_gpu.slurm)."
else
  echo ">>> !! CPU CONTRACT GATE DID NOT PASS — do NOT submit this archive as-is."
fi
rm -rf "$ST/root" "$ST/runroot" 2>/dev/null
echo "=== DONE $(date -Is) ==="
