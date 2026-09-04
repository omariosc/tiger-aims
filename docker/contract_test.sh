#!/bin/bash
# ★ STANDING PRE-SUBMISSION GATE for TIGER SQ-AI 2026.
#
# Runs the image with the EXACT command published in the submission instructions §6
# ("Run exactly the command we run. If it does not work this way on your machine, it
#  will not work on ours."):
#
#   docker run --rm --gpus all --network none --shm-size 8g \
#     -v /path/to/sanity_set:/input:ro -v $(pwd)/results:/output tigersqai_myteam
#
# ...then asserts the output tree is EXACTLY task1/ and/or task2/ and/or task3.csv
# "and nothing else" (contract §6), and that the container is readable by a non-root uid.
#
# WHY THE PERMISSION AUDIT IS HERE EVEN THOUGH TIGER HAS NO `--user`:
#   CLiMB submissions v1 (9772524) and v2 (9773586) were both INVALID because /app was
#   root-only-readable (COPY preserves the 0600 source bits produced by our 077 umask)
#   and that challenge's evaluator runs `--user "$(id -u)":"$(id -g)"` -> the container
#   could not read its own entrypoint. TIGER's published command contains NO --user
#   (verified by grep over the submission instructions, metrics/00_README.md and the
#   repo README: zero hits for --user / id -u / non-root), so TIGER is NOT exposed.
#   The check costs nothing and removes the class entirely, so we keep it.
#
# NOTE: apptainer is NOT a substitute for this test. Converting an archive to a SIF
# extracts every layer as the INVOKING user, so a 0600 file owned by us is readable by
# us — which is exactly how the CLiMB defect stayed invisible for two submissions.
#
# Usage:
#   TARGZ=... IMAGE=... IN=... [GPU=1] [ST=...] bash contract_test.sh
# Exit codes:
#   0 = PASS   1 = FAIL (do not submit)   2 = STATIC PASS / dynamic step infra-blocked
set -u

TARGZ=${TARGZ:-/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/submission/tigersqai_AIMS_task123.tar.gz}
IMAGE=${IMAGE:-localhost/tigersqai_aims:v1}
IN=${IN:?set IN=<dir of RGB PNG frames>}
OUT=${OUT:-}
GPU=${GPU:-0}
ST=${ST:-/tmp/tigercontract_$$}
KEEP=${KEEP:-0}

mkdir -p "$ST/root" "$ST/runroot"
[ -n "$OUT" ] || OUT=$ST/output
rm -rf "$OUT"; mkdir -p "$OUT"
P="podman --root $ST/root --runroot $ST/runroot --storage-driver=vfs --storage-opt vfs.ignore_chown_errors=true"

echo "############################################################"
echo "=== TIGER contract gate ==="
echo "  archive : $TARGZ ($(du -h "$TARGZ" 2>/dev/null | cut -f1))"
echo "  image   : $IMAGE"
echo "  input   : $IN  ($(ls -1 "$IN"/*.png 2>/dev/null | wc -l) frames)"
echo "  output  : $OUT"

echo "=== load the docker-archive (the artifact we would actually upload) ==="
$P load -i "$TARGZ" 2>&1 | tail -2 || { echo "!! podman load FAILED"; exit 1; }
$P images 2>&1 | head -5

# ---------------------------------------------------------------------------
# STATIC PERMISSION AUDIT — uid-mapping-independent, always runs.
# Instrument self-check FIRST, in BOTH directions: it must FLAG a planted 0600 file
# and a planted non-traversable 0744 dir, and must SPARE a good 0644 file / 0755 dir.
# A check that silently never runs would otherwise report "clean" for a broken image
# (cf. the cuobjdump cascade that once produced a false "MISSING sm_120").
# ---------------------------------------------------------------------------
echo "=== STATIC /opt/tiger PERMISSION AUDIT ==="
CTL=$(mktemp -d); mkdir -p "$CTL/bad_dir" "$CTL/good_dir"; : > "$CTL/bad_file"; : > "$CTL/good_file"
chmod 755 "$CTL"; chmod 600 "$CTL/bad_file"; chmod 644 "$CTL/good_file"
chmod 744 "$CTL/bad_dir"; chmod 755 "$CTL/good_dir"
CTL_F=$(find "$CTL" -mindepth 1 -type f ! -perm -o+r)
CTL_D=$(find "$CTL" -mindepth 1 -type d ! -perm -o+x)
NF=$(printf '%s\n' "$CTL_F" | grep -c . ); ND=$(printf '%s\n' "$CTL_D" | grep -c .)
echo "  control: flagged $NF file(s) [want 1: bad_file] and $ND dir(s) [want 1: bad_dir]"
if [ "$NF" -ne 1 ] || [ "$ND" -ne 1 ] || ! echo "$CTL_F" | grep -q bad_file || ! echo "$CTL_D" | grep -q bad_dir; then
  echo "!! AUDIT INSTRUMENT BROKEN — it does not flag exactly the planted bad modes."
  rm -rf "$CTL"; echo "=== CONTRACT TEST FAILED (untrustworthy instrument) ==="; exit 1
fi
chmod 644 "$CTL/bad_file"; chmod 755 "$CTL/bad_dir"
if [ -n "$(find "$CTL" -mindepth 1 -type f ! -perm -o+r)" ] || [ -n "$(find "$CTL" -mindepth 1 -type d ! -perm -o+x)" ]; then
  echo "!! AUDIT INSTRUMENT BROKEN — it still flags a REPAIRED tree."
  rm -rf "$CTL"; echo "=== CONTRACT TEST FAILED (untrustworthy instrument) ==="; exit 1
fi
rm -rf "$CTL"
echo "  instrument verified in BOTH directions (flags 0600 file + 0744 dir, clears both once fixed)"

AUDIT=$($P run --rm --entrypoint "" "$IMAGE" sh -c \
  'find /opt/tiger ! -perm -o+r -printf "NOT-READABLE %M %p\n" 2>/dev/null;
   find /opt/tiger -type d ! -perm -o+x -printf "NOT-TRAVERSABLE %M %p\n" 2>/dev/null;
   echo "AUDITED $(find /opt/tiger | wc -l) paths under /opt/tiger"' 2>/dev/null)
echo "$AUDIT" | tail -3
if echo "$AUDIT" | grep -qE "^NOT-(READABLE|TRAVERSABLE)"; then
  echo "$AUDIT" | grep -E "^NOT-" | head -20
  echo "!! STATIC AUDIT FAILED -> a non-root uid could not read /opt/tiger."
  echo "   Fix the modes in the BUILD CONTEXT host-side (chmod -R a+rX) — not with a"
  echo "   blanket in-image chmod -R, which forces an overlayfs copy-up of the weights."
  echo "=== CONTRACT TEST FAILED ==="; exit 1
fi
echo "  PASS: every /opt/tiger path world-readable, every dir world-traversable"

# ---------------------------------------------------------------------------
# THE EXACT CONTRACT COMMAND (§6). Note: NO --user (TIGER does not pass one) and NO
# arguments after the image name -> this also proves ENTRYPOINT+CMD are self-sufficient.
# ---------------------------------------------------------------------------
GPUFLAGS=""
if [ "$GPU" = 1 ]; then GPUFLAGS="--gpus all"; fi
echo "=== EXACT CONTRACT COMMAND ==="
echo "docker run --rm ${GPUFLAGS:---gpus all (omitted: CPU-mode gate)} --network none --shm-size 8g \\"
echo "  -v $IN:/input:ro -v $OUT:/output $IMAGE"
# NETNS PROBE (added 2026-09-03, mirroring the CLiMB gate). Many AIRE nodes run with
# user.max_net_namespaces=0, where podman/crun fails with the MISLEADING
# "clone: No space left on device" -- it is a NAMESPACE limit, not disk (the node that hit it had
# 9.8 TB free on /tmp). Without this, an infrastructure limit is reported as "do NOT submit this
# archive", which is how build 7720071 branded a CORRECT archive unsubmittable.
NETNS="--network none"
if ! $P run --rm --network none --entrypoint "" "$IMAGE" /bin/true >/dev/null 2>&1; then
  if [ "$(cat /proc/sys/user/max_net_namespaces 2>/dev/null || echo -1)" = "0" ]; then
    NETNS="--network host"
    echo "!! DOCUMENTED DIVERGENCE: user.max_net_namespaces=0 on $(hostname) -> --network none is"
    echo "   impossible here; running with --network host. Network ISOLATION is therefore NOT"
    echo "   exercised by this run and must be argued separately (no runtime downloads, offline env)."
  fi
fi
RUNLOG=$ST/run.log
set -x
$P run --rm $GPUFLAGS $NETNS --shm-size 8g \
  -v "$IN":/input:ro -v "$OUT":/output "$IMAGE" 2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
set +x
echo "CONTRACT_RUN_RC=$RC"

# GPU passthrough on this cluster is CDI-flaky; retry with the explicit CDI device.
if [ "$GPU" = 1 ] && [ $RC -ne 0 ] && grep -qiE "CDI|nvidia.com/gpu|--gpus" "$RUNLOG"; then
  echo "=== --gpus all failed (CDI); retrying with --device nvidia.com/gpu=all ==="
  rm -rf "$OUT"; mkdir -p "$OUT"
  $P run --rm --device nvidia.com/gpu=all $NETNS --shm-size 8g \
    -v "$IN":/input:ro -v "$OUT":/output "$IMAGE" 2>&1 | tee "$RUNLOG"
  RC=${PIPESTATUS[0]}
  echo "CONTRACT_RUN_RC(cdi)=$RC"
fi

# Distinguish "this HOST cannot run the test" from "the IMAGE failed the test".
if grep -qE "setres(uid|gid) to .*Invalid argument|no subuid ranges found|unresolvable CDI devices|clone: No space left on device" "$RUNLOG" 2>/dev/null; then
  echo "== INFRA_BLOCKED: this host could not run the container (uid mapping / CDI passthrough /"
  echo "   namespace limit -- note 'No space left on device' from crun is NOT disk; check df)."
  echo "   The STATIC AUDIT above verified the permission property; the run itself is"
  echo "   not executable here. This is an infrastructure limit, NOT an image defect."
  [ "$KEEP" = 1 ] || { podman unshare rm -rf "$ST" 2>/dev/null; rm -rf "$ST" 2>/dev/null; }
  echo "=== CONTRACT TEST: STATIC PASS / DYNAMIC INFRA_BLOCKED ==="; exit 2
fi
if grep -q "Permission denied" "$RUNLOG" 2>/dev/null; then
  echo "!! 'Permission denied' in the container log -> the CLiMB v1/v2 defect class. DO NOT SUBMIT."
  echo "=== CONTRACT TEST FAILED ==="; exit 1
fi

# ---------------------------------------------------------------------------
# OUTPUT TREE: contract §6 — "task1/ and/or task2/ and/or task3.csv ... and nothing else"
# ---------------------------------------------------------------------------
echo "=== OUTPUT TREE (must be EXACTLY task1/ task2/ task3.csv) ==="
TOP=$(cd "$OUT" && ls -A)
echo "$TOP" | sed 's/^/  /'
EXTRA=$(cd "$OUT" && ls -A | grep -vxE 'task1|task2|task3\.csv' || true)
N1=$(ls -1 "$OUT/task1"/*.png 2>/dev/null | wc -l)
N2=$(ls -1 "$OUT/task2"/*.png 2>/dev/null | wc -l)
N3=$(( $(wc -l < "$OUT/task3.csv" 2>/dev/null || echo 1) - 1 ))
NIN=$(ls -1 "$IN"/*.png 2>/dev/null | wc -l)
# non-PNG strays inside the task folders count as extras too
STRAY=$( { find "$OUT/task1" "$OUT/task2" -mindepth 1 ! -name '*.png' 2>/dev/null; } | head -5 )
echo "  task1 pngs=$N1  task2 pngs=$N2  task3 rows=$N3  (input frames=$NIN)"
TREE_OK=1
[ -z "$EXTRA" ]  || { echo "!! EXTRA top-level entries under /output: $EXTRA"; TREE_OK=0; }
[ -z "$STRAY" ]  || { echo "!! non-PNG files inside task folders: $STRAY"; TREE_OK=0; }
[ "$N1" = "$NIN" ] || { echo "!! task1 png count != input frame count"; TREE_OK=0; }
[ "$N2" = "$NIN" ] || { echo "!! task2 png count != input frame count"; TREE_OK=0; }
[ "$N3" = "$NIN" ] || { echo "!! task3.csv row count != input frame count"; TREE_OK=0; }
# NOTE 2026-07-28: strip a trailing CR before comparing. Python's csv.writer defaults to
# lineterminator='\r\n' (RFC 4180), so the header legitimately ends CRLF; the scorer's
# pd.read_csv handles it transparently (verified against metrics/evaluate_cls.py's exact
# code path: columns match the registry, index_col='case_id' resolves). The previous
# `grep -qx` failed on the trailing \r and reported a CORRECT header as wrong.
head -1 "$OUT/task3.csv" 2>/dev/null | tr -d '\r' | grep -qx 'case_id,6L,6R,7L,7R,8,9,10L,10R,11L,11R,12L,12R,13L,13R' \
  || { echo "!! task3.csv header is not the exact contract header"; TREE_OK=0; }
[ "$TREE_OK" = 1 ] && echo "  PASS: output tree matches the contract exactly"

[ "$KEEP" = 1 ] || { podman unshare rm -rf "$ST/root" "$ST/runroot" 2>/dev/null; }
if [ $RC -eq 0 ] && [ "$TREE_OK" = 1 ]; then
  echo "=== CONTRACT TEST PASSED ==="; exit 0
fi
echo "=== CONTRACT TEST FAILED (RC=$RC TREE_OK=$TREE_OK) ==="; exit 1
