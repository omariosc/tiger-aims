#!/bin/bash
#SBATCH --job-name=tiger_archverify
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/submission/verify_%j.log
# =============================================================================
# TIGER SQ-AI 2026 — verify the SUBMISSION ARCHIVE with the EXACT contract command.
#
# Contract §6, verbatim: "Run exactly the command we run. If it does not work this
# way on your machine, it will not work on ours."
#
#   docker run --rm --gpus all --network none --shm-size 8g \
#     -v /path/to/sanity_set:/input:ro -v $(pwd)/results:/output tigersqai_myteam
#
# Note there is NO `--user` in the TIGER command (verified by grep over the
# submission instructions, metrics/00_README.md and the repo README: zero hits for
# `--user`, `id -u`, `non-root`) -> TIGER is NOT exposed to the defect that made two
# CLiMB submissions INVALID. The permission property is asserted anyway.
#
# CRITICAL DIFFERENCE FROM EVERY PREVIOUS TIGER CONTAINER TEST: the earlier apptainer
# checks invoked `python /opt/tiger/inference.py --input ... --output ...` explicitly,
# which BYPASSES ENTRYPOINT/CMD. The organizers pass NO arguments. This job runs the
# image bare, so the entrypoint itself is under test.
#
# Model/weights/inference logic untouched — pure delivery verification.
# =============================================================================
set +e
ROOT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI
SRC=/users/USERNAME/Tiger-SQ-AI/docker
EXP=/users/USERNAME/Tiger-SQ-AI/experiments
REPO=/users/USERNAME/Tiger-SQ-AI/challenge_repo
SUB=$ROOT/submission
TEAM=AIMS
IMAGE=localhost/tigersqai_aims:v1
TARGZ=${TARGZ:-$SUB/tigersqai_${TEAM}_task123.tar.gz}
PY=/scratch/USERNAME/miccai-2026/seg_env/bin/python
BASE=$SUB/gpu_verify_$SLURM_JOB_ID
mkdir -p "$BASE"

echo "=== JOB $SLURM_JOB_ID on $(hostname) $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# Lesson 4.11: assert we are verifying the artifact we think we are.
[ -s "$TARGZ" ] || { echo "FATAL: archive missing: $TARGZ"; exit 1; }
echo "--- ARTIFACT UNDER TEST ---"; ls -l --time-style=full-iso "$TARGZ"
echo -n "sha256: "; sha256sum "$TARGZ" | cut -d' ' -f1
if [ -f "$TARGZ.sha256" ]; then
  sha256sum -c "$TARGZ.sha256" || { echo "FATAL: archive digest does not match the build's record"; exit 1; }
fi

# ---------------------------------------------------------------------------
# (0) RE-VERIFY THE task1/task2 ROUTING AGAINST THE CURRENT UPSTREAM REPO.
#     The organizers already flipped task1<->task2 once after our June clone
#     (task1=MERGED, task2=FINE). A spec can drift twice, so re-check rather than
#     trust the dossier.
# ---------------------------------------------------------------------------
echo "############ (0) task1/task2 routing vs CURRENT origin/main ############"
cd "$REPO"
LOCAL=$(git rev-parse HEAD)
UPSTREAM=$(timeout 120 git ls-remote origin HEAD 2>/dev/null | awk '{print $1}')
echo "local clone HEAD : $LOCAL"
echo "origin/main HEAD : ${UPSTREAM:-<unreachable>}"
if [ -n "$UPSTREAM" ] && [ "$UPSTREAM" != "$LOCAL" ]; then
  echo "!! ⚠ UPSTREAM HAS MOVED — the spec may have drifted AGAIN. Diffing:"
  timeout 300 git fetch origin 2>&1 | tail -3
  git diff --stat "$LOCAL..$UPSTREAM" -- metrics/ TIGER_SQ-AI_2026_submission_instructions.md 2>&1 | tail -20
  ROUTING_FRESH=0
else
  echo "   clone is current with origin/main"
  ROUTING_FRESH=1
fi
echo "--- what the CURRENT evaluator scores each task with ---"
grep -n 'gt_root / "task1"' -A3 metrics/01_evaluate_challenge.py | head -8
grep -n 'gt_root / "task2"' -A2 metrics/01_evaluate_challenge.py | head -6
echo "--- what OUR container writes ---"
grep -n 'ids_to_rgb_png(.*task[12]_dir' "$SRC/inference.py"
echo "   (expect: task1 <- MERGED_LUT, task2 <- FINE_LUT)"

# ---------------------------------------------------------------------------
# (1) EXACT CONTRACT COMMAND on the full held-out val set (28 native frames,
#     cases 14+15) — loaded from the ARCHIVE, run bare (no arguments).
# ---------------------------------------------------------------------------
echo "############ (1) EXACT contract command, 28 native held-out frames ############"
IN=$BASE/input; OUT=$BASE/results; rm -rf "$IN" "$OUT"; mkdir -p "$IN"
for c in 14 15; do cp "$ROOT"/data/synapse/images/center_1_case_${c}_*.png "$IN"/ 2>/dev/null; done
NIN=$(ls -1 "$IN"/*.png | wc -l); echo "input frames: $NIN"
T0=$(date +%s)
TARGZ="$TARGZ" IMAGE="$IMAGE" IN="$IN" OUT="$OUT" GPU=1 ST=/tmp/tgate_$SLURM_JOB_ID KEEP=1 \
  bash "$SRC/contract_test.sh"
GATE_RC=$?
T1=$(date +%s); ELAPSED=$((T1-T0))
echo "CONTRACT_GATE_RC=$GATE_RC   wall=${ELAPSED}s"

# ---- runtime budget: 1 min/frame/task; a 3-task image gets 3 min/frame (420 min/140)
if [ "$NIN" -gt 0 ]; then
  PERFRAME=$(awk "BEGIN{printf \"%.1f\", $ELAPSED/$NIN}")
  EST140=$(awk "BEGIN{printf \"%.0f\", $ELAPSED/$NIN*140/60}")
  echo "--- RUNTIME BUDGET (contract §4: 3 tasks => 420 min for 140 frames) ---"
  echo "    measured  : ${PERFRAME}s/frame  (includes one-time model loading)"
  echo "    140-frame : ~${EST140} min   vs budget 420 min"
  echo "    NOTE: eval HW is an RTX A5000 (24 GB), slower than this L40S -> keep margin."
  echo "    Safety lever if tight: TIGER_FINE_GRAFT=0 (FINE w3 0.7092 -> 0.701, halves FINE cost)."
fi

if [ "$GATE_RC" != 0 ] && [ ! -f "$OUT/task3.csv" ]; then
  echo "!! contract run produced no output — cannot score. Stopping."; exit 1
fi

# ---------------------------------------------------------------------------
# (1b) NOVEL-RESOLUTION COVERAGE — the named delivery hazard for this challenge.
#      Contract §2: "Frames are RGB PNG in the native resolution of the recording.
#      Resolutions differ between centers and between laparoscopic and robotic cases.
#      Do not assume a fixed input size." §6's failure list names a "hard coded input
#      size that breaks on a center with a different resolution".
#      This axis had ONLY ever been proven on the apptainer .sif (jobs 6704195/96) —
#      which, by the same argument that found the stale Dockerfile, is the WRONG
#      ARTIFACT. Prove it on the .tar.gz we will actually upload.
#      Frames are REGENERATED here (not reused from selftest_0719/input_varres) so
#      this job is self-contained and cannot silently pass on a stale directory.
# ---------------------------------------------------------------------------
echo "############ (1b) EXACT contract command, 6 NOVEL resolutions ############"
INV=$BASE/input_varres; OUTV=$BASE/results_varres; rm -rf "$INV" "$OUTV"; mkdir -p "$INV"
$PY - "$IN" "$INV" <<'PYEOF'
import sys, glob, os
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
# resolutions the model has NEVER seen: none equals the 1920x1080 / 1280x720 training
# sizes; mixes 16:9, 4:3, 5:4 and an ABOVE-native 1936x1088 (upscale path).
RES = [(1600, 900), (1333, 750), (1280, 1024), (1024, 768), (960, 544), (1936, 1088)]
frames = sorted(glob.glob(os.path.join(src, "*.png")))[:len(RES)]
assert len(frames) == len(RES), f"need {len(RES)} source frames, got {len(frames)}"
for p, wh in zip(frames, RES):
    im = Image.open(p).convert("RGB").resize(wh, Image.BILINEAR)
    im.save(os.path.join(dst, os.path.basename(p)))   # filename UNCHANGED (contract §2)
    print(f"  {os.path.basename(p)} -> {wh}")
PYEOF
NINV=$(ls -1 "$INV"/*.png | wc -l); echo "varres input frames: $NINV"
TARGZ="$TARGZ" IMAGE="$IMAGE" IN="$INV" OUT="$OUTV" GPU=1 ST=/tmp/tgate_$SLURM_JOB_ID KEEP=1 \
  bash "$SRC/contract_test.sh"
VARRES_GATE_RC=$?
echo "VARRES_CONTRACT_GATE_RC=$VARRES_GATE_RC"
echo "--- output size == input size, per novel resolution ---"
$PY - "$INV" "$OUTV" <<'PYEOF'
import sys, glob, os
from PIL import Image
inp, out = sys.argv[1], sys.argv[2]
bad = 0
for p in sorted(glob.glob(os.path.join(inp, "*.png"))):
    n = os.path.basename(p); isz = Image.open(p).size
    for t in ("task1", "task2"):
        q = os.path.join(out, t, n)
        if not os.path.exists(q):
            print(f"  MISSING {t}/{n}"); bad += 1; continue
        osz = Image.open(q).size
        flag = "OK " if osz == isz else "MISMATCH"
        if osz != isz: bad += 1
        print(f"  {flag} {t}/{n}: in {isz} -> out {osz}")
print(f"VARRES_SIZE_MISMATCHES={bad}")
PYEOF

# ---------------------------------------------------------------------------
# (2) FORMAT / ROUTING SELF-TEST on the REAL container output (native + varres)
# ---------------------------------------------------------------------------
echo "############ (2) format + routing self-test on the CONTAINER output ############"
$PY "$EXP/selftest_verify.py" \
    --repo "$REPO" --gt-root "$ROOT/data/synapse" \
    --native-out "$OUT" --native-in "$IN" \
    --varres-out "$OUTV" --varres-in "$INV" \
    --make-downscaled "$BASE/official256" \
    --verdict "$BASE/verdict.json"
SELFTEST_RC=$?
echo "SELFTEST_RC=$SELFTEST_RC"

# ---------------------------------------------------------------------------
# (3) OFFICIAL SCORER — metrics/01_evaluate_challenge.py
#     (it discovers methods as immediate sub-dirs of --pred, hence pred/AIMS/)
#
# ★ STRUCTURAL SEPARATION, NOT AN ANNOTATION.
#   The baked covis_matrix.json is fitted on ALL 10 training cases — including the
#   held-out cases 14/15 used here — so a task3 score measured on this output is
#   FIT-ON-EVAL for the prior half. We have spent this month unwinding exactly that
#   class of contamination (MultiBypass val==test; RARE26's four apparatus defects),
#   so the contaminated number is NOT produced next to the clean ones:
#     * PRIMARY run  = task1 + task2 ONLY  -> the reproduction check. The scorer
#       auto-detects tasks from the GT tree, so REMOVING gt/task3.csv makes it
#       structurally impossible for a task3 number to appear in this report.
#     * DIAGNOSTIC run = task3, in a directory whose NAME says fit-on-eval, with a
#       banner prepended to the report and a marker file beside it.
# ---------------------------------------------------------------------------
echo "############ (3a) OFFICIAL SCORER — task1+task2 = THE REPRODUCTION CHECK ############"
cd "$REPO"
T3GT=$BASE/official256/gt/task3.csv
T3HOLD=$BASE/_task3_gt_withheld_from_primary.csv
[ -f "$T3GT" ] && mv "$T3GT" "$T3HOLD"     # scorer now cannot detect task3
$PY metrics/01_evaluate_challenge.py \
    --gt   "$BASE/official256/gt" \
    --pred "$BASE/official256/pred" \
    --out  "$BASE/official_report_task12_REPRODUCTION.md" \
    --figures-dir none --save-json
OFFICIAL_RC=$?
echo "OFFICIAL_RC=$OFFICIAL_RC"
echo "--- task1/task2 ranking (the honest reproduction check) ---"
grep -A4 "^| Rank" "$BASE/official_report_task12_REPRODUCTION.md" 2>/dev/null | head -30
echo "--- dossier reference: task1 MERGED w3 0.552 · task2 FINE w3 0.7092 (160px, w3 sub-avg);"
echo "    the official scorer reports the WEIGHTED all-class Dice at 256px, so compare against"
echo "    the previously recorded official-256 figures, and use verdict.json for the w3 numbers. ---"
$PY -c "
import json,sys
d=json.load(open('$BASE/verdict.json'))['dice']
print('    verdict.json (160px, official both-empty semantics, case-mean):')
print('      task1 MERGED overall %.4f  w3 %.4f   (dossier w3 0.552)'%(d['merged_overall'],d['merged_w3']))
print('      task2 FINE   overall %.4f  w3 %.4f   (dossier w3 0.7092)'%(d['fine_overall'],d['fine_w3']))
" 2>/dev/null

echo "############ (3b) task3 — QUARANTINED DIAGNOSTIC (fit-on-eval, NOT a result) ############"
DIAG=$BASE/task3_FIT-ON-EVAL_DIAGNOSTIC-ONLY
rm -rf "$DIAG"; mkdir -p "$DIAG/gt" "$DIAG/pred/AIMS"
cat > "$DIAG/DO_NOT_QUOTE_THIS_NUMBER.txt" <<'MARK'
The task3 score in this directory is FIT-ON-EVAL and must never be quoted as a result.

The shipped /opt/tiger/covis_matrix.json is fitted on ALL 10 training cases (n_cases=10,
n_frames=140), which INCLUDES the held-out cases 14 and 15 scored here. The station
co-visibility prior therefore saw these frames' labels during fitting, so this number is
contaminated for the prior half of the blend.

Fitting the shipped matrix on all 10 cases is CORRECT for the hidden test set — the defect
is in measuring it here, not in the artifact.

The honest, pre-registered, held-out task3 numbers (matrix fit on the OTHER 8 cases only)
are recorded in the dossier and in t3_prior/prior_results.json:
    pre-registered lambda=0.5 blend (SHIPPED) : macro-F1 0.71277 / AUROC 0.94860
    prior-alone head-to-head                  : macro-F1 0.73066
    deployed seg-cond head (no prior)         : macro-F1 0.62757

This directory exists ONLY to confirm the container emits a scorable task3.csv.
The reproduction check is task1/task2 Dice, in official_report_task12_REPRODUCTION.md.
MARK
if [ -f "$T3HOLD" ]; then
  cp "$T3HOLD" "$DIAG/gt/task3.csv"
  cp "$BASE/official256/pred/AIMS/task3.csv" "$DIAG/pred/AIMS/task3.csv"
  $PY metrics/01_evaluate_challenge.py \
      --gt "$DIAG/gt" --pred "$DIAG/pred" \
      --out "$DIAG/REPORT_FIT-ON-EVAL_DIAGNOSTIC-ONLY.md" --figures-dir none
  T3DIAG_RC=$?
  # prepend the banner INTO the report so the file cannot be read without it
  if [ -f "$DIAG/REPORT_FIT-ON-EVAL_DIAGNOSTIC-ONLY.md" ]; then
    { echo '# ⛔ FIT-ON-EVAL — DIAGNOSTIC ONLY — DO NOT QUOTE'
      echo
      echo 'The shipped covis_matrix.json was fitted on all 10 training cases, INCLUDING the'
      echo 'cases 14/15 scored below. This task3 number is contaminated for the prior half of'
      echo 'the blend and is NOT comparable to the pre-registered held-out 0.71277.'
      echo 'It confirms only that the container emits a scorable task3.csv.'
      echo 'See DO_NOT_QUOTE_THIS_NUMBER.txt next to this file.'
      echo
      echo '---'
      echo
      cat "$DIAG/REPORT_FIT-ON-EVAL_DIAGNOSTIC-ONLY.md"
    } > "$DIAG/.tmp" && mv "$DIAG/.tmp" "$DIAG/REPORT_FIT-ON-EVAL_DIAGNOSTIC-ONLY.md"
  fi
  echo "task3 diagnostic rc=$T3DIAG_RC (report + marker in $DIAG)"
  echo "⛔ the task3 figure in $DIAG is FIT-ON-EVAL — the honest held-out number is 0.71277"
else
  echo "(no task3 GT built; diagnostic skipped)"
  T3DIAG_RC=NA
fi

# ---------------------------------------------------------------------------
# (4) ADVERSARIAL FILENAME TEST inside the REAL container (task3 station prior must
#     fail closed on renamed files and fall back to pure seg-cond, bit-identically).
# ---------------------------------------------------------------------------
echo "############ (4) adversarial-filename test in the REAL container ############"
AIN=$BASE/adv_in; AOUT=$BASE/adv_out; rm -rf "$AIN" "$AOUT"; mkdir -p "$AIN" "$AOUT"
i=0
for f in $(ls "$IN"/*.png | head -6); do
  i=$((i+1)); n=$(basename "$f" .png)
  if   [ $i -le 3 ]; then cp "$f" "$AIN/$n.png"
  elif [ $i -eq 4 ]; then cp "$f" "$AIN/frame_0004.png"
  elif [ $i -eq 5 ]; then cp "$f" "$AIN/${n}_copy.png"
  else                    cp "$f" "$AIN/$(echo "$n" | tr 'A-Z' 'a-z').png"; fi
done
ls "$AIN"
ST2=/tmp/tadv_$SLURM_JOB_ID; mkdir -p "$ST2/root" "$ST2/runroot"
P2="podman --root $ST2/root --runroot $ST2/runroot --storage-driver=vfs --storage-opt vfs.ignore_chown_errors=true"
$P2 load -i "$TARGZ" >/dev/null 2>&1
for G in "--gpus all" "--device nvidia.com/gpu=all"; do
  $P2 run --rm $G --network none --shm-size 8g \
    -v "$AIN":/input:ro -v "$AOUT":/output "$IMAGE" 2>&1 | tee "$BASE/adv_run.log" | grep -E "T3|PRIOR|fallback"
  [ ${PIPESTATUS[0]} -eq 0 ] && break
done
NPRIOR=$(grep -c "PRIOR+SEG\[" "$BASE/adv_run.log")
NFALL=$(grep -c "SEG-ONLY(fallback)" "$BASE/adv_run.log")
echo ">>> prior path on $NPRIOR frames, fallback on $NFALL frames (expect 3 and 3)"
ADV_OK=0; { [ "$NPRIOR" = 3 ] && [ "$NFALL" = 3 ]; } && ADV_OK=1
echo "ADV_OK=$ADV_OK"
rm -rf "$ST2" /tmp/tgate_$SLURM_JOB_ID 2>/dev/null

# ---------------------------------------------------------------------------
echo "############ VERDICT ############"
echo "routing_clone_current : $ROUTING_FRESH   (1 = our clone == origin/main; spec has not drifted again)"
echo "contract_gate_native  : $GATE_RC        (exact §6 command, bare image, --network none, 28 frames)"
echo "contract_gate_varres  : $VARRES_GATE_RC (same command, 6 NOVEL resolutions incl. above-native)"
echo "selftest_rc           : $SELFTEST_RC    (routing/palette/size, native AND varres)"
echo "official_scorer_rc    : $OFFICIAL_RC    (task1+task2 ONLY = the reproduction check)"
echo "adversarial_task3     : $ADV_OK"
echo "artifact              : $TARGZ"
echo "sha256                : $(sha256sum "$TARGZ" | cut -d' ' -f1)"
echo "verdict json          : $BASE/verdict.json"
echo "REPRODUCTION report   : $BASE/official_report_task12_REPRODUCTION.md   (task1 + task2)"
echo "task3 (quarantined)   : $BASE/task3_FIT-ON-EVAL_DIAGNOSTIC-ONLY/"
echo
echo "⛔ task3 LOCAL NUMBER IS DIAGNOSTIC ONLY — DO NOT QUOTE, DO NOT LOG AS A RESULT."
echo "   The baked covis_matrix.json is fitted on ALL 10 training cases (n_cases=10),"
echo "   INCLUDING held-out cases 14/15 => fit-on-eval for the prior half. It is kept out"
echo "   of the primary report structurally (gt/task3.csv withheld, so the scorer cannot"
echo "   even detect task3) and lives only in the quarantine directory above."
echo "   The honest pre-registered held-out task3 numbers stay: blend 0.71277 / AUROC 0.94860."
echo "✅ THE REPRODUCTION CHECK IS task1/task2 DICE (dossier: MERGED w3 0.552, FINE w3 0.7092)."
echo "=== DONE $(date -Is) ==="
