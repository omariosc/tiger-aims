#!/bin/bash
# Stage a self-contained build context on SCRATCH (home quota ~65 GB; the two
# nnU-Net checkpoints are ~900 MB total -> keep off home).
# Copies: minimal model weights (checkpoint_final.pth + plans.json + dataset.json),
# the T3 head, the challenge_repo, the custom trainers, the FP-suppress configs,
# the source files, and the apptainer def + Dockerfile.
set -e
CTX=${1:-/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/docker_ctx}
SRC=/users/USERNAME/Tiger-SQ-AI/docker
RES=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/nnUNet_results
NNJ=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet
SEGPKG=/scratch/USERNAME/miccai-2026/seg_env/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/variants/loss

# FINE_DS is overridable so the 40-case/6-centre retrain (Dataset212_TigerT2fineV2) can be
# staged without editing this file. Adoption rule 2026-09-03: T2 retrain WON ensemble-vs-
# ensemble (+0.3253 w3 on held-out centres 2/3/4/6/7); T1 LOST (-0.0140) so MERGED stays 201.
FINE_DS=${FINE_DS:-Dataset202_TigerT2fine}
MERGED_DS=Dataset201_TigerT1coarse
SYNTH_DS=Dataset205_TigerT2fineSynthCT
# CHAMPION stack (CONFIRMED 2026-06-25 job 6291940, deciding metric = w3 sub-avg Dice @160px):
#   FINE   (official task1) = 3-seed FT ENSEMBLE + replace{8} CT205-synth graft + FP  (w3 0.7092)
#   MERGED (official task2) = TopK-Tversky single model                              (w3 0.552)
#   T3                      = seg-cond head on ENSEMBLE masks (ckpt_segcond_ensmask, F1 0.6276)
# FINE_ENS_TRS = comma-separated FINE ensemble member trainers (softmax-prob avg).
FINE_ENS_TRS=${FINE_ENS_TRS:-nnUNetTrainer_250epochs_NoMirror_FocalTversky__nnUNetPlans__2d,nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed1__nnUNetPlans__2d,nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed2__nnUNetPlans__2d}
MERGED_TR=${MERGED_TR:-nnUNetTrainer_250epochs_NoMirror_TopKTversky__nnUNetPlans__2d}
# CT205 diffusion-synth ensemble members for the replace{8} graft (same trainer names,
# different dataset = Dataset205). Set FINE_GRAFT_IDS="" to skip staging the synth weights.
SYNTH_ENS_TRS=${SYNTH_ENS_TRS:-nnUNetTrainer_250epochs_NoMirror_FocalTversky__nnUNetPlans__2d,nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed1__nnUNetPlans__2d,nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed2__nnUNetPlans__2d}
# ${VAR:-D} substitutes D when VAR is unset OR EMPTY, so FINE_GRAFT_IDS="" -- the documented way
# to disable the graft -- silently became "8" and the build staged the synth weights anyway.
# ${VAR-D} substitutes only when UNSET, which is what is meant here.
FINE_GRAFT_IDS=${FINE_GRAFT_IDS-8}
# T3 head + FP-suppress configs also overridable to match the chosen seg stack.
# CHAMPION T3 = seg-cond head trained on the ENSEMBLE masks (ckpt_segcond_ensmask, F1 0.6276).
T3_CKPT=${T3_CKPT:-/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/t3/ckpt_segcond_ensmask/best.pt}
# Winner-tuned FP configs (tuned ON the ensemble / TopK preds).
FP_FINE_CFG=${FP_FINE_CFG:-$NNJ/t2_ft_ens_fpsuppress_fullres.json}
FP_MERGED_CFG=${FP_MERGED_CFG:-$NNJ/t1_topk_fpsuppress_fullres.json}

echo "=== staging build context at $CTX ==="
echo "    FINE_ENS_TRS=$FINE_ENS_TRS"
echo "    MERGED_TR=$MERGED_TR"
echo "    T3_CKPT=$T3_CKPT"
rm -rf "$CTX"; mkdir -p "$CTX"

# source + trainers + build recipes
cp $SRC/inference.py $SRC/station_boundary_map.py $SRC/Dockerfile $SRC/tiger.def "$CTX/"
# build-time assertions run INSIDE the image (trainer discovery / checkpoints / prior)
cp $SRC/build_checks.py "$CTX/"
cp /users/USERNAME/Tiger-SQ-AI/experiments/fp_suppress.py "$CTX/"
# T-1 acquisition-station co-visibility prior (task3); baked in, no runtime download
cp $SRC/covis_matrix.json "$CTX/"
cp $SEGPKG/nnUNetTrainerTverskyBoundary.py "$CTX/"
# cbdice_tiger.py = dependency of the cbDice trainers imported at the top of the
# TverskyBoundary module; MUST be staged or trainer-discovery crashes inside the
# container (job 6209008 FAILED on the missing module). Belt-and-braces: the import
# is also now guarded fail-soft, but stage the file so the cbDice trainers work too.
cp /scratch/USERNAME/miccai-2026/seg_env/lib/python3.12/site-packages/nnunetv2/training/loss/cbdice_tiger.py "$CTX/"
# FP-suppression tuned configs -> ALWAYS land at the canonical names inference.py expects
cp "$FP_FINE_CFG"   "$CTX/t2_ft_ens_fpsuppress_fullres.json"
cp "$FP_MERGED_CFG" "$CTX/t1_topk_fpsuppress_fullres.json"
cp -r /users/USERNAME/Tiger-SQ-AI/challenge_repo "$CTX/challenge_repo"
rm -rf "$CTX/challenge_repo/.git" "$CTX"/challenge_repo/metrics/__pycache__ \
       "$CTX"/challenge_repo/metrics/demo_output "$CTX"/challenge_repo/metrics/demo_data

# minimal model weights (exclude the 5 GB validation .npz bloat). INTERNAL dir names
# (Dataset202=fine, Dataset201=merged) so inference.py finds them. FINE stages ALL
# ensemble members; MERGED stages the single TopK model.
stage_model() {  # $1=dataset $2=trainer
  local D=$1 TR=$2
  mkdir -p "$CTX/nnUNet_results/$D/$TR/fold_0"
  cp "$RES/$D/$TR/plans.json"   "$CTX/nnUNet_results/$D/$TR/" 2>/dev/null || cp "$RES/$D/plans.json" "$CTX/nnUNet_results/$D/$TR/" 2>/dev/null || true
  cp "$RES/$D/$TR/dataset.json" "$CTX/nnUNet_results/$D/$TR/" 2>/dev/null || cp "$RES/$D/dataset.json" "$CTX/nnUNet_results/$D/$TR/" 2>/dev/null || true
  [ -f "$RES/$D/$TR/dataset_fingerprint.json" ] && cp "$RES/$D/$TR/dataset_fingerprint.json" "$CTX/nnUNet_results/$D/$TR/"
  cp "$RES/$D/$TR/fold_0/checkpoint_final.pth" "$CTX/nnUNet_results/$D/$TR/fold_0/"
}
IFS=',' read -ra FINE_TR_ARR <<< "$FINE_ENS_TRS"
for TR in "${FINE_TR_ARR[@]}"; do stage_model "$FINE_DS" "$TR"; done
stage_model "$MERGED_DS" "$MERGED_TR"
FINE_TR0=${FINE_TR_ARR[0]}
# CT205 synth ensemble for the replace{8} graft (only if grafting is enabled)
if [ -n "$FINE_GRAFT_IDS" ]; then
  echo "    staging CT205 synth ensemble ($SYNTH_DS) for replace{$FINE_GRAFT_IDS} graft"
  IFS=',' read -ra SYNTH_TR_ARR <<< "$SYNTH_ENS_TRS"
  for TR in "${SYNTH_TR_ARR[@]}"; do stage_model "$SYNTH_DS" "$TR"; done
fi

# T3 head
mkdir -p "$CTX/t3_head"
cp "$T3_CKPT" "$CTX/t3_head/best.pt"

# The synth replace{8} graft is DROPPED for the retrained FINE model (2026-09-03): it was worth
# +0.0082 against the OLD weak champion, and its premise -- CT205 synth beats the champion on
# class 8 -- no longer holds now the champion is +0.325 stronger. Empty FINE_GRAFT_IDS must
# therefore also turn the FLAG off; it used to be hardcoded to 1, which would have shipped
# graft-enabled with no ids.
if [ -n "$FINE_GRAFT_IDS" ]; then FINE_GRAFT_FLAG=1; else FINE_GRAFT_FLAG=0; fi

# record the chosen trainer dir names so inference.py can be pointed at them
cat > "$CTX/stack.env" <<ENV
FINE_DATASET=$FINE_DS
FINE_TRAINER=$FINE_TR0
FINE_ENSEMBLE_TRAINERS=$FINE_ENS_TRS
MERGED_DATASET=$MERGED_DS
MERGED_TRAINER=$MERGED_TR
FINE_GRAFT=${FINE_GRAFT_FLAG}
FINE_GRAFT_IDS=$FINE_GRAFT_IDS
SYNTH_DATASET=$SYNTH_DS
SYNTH_ENSEMBLE_TRAINERS=$SYNTH_ENS_TRS
T3_CKPT=$T3_CKPT
ENV

# ---------------------------------------------------------------------------
# ★ FIX THE FILE MODES HOST-SIDE. Our umask is 077, so everything staged above is
# 0600/0700 and `COPY` PRESERVES THE SOURCE MODES -> the image's own code ends up
# readable only by root. That defect made CLiMB's first two submissions INVALID
# ("python: can't open file '/app/container_run.py': [Errno 13] Permission denied",
# their evaluator runs `--user $(id -u):$(id -g)`) and shipped in all three ORENA
# images. TIGER's contract passes no --user so we are not exposed, but the class
# costs nothing to remove. Fixing it HERE rather than with an in-image `chmod -R`
# matters: a chmod on files from a lower layer forces an overlayfs copy-up and can
# double the image (ORENA: 17.7 GB -> 35 GB).
# ---------------------------------------------------------------------------
chmod -R a+rX "$CTX"
_BADF=$(find "$CTX" ! -perm -o+r | wc -l)
_BADD=$(find "$CTX" -type d ! -perm -o+x | wc -l)
echo "=== context modes: $_BADF not-world-readable, $_BADD dirs not-traversable (want 0 / 0) ==="
[ "$_BADF" -eq 0 ] && [ "$_BADD" -eq 0 ] || { echo "FATAL: chmod did not take"; exit 1; }

echo "=== context staged ==="
du -sh "$CTX"
find "$CTX" -maxdepth 3 -type f | head -40
echo
echo "Apptainer build:  cd $CTX && apptainer build --ignore-fakeroot-command tiger.sif tiger.def"
