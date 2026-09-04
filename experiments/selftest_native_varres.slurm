#!/bin/bash
#SBATCH --job-name=tiger_selftest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/selftest_0719/selftest_%j.out
set -e
# TIGER SQ-AI 2026 — end-to-end container self-test vs the PUBLISHED contract (2026-07-19).
# Runs the ACTUAL fixed inference.py entrypoint (seg_env == the .sif's baked env) from RAW RGB
# frames -> full champion stack (3-seed FT ens + CT205 replace{8} graft + FP + TopK merged +
# seg-cond T3) -> /output, on (a) 28 native val frames and (b) 6 NOVEL-resolution frames.
# Then verifies task1=MERGED / task2=FINE routing, native-variable-resolution correctness,
# task3 schema, sanity Dice, + runs the OFFICIAL 01_evaluate_challenge.py on 256px copies.

PY=/scratch/USERNAME/miccai-2026/seg_env/bin/python
ROOT=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI
BASE=$ROOT/selftest_0719
REPO=/users/USERNAME/Tiger-SQ-AI/challenge_repo
export TIGER_RESULTS=$ROOT/nnunet/nnUNet_results
export TIGER_T3_CKPT=$ROOT/t3/ckpt_segcond_ensmask/best.pt
export TIGER_REPO=$REPO
export TIGER_FP_FINE=$ROOT/nnunet/t2_ft_ens_fpsuppress_fullres.json
export TIGER_FP_MERGED=$ROOT/nnunet/t1_topk_fpsuppress_fullres.json
export nnUNet_raw=/tmp/nnraw_$SLURM_JOB_ID
export nnUNet_preprocessed=/tmp/nnprep_$SLURM_JOB_ID
export nnUNet_results=$TIGER_RESULTS
export MKL_THREADING_LAYER=GNU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p $nnUNet_raw $nnUNet_preprocessed
cd /users/USERNAME/Tiger-SQ-AI/docker

echo "############ (1) NATIVE run — 28 val frames (cases 14,15), TIMED ############"
rm -rf $BASE/out_native $BASE/out_varres
t0=$(date +%s)
$PY inference.py --input $BASE/input_native --output $BASE/out_native
t1=$(date +%s)
NAT_S=$((t1-t0)); NAT_N=$(ls $BASE/input_native | wc -l)
echo ">>> NATIVE: $NAT_N frames in ${NAT_S}s = $(python3 -c "print(f'{$NAT_S/$NAT_N:.2f}')") s/frame (3-task combined)"
echo "    contract budget = 3 tasks x 60s = 180 s/frame; on eval RTX A5000 (slower than L40S) keep margin"

echo "############ (2) VARIABLE-RESOLUTION run — 6 novel sizes ############"
$PY inference.py --input $BASE/input_varres --output $BASE/out_varres
t2=$(date +%s); echo ">>> VARRES: 6 novel-res frames in $((t2-t1))s"

echo "############ (3) VERIFY (routing / native-varres size / task3 / sanity Dice) ############"
$PY /users/USERNAME/Tiger-SQ-AI/experiments/selftest_verify.py \
    --repo $REPO --gt-root $ROOT/data/synapse \
    --native-out $BASE/out_native --native-in $BASE/input_native \
    --varres-out $BASE/out_varres --varres-in $BASE/input_varres \
    --make-downscaled $BASE/eval256 --verdict $BASE/selftest_verdict.json
VERIFY_RC=$?

echo "############ (4) OFFICIAL 01_evaluate_challenge.py on 256px copies (scorer-ingestion proof) ############"
cd $REPO
$PY metrics/01_evaluate_challenge.py --gt $BASE/eval256/gt --pred $BASE/eval256/pred \
    --out $BASE/official_report_256.md --figures-dir none 2>&1 | tail -25 || echo "(official eval had issues — see above)"
echo "--- official report head ---"; head -40 $BASE/official_report_256.md 2>/dev/null || true

rm -rf $nnUNet_raw $nnUNet_preprocessed
echo "############ SELF-TEST DONE (verify rc=$VERIFY_RC) ############"
