#!/bin/bash
#SBATCH --job-name=tiger_adopt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=gpu013
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --nice=600
#SBATCH --output=/users/USERNAME/joblogs/tiger_adopt_%j.out
#SBATCH --error=/users/USERNAME/joblogs/tiger_adopt_%j.err
#
# THE PRE-REGISTERED ADOPTION RULE for the Tiger retrain, executed.
#
# Compares the INCUMBENT (shipped; trained on 140 frames, ALL from centre 1) against the
# RETRAIN (new 40-case / 524-frame / 6-centre release, same recipe) on the SAME frames,
# restricted to held-out val cases from centres 2,3,4,6,7 — centres the incumbent never
# trained on.
#
# WHY THE RESTRICTION: our new centre-stratified val also holds out 3 centre-1 cases, and the
# incumbent may have TRAINED on those. Scoring both on the full val set would hand the incumbent
# frames it has already seen. Comparing against the recorded 0.7092/0.552 would be worse still —
# different protocol entirely.
#
# ADOPT only if the retrain wins on these frames. Ties go to the frozen artifact (already built,
# gated and submitted). T1 and T2 are decided INDEPENDENTLY.
set -uo pipefail
[ -d /users/USERNAME/Tiger-SQ-AI/experiments ] || { echo "[eio-guard] HOME unreadable -> abort"; exit 3; }
SEG=/scratch/USERNAME/miccai-2026/seg_env/bin
NN=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet
export nnUNet_raw=$NN/nnUNet_raw nnUNet_preprocessed=$NN/nnUNet_preprocessed nnUNet_results=$NN/nnUNet_results
W=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/adopt_${SLURM_JOB_ID}
mkdir -p $W

# ---- build the evaluation frame set: val cases from centres 2,3,4,6,7 only -------------------
$SEG/python - "$W" <<'PY'
import json, os, re, shutil, sys
W = sys.argv[1]
pre = "/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/nnUNet_preprocessed"
raw = "/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/nnUNet_raw"
KEEP = ("center_2", "center_3", "center_4", "center_6", "center_7")
for ds, name in ((211, "TigerT1coarseV2"), (212, "TigerT2fineV2")):
    sp = json.load(open(f"{pre}/Dataset{ds}_{name}/splits_final.json"))[0]
    val = [s for s in sp["val"] if s.startswith(KEEP)]
    dropped = [s for s in sp["val"] if not s.startswith(KEEP)]
    src = f"{raw}/Dataset{ds}_{name}"
    di, dl = f"{W}/in_{ds}", f"{W}/gt_{ds}"
    os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
    for s in val:
        for c in range(3):
            shutil.copy(f"{src}/imagesTr/{s}_{c:04d}.png", di)
        shutil.copy(f"{src}/labelsTr/{s}.png", dl)
    cases = sorted({re.match(r"(center_\d+_case_\d+)", s).group(1) for s in val})
    print(f"  Dataset{ds}: {len(val)} eval frames from {len(cases)} cases {cases}")
    print(f"             dropped {len(dropped)} centre-1 val frames the incumbent may have trained on")
PY
[ $? -eq 0 ] || { echo "[FAIL] could not build the eval set"; exit 8; }

# ---- predict: incumbent and retrain, same frames ---------------------------------------------
# T1 (merged):  incumbent Dataset201 TopKTversky   vs  retrain Dataset211 TopKTversky
# T2 (fine):    incumbent Dataset202 FocalTversky  vs  retrain Dataset212 FocalTversky
run () {  # $1 label  $2 dataset-id  $3 trainer  $4 plans  $5 input  $6 out
  echo "=== predict $1 (d$2 $3 $4) ==="
  $SEG/nnUNetv2_predict -i "$5" -o "$6" -d "$2" -c 2d -f 0 -tr "$3" -p "$4" \
      --disable_tta -chk checkpoint_final.pth 2>&1 | tail -3
  echo "    rc=$? files=$(ls "$6" 2>/dev/null | wc -l)"
}
run "T1 INCUMBENT" 201 nnUNetTrainer_250epochs_NoMirror_TopKTversky  nnUNetPlans $W/in_211 $W/p_t1_inc
run "T1 RETRAIN"   211 nnUNetTrainer_250epochs_NoMirror_TopKTversky  nnUNetPlans $W/in_211 $W/p_t1_new
run "T2 INCUMBENT" 202 nnUNetTrainer_250epochs_NoMirror_FocalTversky nnUNetPlans $W/in_212 $W/p_t2_inc
run "T2 RETRAIN"   212 nnUNetTrainer_250epochs_NoMirror_FocalTversky nnUNetPlans $W/in_212 $W/p_t2_new

# ---- score all four with the OFFICIAL weighted metric ----------------------------------------
$SEG/python /users/USERNAME/Tiger-SQ-AI/experiments/adoption_rule_score.py "$W"
RC=$?
echo "=== adoption scorer rc=$RC ==="
exit $RC
