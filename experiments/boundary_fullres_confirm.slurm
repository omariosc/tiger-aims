#!/bin/bash
#SBATCH --job-name=tiger_bnd_fullres
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet/bnd_fullres_%j.out
set -e
# TIGER SQ-AI — FULL-RES confirmation of the nHD-targeted boundary-refinement lever (2026-07-20).
# The 160px sweep showed relative island removal cuts weighted nHD by -0.006..-0.008 at ZERO Dice
# cost on BOTH tasks. 160px is a PROXY and the effect is small -> confirm at the SUBMITTED scale
# before banking. Official nHD is O(n^2) over ALL fg pixels -> intractable full-res for the big
# classes, so we score --w3-only (the deciding tier; thin classes have few fg px => tractable).
# NOTE: this is CPU-bound scoring; it runs as a batch job (NOT on a login node) because the
# dossier documents full-res nHD taking many hours (0i: 10h timeouts even 12-way parallel).
PY=/scratch/USERNAME/miccai-2026/seg_env/bin/python
N=/scratch/USERNAME/miccai-2026/Tiger-SQ-AI/nnunet
cd /users/USERNAME/Tiger-SQ-AI

echo "########## MERGED (official task1) — champion t1_topk_fp, w3 0.5520 ##########"
$PY experiments/boundary_refine_sweep.py --task t1 --w3-only --downscale 0 \
  --pred $N/predTs/t1_topk_fp --gt $N/nnUNet_raw/Dataset201_TigerT1coarse/labelsTr \
  --val-cases center_1_case_14 center_1_case_15 --workers 24 \
  --variants baseline rel0.05 rel0.10 --out $N/boundary_fullres_merged.json

echo "########## FINE (official task2) — champion t2_ct205_rep8_fp, w3 0.7092 ##########"
$PY experiments/boundary_refine_sweep.py --task t2 --w3-only --downscale 0 \
  --pred $N/predTs/t2_ct205_rep8_fp --gt $N/nnUNet_raw/Dataset202_TigerT2fine/labelsTr \
  --val-cases center_1_case_14 center_1_case_15 --workers 24 \
  --variants baseline rel0.05 rel0.10 --out $N/boundary_fullres_fine.json

echo "########## DONE — BANK the lever only if full-res w3 nHD improves with w3 Dice flat ##########"
