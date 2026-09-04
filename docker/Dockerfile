# TIGER SQ-AI 2026 — SUBMISSION container (OCI/docker build for the contract-§7
# `docker save | gzip > tigersqai_<team>_task<list>.tar.gz` docker-archive).
#
#   task1/  (MERGED 15/16-cls) <- nnU-Net Dataset201_TigerT1coarse, TopK-Tversky  (w3 0.552)
#   task2/  (FINE   30/31-cls) <- nnU-Net Dataset202 3-seed Focal-Tversky ENSEMBLE
#                                 + CT205 replace{8} synth graft                  (w3 0.7092)
#   task3.csv (14 stations)    <- seg-conditioned MLP head on the FINE masks
#                                 blended (pre-registered lambda=0.5) with the baked
#                                 acquisition-station co-visibility prior          (F1 0.7128)
#
# ⚠ task1=MERGED / task2=FINE is the ORGANIZERS' CURRENT numbering (they flipped it
#   after our Jun-15 clone; re-verified against gitlab origin/main 29c6a02).
#
# This file MIRRORS docker/tiger.def (the apptainer recipe that produced the VERIFIED
# tiger_v2.sif) — same base, same pip set, same files, same env, same entrypoint.
# Build context = the staging dir produced by stage_build_context.sh (~2.8 GB).
#
# Base is FULLY QUALIFIED (docker.io/...): rootless buildah on AIRE enforces
# short-name resolution and cannot prompt without a TTY (this exact bug cost CLiMB
# a build cycle: "short-name resolution enforced, cannot prompt without TTY").
FROM docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCH_HOME=/opt/checkpoints \
    TIGER_RESULTS=/opt/tiger/nnUNet_results \
    TIGER_T3_CKPT=/opt/tiger/t3_head/best.pt \
    TIGER_REPO=/opt/tiger/challenge_repo \
    nnUNet_raw=/tmp/nnraw nnUNet_preprocessed=/tmp/nnprep \
    nnUNet_results=/opt/tiger/nnUNet_results \
    MKL_THREADING_LAYER=GNU

# No apt-get anywhere: the rootless single-UID namespace on this cluster breaks dpkg
# postinst (setgroups EPERM / chown-to-unmapped-uid) — that is what blocked the CLiMB
# ORB-SLAM3 image. Everything we need is pip-only.
RUN pip install --no-cache-dir \
        nnunetv2==2.8.0 "numpy<3" pillow pandas scipy scikit-image \
        scikit-learn batchgenerators dynamic-network-architectures acvl-utils

WORKDIR /opt/tiger

# --- code + baked config (no runtime downloads; --network none safe) ------------
COPY inference.py station_boundary_map.py fp_suppress.py /opt/tiger/
# T-1 acquisition-station co-visibility prior for task3 (BAKED, zero-parameter)
COPY covis_matrix.json /opt/tiger/
# FP-suppression tuned configs (test-time, training-free; resolution-invariant fracs)
COPY t2_ft_ens_fpsuppress_fullres.json t1_topk_fpsuppress_fullres.json /opt/tiger/
COPY challenge_repo /opt/tiger/challenge_repo

# --- custom nnU-Net trainers (needed for checkpoint loading / trainer discovery) --
# cbdice_tiger.py is imported at the TOP of the TverskyBoundary module, so it MUST be
# present or ALL nnU-Net trainer discovery crashes (job 6209008 died exactly this way)
# even though the submission trainers (TopK / FocalTversky) do not use cbDice.
# stack.env records WHICH models this image was staged with. Shipping it makes the image
# self-describing and lets build_checks.py assert the checkpoints the stack actually declares
# instead of a hardcoded list that goes stale the moment the stack changes (it did: the v3
# build failed 6 assertions demanding Dataset202/205 after the adopted stack correctly
# replaced them with Dataset212 and dropped the graft).
COPY stack.env /opt/tiger/
COPY nnUNetTrainerTverskyBoundary.py cbdice_tiger.py build_checks.py /opt/tiger/
RUN set -eu; \
    NN="$(python -c 'import nnunetv2,os;print(os.path.dirname(nnunetv2.__file__))')"; \
    echo "[build] nnunetv2 package at $NN"; \
    cp /opt/tiger/nnUNetTrainerTverskyBoundary.py \
       "$NN/training/nnUNetTrainer/variants/loss/nnUNetTrainerTverskyBoundary.py"; \
    cp /opt/tiger/cbdice_tiger.py "$NN/training/loss/cbdice_tiger.py"

# --- banked weights (staged by stage_build_context.sh) --------------------------
COPY nnUNet_results /opt/tiger/nnUNet_results
COPY t3_head /opt/tiger/t3_head

# --- BUILD-TIME ASSERTIONS (fail the build, never the submission) ---------------
RUN python /opt/tiger/build_checks.py

# --- PERMISSION GUARD (last step touching /opt/tiger) ---------------------------
# WHY: COPY preserves the SOURCE modes, and our umask is 077 -> a build context of
# 0600 files produces an image whose own code is unreadable by any uid but root.
# That defect made CLiMB's first TWO submissions INVALID ("python: can't open file
# '/app/container_run.py': [Errno 13] Permission denied") and shipped in all three
# ORENA images. The TIGER contract (§6) does NOT pass --user, so we are not exposed
# — but the class costs nothing to eliminate, so we eliminate it.
#
# The modes are fixed HOST-SIDE in stage_build_context.sh (`chmod -R a+rX`), so the
# `find` below normally matches NOTHING: a blanket `chmod -R` here would setattr every
# file of a lower layer and force an overlayfs copy-up, doubling a 2.8 GB tree
# (ORENA: 17.7 -> 35 GB). This guard therefore only repairs what is broken, and then
# ASSERTS the property. It also SELF-TESTS its own find expressions in BOTH directions
# against a planted 0600 file and a planted non-traversable 0744 dir first, so a
# silently-broken check can never report "clean" (the cuobjdump-cascade lesson).
RUN set -eu; \
    C=/tmp/_permctl; rm -rf "$C"; mkdir -p "$C/d" "$C/dok"; : > "$C/f"; : > "$C/fok"; \
    chmod 755 "$C"; chmod 600 "$C/f"; chmod 644 "$C/fok"; chmod 744 "$C/d"; chmod 755 "$C/dok"; \
    CF="$(find "$C" -mindepth 1 -type f ! -perm -o+r | wc -l)"; \
    CD="$(find "$C" -mindepth 1 -type d ! -perm -o+x | wc -l)"; \
    echo "[permguard] instrument control (planted 1 bad + 1 good of each):"; \
    echo "[permguard]   unreadable-file hits=$CF (want exactly 1)  non-traversable-dir hits=$CD (want exactly 1)"; \
    { [ "$CF" -eq 1 ] && [ "$CD" -eq 1 ]; } || { echo "!! permguard instrument BROKEN (must flag the bad AND spare the good) — refusing to certify"; exit 1; }; \
    chmod 644 "$C/f"; chmod 755 "$C/d"; \
    { [ "$(find "$C" -mindepth 1 -type f ! -perm -o+r | wc -l)" -eq 0 ] && \
      [ "$(find "$C" -mindepth 1 -type d ! -perm -o+x | wc -l)" -eq 0 ]; } \
      || { echo "!! permguard reports a REPAIRED tree as still-broken — refusing to certify"; exit 1; }; \
    echo "[permguard] instrument verified in BOTH directions (flags bad, clears fixed)"; \
    rm -rf "$C"; \
    BADF="$(find /opt/tiger ! -perm -o+r | wc -l)"; \
    BADD="$(find /opt/tiger -type d ! -perm -o+x | wc -l)"; \
    echo "[permguard] pre-fix: $BADF path(s) not world-readable, $BADD dir(s) not world-traversable"; \
    if [ "$BADF" -ne 0 ] || [ "$BADD" -ne 0 ]; then \
        echo "[permguard] WARNING: repairing in-image (fix the build context host-side to avoid layer copy-up)"; \
        find /opt/tiger ! -perm -o+r -exec chmod o+r {} + ; \
        find /opt/tiger -type d ! -perm -o+x -exec chmod o+x {} + ; \
    fi; \
    BADF2="$(find /opt/tiger ! -perm -o+r | wc -l)"; \
    BADD2="$(find /opt/tiger -type d ! -perm -o+x | wc -l)"; \
    echo "[permguard] post-fix: $BADF2 unreadable, $BADD2 non-traversable"; \
    [ "$BADF2" -eq 0 ] && [ "$BADD2" -eq 0 ] || { echo "!! /opt/tiger STILL not world-readable"; exit 1; }; \
    echo "[permguard] PASS: every /opt/tiger path is world-readable and every dir world-traversable"

# Contract §6 runs the image with NO arguments: `docker run ... tigersqai_myteam`.
# ENTRYPOINT+CMD must therefore be a complete, self-sufficient invocation.
ENTRYPOINT ["python", "/opt/tiger/inference.py"]
CMD ["--input", "/input", "--output", "/output"]
