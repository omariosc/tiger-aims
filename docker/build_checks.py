#!/usr/bin/env python3
"""TIGER SQ-AI — BUILD-TIME assertions, run inside the image (docker/Dockerfile).

Fails the BUILD (loud, cheap) rather than the SUBMISSION (silent, one shot) if any
of the champion stack's load-bearing pieces are missing.  Every check here maps to a
real campaign failure:

  1. custom trainer discovery            -> job 6209008 died on a missing cbdice_tiger
  2. champion checkpoints present        -> RARE26 "no checkpoints on disk" (lesson 4.7)
  3. baked task3 co-visibility prior     -> would silently fall back to seg-only
  4. FP-suppress configs + T3 head       -> silent accuracy loss, no error
  5. exact-RGB registries importable     -> unknown-colour pixels = unscorable frames
  6. NO network at runtime               -> HF/torch offline env actually set

Exit 0 iff every check passes.
"""
import json
import os
import sys
from pathlib import Path

TIGER = Path("/opt/tiger")
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


print("=== TIGER build-time assertions ===")

# 1. custom trainers must be importable AND discoverable by nnU-Net -------------
TRAINERS = [
    "nnUNetTrainer_250epochs_NoMirror_FocalTversky",
    "nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed1",
    "nnUNetTrainer_250epochs_NoMirror_FocalTversky_seed2",
    "nnUNetTrainer_250epochs_NoMirror_TopKTversky",
]
try:
    import importlib

    mod = importlib.import_module(
        "nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerTverskyBoundary"
    )
    for t in TRAINERS:
        check(f"trainer_importable::{t}", hasattr(mod, t))
except Exception as exc:  # noqa: BLE001
    check("trainer_module_import", False, repr(exc))

# nnU-Net's own recursive discovery is what actually runs at predict time.
try:
    import nnunetv2
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

    base = os.path.join(os.path.dirname(nnunetv2.__file__), "training", "nnUNetTrainer")
    for t in TRAINERS:
        cls = recursive_find_python_class(base, t, "nnunetv2.training.nnUNetTrainer")
        check(f"trainer_discoverable::{t}", cls is not None)
except Exception as exc:  # noqa: BLE001
    check("trainer_recursive_discovery", False, repr(exc))

# 2. champion checkpoints -------------------------------------------------------
# Derived from the image's OWN stack.env, never hardcoded. A hardcoded list describes the stack
# you had when you wrote it: build 7716155 failed 6 assertions demanding Dataset202 + the Dataset205
# graft weights, when the adopted stack had correctly replaced them with Dataset212 and dropped the
# graft. The check was stale, not the artifact. This version cannot go stale, and it is STRICTER --
# it also asserts the graft weights are ABSENT when the stack says the graft is off.
STACK = {}
_se = TIGER / "stack.env"
if _se.is_file():
    for _l in _se.read_text().splitlines():
        if "=" in _l and not _l.lstrip().startswith("#"):
            _k, _v = _l.split("=", 1)
            STACK[_k.strip()] = _v.strip()
check("stack_env_present", bool(STACK), ",".join(sorted(STACK)) or "MISSING")

def _trs(key):
    return [t for t in STACK.get(key, "").split(",") if t.strip()]

EXPECTED_CKPTS = []
if STACK.get("MERGED_DATASET") and STACK.get("MERGED_TRAINER"):
    EXPECTED_CKPTS.append((STACK["MERGED_DATASET"], STACK["MERGED_TRAINER"]))
for _t in _trs("FINE_ENSEMBLE_TRAINERS"):
    EXPECTED_CKPTS.append((STACK.get("FINE_DATASET", ""), _t))
_graft_on = STACK.get("FINE_GRAFT", "0") == "1"
if _graft_on:
    for _t in _trs("SYNTH_ENSEMBLE_TRAINERS"):
        EXPECTED_CKPTS.append((STACK.get("SYNTH_DATASET", ""), _t))
check("expected_ckpt_set_derived", len(EXPECTED_CKPTS) >= 4,
      f"{len(EXPECTED_CKPTS)} ckpts, graft={'ON' if _graft_on else 'OFF'}, fine={STACK.get('FINE_DATASET')}")

for ds, tr in EXPECTED_CKPTS:
    root = TIGER / "nnUNet_results" / ds / tr
    ck = root / "fold_0" / "checkpoint_final.pth"
    ok = ck.is_file() and (root / "plans.json").is_file() and (root / "dataset.json").is_file()
    size = f"{ck.stat().st_size / 1e6:.0f} MB" if ck.is_file() else "MISSING"
    check(f"checkpoint::{ds}/{tr.split('__')[0]}", ok, size)

# negative assertion: graft off => the synth weights must NOT be shipped (dead weight, and a
# graft silently re-enabled later would then find them and change behaviour)
if not _graft_on:
    _sd = STACK.get("SYNTH_DATASET", "Dataset205_TigerT2fineSynthCT")
    _present = (TIGER / "nnUNet_results" / _sd).exists()
    check(f"graft_off_implies_no_synth_weights::{_sd}", not _present,
          "ABSENT as expected" if not _present else "PRESENT despite FINE_GRAFT=0")

# 3. baked task3 co-visibility prior --------------------------------------------
prior = TIGER / "covis_matrix.json"
ok = prior.is_file()
detail = ""
if ok:
    d = json.loads(prior.read_text())
    ok = (
        len(d.get("stations", [])) == 14
        and len(d.get("matrix", {})) == 14
        and float(d.get("lambda", -1)) == 0.5
    )
    detail = f"n_cases={d.get('n_cases')} lambda={d.get('lambda')} alpha={d.get('alpha')}"
check("task3_covis_prior_baked", ok, detail)

# 4. FP-suppress configs + T3 head ----------------------------------------------
for f in ("t2_ft_ens_fpsuppress_fullres.json", "t1_topk_fpsuppress_fullres.json"):
    check(f"fp_config::{f}", (TIGER / f).is_file())
check("t3_head::best.pt", (TIGER / "t3_head" / "best.pt").is_file())

# 5. official exact-RGB registries -----------------------------------------------
try:
    sys.path.insert(0, str(TIGER / "challenge_repo"))
    from metrics.classes import CLASSES
    from metrics.classes_merged import CLASSES_MERGED
    from metrics.classes_stations import CLASSES_STATIONS

    check("registry::fine_31", len(CLASSES) == 31, f"{len(CLASSES)} classes")
    check("registry::merged_16", len(CLASSES_MERGED) == 16, f"{len(CLASSES_MERGED)} classes")
    check("registry::stations_14", len(CLASSES_STATIONS) == 14, f"{len(CLASSES_STATIONS)} stations")
except Exception as exc:  # noqa: BLE001
    check("registry_import", False, repr(exc))

# 6. offline env -----------------------------------------------------------------
for k, v in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1")):
    check(f"offline_env::{k}", os.environ.get(k) == v, os.environ.get(k, "<unset>"))

print("=== BUILD ASSERTIONS:", "PASS ===" if not FAIL else f"FAIL ({len(FAIL)}) ===")
sys.exit(1 if FAIL else 0)
