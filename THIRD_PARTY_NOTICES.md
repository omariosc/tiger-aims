# Third-party notices

This repository is MIT licensed (see [LICENSE](LICENSE)) with the exceptions below.
No model weights and no challenge data are distributed here.

## Vendored code

`nnunet_trainers/cbdice_tiger.py` is adapted from
[PengchengShi1220/cbDice](https://github.com/PengchengShi1220/cbDice), licensed under
**Apache-2.0**, and remains under that licence. It is a self-contained port of
`loss/cbdice_loss.py` and `loss/soft_skeleton.py` that replaces the MONAI/cuCIM Euclidean
distance transform with a SciPy one; the changes are documented in the file's own header.
cbDice was one of the topology-aware losses evaluated for the thin-structure tier and was
not adopted for the submitted configuration.

## Referenced, not included

- **nnU-Net v2** (`nnunetv2==2.8.0`), Apache-2.0. Installed by `docker/Dockerfile` from PyPI.
  `nnunet_trainers/nnUNetTrainerTverskyBoundary.py` is original work written against its
  trainer API and is copied into the installed package at image-build time.
- **SAM 2.1**, Apache-2.0. Evaluated as a promptable refiner and not adopted; no SAM code or
  weights are present here.
- **DINOv3** (Meta). Referenced by three Task-3 experiment scripts through `timm`. The DINOv3
  weights are released under Meta's own licence, which is not MIT and cannot be relicensed, so
  nothing from DINOv3 is redistributed here. That branch was not part of the submitted
  configuration either.
- **TIGER SQ-AI challenge data and evaluation code**, which remain under the challenge's own
  terms and are not redistributed. `docker/covis_matrix.json` holds conditional probabilities
  estimated from the training labels; it is a 14x14 table of aggregate statistics, not data.
