# TIGER SQ-AI 2026 submission (team AIMS Lab)

Segmentation of the thoracic field during lymphadenectomy for minimally invasive
esophagectomy, and lymph node station visibility, submitted to the
[TIGER SQ-AI 2026 challenge](https://www.synapse.org/Synapse:syn74209386).

Tasks 1 and 2 are semantic segmentation at two label granularities. Task 3 asks which of 14
lymph node stations are visible in a frame. The two are scored on different quantities, and the
scoring is what determined the design.

## Results

Measured on 75 held-out frames from centres 2, 3, 4, 6 and 7, none of which the incumbent models
had trained on, using the organisers' scoring code. No official score exists at time of writing.

| task | quantity | incumbent | retrained |
|---|---|---|---|
| Task 2, 31 classes | w=3 Dice | 0.3733 | **0.7372** |
| of which, from multi-centre data | | | +0.3253 |
| of which, from area filtering | | | +0.0386 |
| Task 1, 16 classes | w=3 Dice | retained | retrain lost 0.0140 |
| Task 3, 14 stations | macro-F1 | | +0.1152 from the improved masks |

Earlier internal figures of 0.552 (Task 1) and 0.702 to 0.709 (Task 2) were measured on the
single-centre split available before the August multi-centre release. They differ in evaluation
data, not only in model, and are not comparable with the table above.

## Method

**Tasks 1 and 2.** nnU-Net v2 in its 2D configuration, since every input is a single video frame.
Task 1 uses TopK cross-entropy on the hardest 10% of pixels plus Tversky; Task 2 uses
cross-entropy plus Focal-Tversky (alpha 0.3, beta 0.7, gamma about 1.33) over three seeds averaged
before one argmax. The split is a measured result: the same Focal-Tversky loss raised Task 2 w=3
Dice from 0.464 to 0.498 and lowered Task 1 from 0.349 to 0.273.

Mirroring is disabled in training and at test time. Thoracic anatomy is laterally asymmetric and
left and right structures are separate classes, so a horizontal flip maps the right recurrent
laryngeal nerve onto the left, inverting labels on exactly the triple-weighted tier.

False positives are suppressed by a training-free per-class connected-component area filter. Dice
scores a class 0.0 as soon as one false component appears on a frame where that class is absent;
on such frames our false components reach 28 px while genuine masks exceed 116 px, so one
threshold separates them.

**Task 3.** Two branches. A segmentation-conditioned head reads presence and log-area for 18
boundary structures from the Task-2 output, 36 features through a small MLP to 14 station logits.
A 14x14 co-visibility prior holds P(station s visible | station a), estimated from training-label
counts with additive shrinkage (alpha 2.0, fixed in advance) toward the pooled marginal; it has no
learned parameters. The two are blended at a fixed lambda of 0.5.

The prior is keyed on the acquisition station named in the input filename. The challenge data
contract specifies that filename format and guarantees it arrives unchanged, so this is documented
input metadata rather than inspection of test labels, but it is not pixel evidence. The image-only
head scores macro-F1 0.6276 and the submitted blend 0.7128.

The blend is submitted even though the prior alone scores higher (0.7307), because lambda was
fixed before any measurement and moving it afterwards would be selection on the evaluation set.
The multi-centre release supported that: leave-one-centre-out macro-F1 for the prior falls to
0.5998 and the worst pairwise drift between per-centre matrices is 0.3737.

The full write-up, including the refuted upgrades and the open risks, is in
[report/Tiger_Writeup_AIMS.pdf](report/Tiger_Writeup_AIMS.pdf).

## Layout

```
report/Tiger_Writeup_AIMS.pdf  the 3-page submission report
docker/inference.py            submission entry point, all three tasks
docker/Dockerfile              container definition
docker/contract_test.sh        interface conformance checks
docker/build_checks.py         artifact content gates run at build time
docker/stage_build_context.sh  assembles the build context
docker/fp_suppress.py          connected-component area filter
docker/covis_matrix.json       14x14 co-visibility prior, baked in, no runtime download
docker/*fpsuppress*.json       tuned per-class area thresholds
nnunet_trainers/               custom nnU-Net loss trainers (Focal-Tversky, cbDice)
experiments/                   training, scoring, ablations and the multi-centre analyses
```

Task-3 analyses referenced in the report:
`t3_prior_multicentre.py` (leave-one-centre-out and pairwise drift),
`t3_headtohead_centres.py` (ranking against calibration),
`t3_threshold_calibration.py`, `adoption_rule_score*.py` (the pre-registered adoption rule).

## Running

No weights are distributed here. The container expects an `nnUNet_results` tree staged into the
build context by `docker/stage_build_context.sh`.

```bash
bash docker/stage_build_context.sh /path/to/context
docker build -t tiger-aims -f docker/Dockerfile /path/to/context
docker run --rm --gpus all -v /path/to/input:/input:ro -v /path/to/output:/output tiger-aims
```

Development used NVIDIA L40S GPUs on the University of Leeds Aire cluster. SLURM scripts carry
cluster-specific paths with the account name replaced by `USERNAME`; edit them for your site.

## Licence

MIT, see [LICENSE](LICENSE). Vendored and referenced third-party components keep their own terms,
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Challenge data and evaluation code are not
redistributed.

## Contact

Omar Choudhry, <O.Choudhry@leeds.ac.uk>
Artificial Intelligence in Medicine and Surgery Group, School of Computer Science,
University of Leeds, Leeds LS2 9JT, UK.
ORCID [0000-0003-4434-3550](https://orcid.org/0000-0003-4434-3550)

Funded by UKRI EPSRC grant EP/S024336/1.
