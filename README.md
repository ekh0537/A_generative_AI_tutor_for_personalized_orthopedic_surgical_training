# A generative AI tutor for personalized orthopedic surgical training with 3D-printed anatomical models

AG-Tri-CT couples fragment-aware anatomical segmentation, evidence-bound orthopedic tutoring, and cold-start knowledge tracing through one 800-concept, 256-dimensional AO/OTA routing space. The package contains the model components, joint objectives, dataset readers, cohort simulator, statistical procedures, experiment settings, and command-line utilities used for the computational retrospective evaluation.

## Installation

Python 3.11 and CUDA 12.1 are the supported runtime.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

The equivalent Conda environment is created with:

```bash
conda env create -f environment.yml
conda activate ag-tri-ct
pip install -e .
```

The container image is built with:

```bash
docker build -t ag-tri-ct:0.1.0 .
```

## Data

All canonical, currently reachable dataset pages and their distribution terms are listed in `dataset_links.txt`. The list was checked on 8 August 2026. OAI requires NIH controlled-public access, MURA uses the Stanford academic terms, and several education corpora restrict commercial use. Access approval and dataset-specific terms remain the operator's responsibility.

The imaging stream uses TotalSegmentator-v2 as the primary anatomical corpus, VerSe and Lumbar Spine MRI for spine transfer, OAI for knee MRI transfer, MURA and GRAZPEDWRI-DX for radiograph transfer, and MedShapeNet for geometric priors. The tutoring stream uses MedQA, MedMCQA, and PubMedQA after ICD-10 filtering. The tracing stream uses ASSISTments, EdNet-KT1, OULAD, Junyi Academy, and KDD EDM Cup 2010.

Prepared volume manifests are CSV files with `identifier`, `volume_path`, `mask_path`, `concept_index`, and `fold` fields. Interaction data are newline-delimited JSON objects with `learner_id`, `skill`, `correct`, and `timestamp`. Raw datasets are intentionally excluded.

## Configuration

`configs/main.yaml` records joint training at 8 A100 80 GB GPUs, bf16 precision, batch size 16 per process, 200 epochs, AdamW-compatible learning rate (5×10⁻⁴), cosine scheduling, and the 0.5/0.3/0.2/0.3 loss schedule. `configs/segmentation.yaml` records the branch-specific 200-epoch, batch-size-2, (1×10⁻⁴) protocol. These settings are separate because the branch fine-tune and joint optimization use different reported values.

Configuration integrity can be checked with:

```bash
ag-tri-ct validate-config configs/main.yaml
```

The reported effective joint batch is 128 across eight processes. The main volume shape is 200×512×512. Retrieval always returns five chunks, and online knowledge-tracing adaptation accepts no more than ten prior events.

## Training

Launch the distributed run on one eight-GPU node:

```bash
bash scripts/train.sh
```

The complete computational program uses about 10,000 A100 GPU-hours: approximately 2,000 for fragment-aware segmentation, 3,000 for tutor tuning, 1,500 for knowledge-tracing meta-learning, and 3,500 for joint optimization and ablations. Joint training requires about 60 GB device memory per process and 1 TB system memory for the reported data pipeline.

The model supplies atomic state writes, deterministic seed restoration, gradient clipping, cosine scheduling, bf16-compatible tensor operations, and distributed wrapping. Raw corpus acquisition and acceptance of third-party licenses occur outside the training command.

## Evaluation

Generate a protocol-conformant public-data-derived cohort:

```bash
python -m ag_tri_ct.cli simulate --trainees 200 --events 10 --seed 2026
```

The evaluation grid uses cohort sizes 200, 500, and 1,000; ten trajectories per seed; five fixed seeds; 10,000 paired bootstrap resamples; and Holm–Bonferroni control at familywise α=0.05. The statistical module also includes Benjamini–Hochberg control for prospective secondary endpoints.

Expected headline targets are mean Dice 0.957 on the 104-structure anatomical evaluation, AO/OTA-C Dice 0.722, strict evidence binding for every generated span, musculoskeletal MedQA accuracy 0.886–0.920, long-form unsupported-claim rate at most 0.06, and cold-start AUC 0.728 at simulated N=200. The composite score uses weights 0.3/0.3/0.3/0.1 for segmentation, text accuracy, tracing AUC, and one minus unsupported-claim rate.

## Verification

```bash
pytest -q
ruff check .
mypy --strict code/ag_tri_ct
```

The test suite covers configuration invariants, ontology dimensions, graph regularization, segmentation tensor contracts, Dice behavior, AUC ordering, evidence binding, knowledge-tracing probabilities, mesh manifold checks, multiple-comparison procedures, optimizer loss descent, and atomic state restoration.

## Package map

`code/ag_tri_ct/segmentation.py` contains the volumetric encoder, FiLM conditioning, auxiliary fragment classifier, and implicit signed-distance decoder. `retrieval.py` and `generation.py` contain hybrid sparse-dense retrieval and generation-time evidence binding. `knowledge_tracing.py` contains the transformer learner state, ontology alignment, and one-layer hierarchy propagation. `losses.py`, `metrics.py`, and `statistics.py` define the scientific objectives and analysis rules. `pipeline.py` joins the three branches through the shared routing tensor.

## License

The software is released under the MIT License. Dataset licenses and access controls are independent and remain in force.
