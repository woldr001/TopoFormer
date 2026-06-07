# How-To: Train, Validate, and Test TopoFormer-MF on the MSU HPCC

This guide explains how to use the `protein_function/` pipeline in the
`protein-motion-topology` branch to predict **Molecular Function (MF)**
Gene Ontology terms from a **protein structural ensemble**.

It is written for our lab's workflow on the **MSU HPCC** and is intended to help a new team
member go from:

**PDB ensemble + NMA-PCA profiles + MF annotations → topology + sequence features → trained model → validated/tested model → single-protein prediction**

---

## 1. What this branch does

The branch adds a new `protein_function/` package that adapts the original TopoFormer
protein–ligand pipeline into a **protein function prediction** workflow.

### Core idea

Instead of capturing cross-distances between a **protein** and a small-molecule **ligand**, we
capture cross-distances between:

1. **Motion residues** — the ~20% of residues with the highest GNM/ANM slow-mode fluctuations
   and/or PCA displacement scores (identified via the
   [Ensemble\_NMA-PCA pipeline](https://github.com/woldr001/Ensemble_NMA-PCA))
2. **Static residues** — all other residues in the protein

This directly mirrors the original protein–ligand sentinel-distance scheme: intra-motion and
intra-static atom-pair distances are set to a large sentinel value so they never appear in
the Persistent Laplacian filtration; only **motion↔static cross-distances** drive the topology.

For each protein, **10 structural conformations** are processed individually and the resulting
topology tensors are aggregated as **mean + std** across the ensemble, capturing both the
average topology and the geometric variability across the conformational space.

### Output topology tensor shape

```
[12, 200, 121]
 │    │    │
 │    │    └── element-pair combination index  (11 motion × 11 static = 121)
 │    └─────── filtration distance step        (0.0 → 19.9 Å, 200 steps of 0.1 Å)
 └──────────── channel                         (6 spectral stats × mean + 6 × std = 12)
```

### Implemented components

| File | Purpose |
|------|---------|
| `topo_extraction/protein_only_topo_embedding.py` | Single-PDB fallback extractor → `[6, 200, 15]` |
| `topo_extraction/ensemble_motion_topo_embedding.py` | **Primary extractor**: motion-guided ensemble → `[12, 200, 121]` |
| `scripts/precompute_topo_features.py` | Batch topology extraction (`--mode ensemble_motion` or `protein_only`) |
| `scripts/precompute_seq_features.py` | Batch ESM-2 + ProtTrans embeddings from FASTA |
| `models/mini_topt_config.py` | `get_ensemble_motion_topt_config()` for `[12, 200, 121]` input |
| `models/modeling_topo_function.py` | Fusion model: topology encoder + ESM proj + ProtTrans proj + MF classifier |
| `data/protein_function_dataset.py` | Dataset loader for `.npy` features and GO labels |
| `evaluation/metrics.py` | Fmax and AUPR metrics |
| `training/train_mf_prediction.py` | HF Trainer-based training pipeline |
| `predict.py` | End-to-end inference from a trained model |

---

## 2. Why each feature stream is needed

### A. Motion-guided ensemble topology

The topology branch is the primary novel contribution.

It captures **3D structural organization** that pure sequence models miss:

- how atom groups are arranged in space across conformations
- multi-scale geometric relationships between mobile and stable regions
- patterns related to active sites, hinges, and functional structural motifs

The motion-guided cross-pair scheme focuses this signal specifically on the **interface
between flexible and rigid parts of the protein**, which is where catalytic activity,
binding, and allosteric communication tend to occur.

### B. ESM-2 embeddings (1280-dim)

Rich learned representation of sequence context from evolutionary information.
Model: `facebook/esm2_t33_650M_UR50D`.

### C. ProtTrans T5 embeddings (1024-dim)

A second sequence representation family with different pretraining history, providing
complementary information. Model: `Rostlab/prot_t5_xl_uniref50`.

### D. Fusion

Molecular function depends on both **what the sequence encodes** and **how the structure
moves**. The fusion model concatenates all three representations:

```
CLS token (topology): [B, 256]
ESM projected:        [B, 512]
ProtTrans projected:  [B, 512]
                       ───────
Concatenated:         [B, 1280]  →  FC(512)  →  GELU  →  logits [B, N_mf]
```

---

## 3. Files you need before running the pipeline

### 3.1 Project directory layout

```text
project_root/
├── pdbs/
│   ├── P12345/
│   │   ├── cluster_repr_0.pdb
│   │   ├── cluster_repr_1.pdb
│   │   └── ...   (10 conformation PDB files per protein)
│   └── Q8ABC1/
│       └── ...
├── nma_pca/
│   ├── P12345/
│   │   ├── anm_gnm_results.npz
│   │   └── pca_results.npz
│   └── Q8ABC1/
│       └── ...
├── sequences.fasta
├── mf_annotations.tsv
├── train_ids.txt
├── val_ids.txt
├── test_ids.txt
└── runs/
```

### 3.2 PDB ensemble files

Each protein needs a **subdirectory** containing its 10 conformations.

Recommended naming (the scripts search these patterns in order):

1. `cluster_repr_*.pdb`
2. `conf_*.pdb`
3. `model_*.pdb`
4. `frame_*.pdb`
5. Any `*.pdb` (sorted, first 10 used)

Good sources for conformations:
- AlphaFold2 with different random seeds or `--num_models 5 --num_seeds 2`
- MD simulation frames sampled at equal time intervals
- Normal-mode-generated conformations from ProDy
- Multiple crystal form depositions

### 3.3 NMA-PCA profile files

Each protein's `nma_pca/<protein_id>/` directory must contain the output of the
Ensemble\_NMA-PCA pipeline (see Step 2 below):

- `anm_gnm_results.npz` — contains `gnm_flucts_mean` `[N_modes, N_residues]` and `resids`
- `pca_results.npz` — contains `pc1_profile` `[N_residues]`, `pc2_profile`, and `resids`

If these files are absent the extractor falls back to treating **all residues as motion
residues**, reducing to a standard intra-protein topology with no motion-guided focus.

### 3.4 Sequence FASTA

Headers must match protein IDs exactly:

```fasta
>P12345
MSEQUENCE...
>Q8ABC1
MSEQUENCE...
```

### 3.5 Annotation file (`mf_annotations.tsv`)

Long-format TSV with three columns:

```tsv
protein_id	go_term	namespace
P12345	GO:0004672	molecular_function
P12345	GO:0005524	molecular_function
Q8ABC1	GO:0016491	molecular_function
```

Only `namespace == molecular_function` rows are kept. Terms with fewer than 50 training
annotations are dropped automatically.

### 3.6 Split ID files

One protein ID per line. Use sequence-cluster splits (30% identity via CD-HIT) or temporal
splits. **Never split a cluster across train/val/test.**

---

## 4. Recommended dataset strategy

| Source | Use |
|--------|-----|
| **UniProtKB/Swiss-Prot** | Reviewed proteins + manually-curated MF GO annotations |
| **AlphaFold DB** | Structural conformations for proteins without experimental structures |
| **CAFA3 splits** | Optional benchmark splits for comparison to published methods |

**Keep only experimental MF evidence** — exclude `IEA` (electronically inferred) for the
main benchmark training set. Recommended evidence codes to keep: `EXP`, `IDA`, `IPI`,
`IMP`, `IGI`, `IEP`, `TAS`, `IC`.

A practical starting point is **Swiss-Prot proteins with AlphaFold structures**, running the
Ensemble\_NMA-PCA pipeline on each to generate conformations and motion profiles.

---

## 5. Recommended train/val/test split strategy

Do **not** use a naive random split. Closely related proteins split across train and test
will inflate apparent performance.

**Recommended**: cluster at **30% sequence identity** (CD-HIT), then assign entire clusters
to train/val/test without overlap.

For benchmarking against published methods, use the official CAFA3 temporal split.

---

## 6. Set up the software environment on the MSU HPCC

The `topoformer_mf` conda environment is pre-created at
`/mnt/home/woldring/.conda/envs/topoformer_mf`. On the MSU HPCC the system
Miniforge3 module overrides `conda activate`, so activate by prepending the
environment's `bin/` directly:

```bash
export PATH="/mnt/home/woldring/.conda/envs/topoformer_mf/bin:$PATH"

# Verify the right Python is active
which python   # should show .../topoformer_mf/bin/python

# Install any missing packages into the environment
python -m pip install "transformers==4.24.0"   # CRITICAL — see below
python -m pip install prody                    # required for NMA-PCA step
```

> **Why `transformers==4.24.0`?**
> `modeling_topt.py` imports `find_pruneable_heads_and_indices` from
> `transformers.pytorch_utils`, a symbol removed in newer releases.

All SLURM batch scripts in this repo already include this PATH export at the top,
so you do not need to activate the environment manually before submitting jobs.

### Quick sanity check

```bash
cd /mnt/research/woldring_lab/TopoFormer-MF/TopoFormer
python - <<'PY'
import torch, transformers, pandas, numpy, scipy, sklearn
from protein_function.models.mini_topt_config import get_ensemble_motion_topt_config
from protein_function.models.modeling_topo_function import TopoFunctionModel

config = get_ensemble_motion_topt_config()
model = TopoFunctionModel(config, num_mf_labels=100)
out = model(torch.randn(2, 12, 200, 121), torch.randn(2, 1280), torch.randn(2, 1024))
print('logits shape:', out.logits.shape)   # → torch.Size([2, 100])
print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())
print('transformers', transformers.__version__)
PY
```

---

## Step 1: Curate the dataset

Build a consistent dataset where every protein has:

1. a valid protein ID
2. a PDB ensemble directory with 10 conformations
3. a FASTA sequence
4. NMA-PCA profile files
5. at least one retained MF GO annotation
6. membership in exactly one split

### Recommended manifest file

Create a master CSV for QA:

| Column | Example |
|--------|---------|
| `protein_id` | P12345 |
| `sequence_length` | 342 |
| `n_pdbs` | 10 |
| `has_nma_pca` | True |
| `has_topology` | False |
| `has_esm` | False |
| `has_prottrans` | False |
| `split` | train |
| `n_mf_terms` | 3 |

Running QA against this manifest before launching large HPCC jobs saves significant
debugging time.

---

## Step 2: Run the NMA-PCA pipeline

This step is **required** for the motion-guided topology extraction. It produces the
GNM/PCA profiles that determine which residues are treated as "motion" atoms.

The NMA-PCA script is built into this repository — no external pipeline is needed.
It requires ProDy (`python -m pip install prody`).

### Single protein (test run)

```bash
cd /mnt/research/woldring_lab/TopoFormer-MF/TopoFormer
export PATH="/mnt/home/woldring/.conda/envs/topoformer_mf/bin:$PATH"

python protein_function/scripts/run_nma_pca.py \
    --pdb_dir      /mnt/research/nodes/giacomo/asam_ensembles/protein_function_prediction/v0/sampling/A0A010/clustering.nodes_pipeline_v1 \
    --output_dir   /mnt/research/woldring_lab/TopoFormer-MF/nma_pca/A0A010 \
    --n_conformers 10 \
    --n_slow_modes 3 \
    --n_pcs        2 \
    --gnm_cutoff   7.5
```

Expected outputs in `/mnt/research/woldring_lab/TopoFormer-MF/nma_pca/A0A010/`:

```
anm_gnm_results.npz    ← gnm_flucts_mean [n_slow_modes, N_residues], resids [N_residues]
pca_results.npz        ← pc1_profile [N_residues], pc2_profile [N_residues], resids
```

### Batch on HPCC (SLURM array job)

The launcher reads `datasets/all_ids.txt` and submits a chunked array job (5 proteins
per task → ~489 tasks for 2,441 proteins), staying within SLURM QOS limits:

```bash
bash protein_function/scripts/submit_nma_pca.sh
# Prints: "Submitting NMA-PCA array job for 2441 proteins as 489 tasks (CHUNK_SIZE=5) ..."
```

Monitor with:
```bash
squeue -u $USER
# Check a task log:
tail -f /mnt/research/woldring_lab/TopoFormer-MF/logs/nma_pca_<JOBID>_1.out
```

### QA check

```bash
python - <<'PY'
import os, numpy as np
nma_dir = '/mnt/research/woldring_lab/TopoFormer-MF/nma_pca'
proteins = [d for d in os.listdir(nma_dir) if os.path.isdir(os.path.join(nma_dir, d))]
ok, missing_gnm, missing_pca = 0, [], []
for pid in proteins:
    d = os.path.join(nma_dir, pid)
    gnm = os.path.exists(os.path.join(d, 'anm_gnm_results.npz'))
    pca = os.path.exists(os.path.join(d, 'pca_results.npz'))
    if gnm and pca:
        ok += 1
    else:
        if not gnm: missing_gnm.append(pid)
        if not pca: missing_pca.append(pid)
print(f'OK: {ok} | missing gnm: {len(missing_gnm)} | missing pca: {len(missing_pca)}')
PY
```

---

## Step 3: Pre-compute topology features

### Why precompute

Topology extraction is CPU-intensive and deterministic. Precompute once; reuse every epoch.

### What the script produces

For each protein: one `.npy` file of shape **`(12, 200, 121)`** (ensemble motion mode).

### Run command (SLURM array job — recommended)

Topology extraction is CPU-intensive (O(N³) eigenvalue computation). Use the array
job launcher to spread work across multiple cluster nodes:

```bash
bash protein_function/scripts/submit_topo_features.sh
# Prints: "Submitting topo-features array job for 2441 proteins as 489 tasks (CHUNK_SIZE=5) ..."
```

Each task processes 5 proteins in parallel on 5 CPUs. The job reads paths from
`sbatch_topo_features_array.sh`:
- PDB ensembles: `/mnt/research/nodes/giacomo/asam_ensembles/protein_function_prediction/v0/sampling`
- NMA-PCA profiles: `/mnt/research/woldring_lab/TopoFormer-MF/nma_pca`
- Output: `/mnt/research/woldring_lab/TopoFormer-MF/topo_features`

To scale to larger datasets (e.g. 50k proteins), increase CHUNK_SIZE:
```bash
CHUNK_SIZE=20 bash protein_function/scripts/submit_topo_features.sh
```

### Key arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `--mode` | `protein_only` | Use `ensemble_motion` for the full motion-guided pipeline |
| `--nma_pca_dir` | *(required for ensemble_motion)* | Root dir with NMA-PCA output subdirectories |
| `--n_conformers` | 10 | Max PDB files per protein |
| `--top_motion_pct` | 0.20 | Top 20% of residues by GNM+PCA score treated as "motion" |
| `--fluctuation_sources` | `gnm pca` | Which profiles to combine for motion scoring |
| `--n_slow_modes` | 3 | Number of slowest GNM modes to sum |
| `--motion_field` | None | If set, restrict static atoms to within N Å of motion atoms (speeds up large proteins) |
| `--ensemble_aggregation` | `mean_std` | `mean_std` → 12 channels; `mean_only` → 6 channels |
| `--overwrite` | False | Re-compute even if output file exists |

### QA check

```bash
python - <<'PY'
import os, numpy as np
feat_dir = '/mnt/research/woldring_lab/TopoFormer-MF/topo_features'
files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
print(f'n_files: {len(files)}')
bad = []
for f in files:
    arr = np.load(os.path.join(feat_dir, f))
    if arr.shape != (12, 200, 121):
        bad.append((f, arr.shape))
    elif not np.isfinite(arr).all():
        bad.append((f, 'has inf/nan'))
print(f'Bad files: {len(bad)}')
for b in bad[:10]:
    print(' ', b)
PY
```

---

## Step 4: Pre-compute sequence embeddings (ESM-2 + ProtTrans)

### Why precompute

These models are GPU-expensive. Precompute once and reuse.

### Embedding specifications

| Model | Identifier | Output dim |
|-------|-----------|-----------|
| ESM-2 650M | `facebook/esm2_t33_650M_UR50D` | **1280** |
| ProtTrans T5 | `Rostlab/prot_t5_xl_uniref50` | **1024** |

### Run command (HPCC — use the provided sbatch script)

```bash
sbatch protein_function/scripts/sbatch_seq_features.sh
```

This script (1 GPU, 8 CPUs, 64 GB, 8h) handles:
- PATH activation for the conda environment
- HuggingFace model cache redirected to node-local `$TMPDIR` (avoids slow networked I/O)
- Reads from `datasets/sequences.fasta`
- Writes to `/mnt/research/woldring_lab/TopoFormer-MF/esm_features/` and `prottrans_features/`

Manual invocation (for testing):
```bash
export PATH="/mnt/home/woldring/.conda/envs/topoformer_mf/bin:$PATH"
python protein_function/scripts/precompute_seq_features.py \
    --fasta_file           /mnt/research/woldring_lab/TopoFormer-MF/datasets/sequences.fasta \
    --esm_output_dir       /mnt/research/woldring_lab/TopoFormer-MF/esm_features \
    --prottrans_output_dir /mnt/research/woldring_lab/TopoFormer-MF/prottrans_features \
    --esm_model            facebook/esm2_t33_650M_UR50D \
    --prottrans_model      Rostlab/prot_t5_xl_uniref50 \
    --batch_size           8 \
    --device               cuda
```

### QA check

```bash
python - <<'PY'
import os, numpy as np
for d, expected_dim in [
    ('/mnt/research/woldring_lab/TopoFormer-MF/esm_features',       1280),
    ('/mnt/research/woldring_lab/TopoFormer-MF/prottrans_features', 1024),
]:
    files = sorted([f for f in os.listdir(d) if f.endswith('.npy')])
    bad = [(f, np.load(os.path.join(d, f)).shape)
           for f in files if np.load(os.path.join(d, f)).shape != (expected_dim,)]
    print(f'{os.path.basename(d)}: {len(files)} files, {len(bad)} wrong-shape')
    for b in bad[:5]:
        print(' ', b)
PY
```

---

## Step 5: Train the MF predictor

### What the training script does

1. Reads train/val/test IDs and builds the MF label vocabulary
2. Drops terms with fewer than 50 training annotations
3. Fits the topology scaler **on training data only** (prevents leakage)
4. Constructs datasets and DataLoaders
5. Trains with cosine LR schedule and warmup
6. Evaluates after every epoch using **Fmax** and **AUPR**
7. Early-stops at patience=10 epochs on validation Fmax
8. Saves model weights, `go_terms.txt`, and `topo_scaler.pkl` to the output directory

### Basic training command

```bash
python protein_function/training/train_mf_prediction.py \
    --topo_dir          /mnt/research/woldring_lab/TopoFormer-MF/topo_features \
    --esm_dir           /mnt/research/woldring_lab/TopoFormer-MF/esm_features \
    --prottrans_dir     /mnt/research/woldring_lab/TopoFormer-MF/prottrans_features \
    --label_file        /mnt/research/woldring_lab/TopoFormer-MF/datasets/mf_annotations.tsv \
    --train_ids_file    /mnt/research/woldring_lab/TopoFormer-MF/datasets/train_ids.txt \
    --val_ids_file      /mnt/research/woldring_lab/TopoFormer-MF/datasets/val_ids.txt \
    --topo_feature_mode ensemble_motion \
    --output_dir        /mnt/research/woldring_lab/TopoFormer-MF/runs/mf_model_run1 \
    --num_train_epochs  50 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size  64 \
    --learning_rate     1e-4 \
    --warmup_ratio      0.05 \
    --weight_decay      0.01
```

### Recommended HPCC batch script

Save as `sbatch_train_mf.sh`:

```bash
#!/bin/bash --login
#SBATCH --job-name=mf_train
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/mnt/scratch/%u/topoformer_mf/logs/%x_%j.out
#SBATCH --error=/mnt/scratch/%u/topoformer_mf/logs/%x_%j.err

set -euo pipefail
export PATH="/mnt/home/woldring/.conda/envs/topoformer_mf/bin:$PATH"

mkdir -p /mnt/research/woldring_lab/TopoFormer-MF/logs
mkdir -p /mnt/research/woldring_lab/TopoFormer-MF/runs

JOB_TMP=${TMPDIR:-/tmp/woldring_mf_${SLURM_JOB_ID}}
mkdir -p "$JOB_TMP"
export HF_HOME="$JOB_TMP/hf"
export TRANSFORMERS_CACHE="$JOB_TMP/hf/transformers"
export HUGGINGFACE_HUB_CACHE="$JOB_TMP/hf/hub"

python /mnt/research/woldring_lab/TopoFormer-MF/TopoFormer/protein_function/training/train_mf_prediction.py \
    --topo_dir          /mnt/research/woldring_lab/TopoFormer-MF/topo_features \
    --esm_dir           /mnt/research/woldring_lab/TopoFormer-MF/esm_features \
    --prottrans_dir     /mnt/research/woldring_lab/TopoFormer-MF/prottrans_features \
    --label_file        /mnt/research/woldring_lab/TopoFormer-MF/datasets/mf_annotations.tsv \
    --train_ids_file    /mnt/research/woldring_lab/TopoFormer-MF/datasets/train_ids.txt \
    --val_ids_file      /mnt/research/woldring_lab/TopoFormer-MF/datasets/val_ids.txt \
    --topo_feature_mode ensemble_motion \
    --output_dir        /mnt/research/woldring_lab/TopoFormer-MF/runs/mf_model_run1 \
    --num_train_epochs  50 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size  64 \
    --learning_rate     1e-4 \
    --warmup_ratio      0.05 \
    --weight_decay      0.01
```

### Hyperparameter guidance

| Parameter | Recommended range | Notes |
|-----------|-----------------|-------|
| `learning_rate` | 5e-5 – 2e-4 | Lower if loss oscillates |
| `batch_size` | 32–128 | Larger → more stable gradients for multi-label |
| `num_train_epochs` | 50–100 | Early stopping limits actual epochs |
| `warmup_ratio` | 0.05–0.10 | Longer warmup helps the topology encoder |
| `fusion_dropout` | 0.2–0.4 | Increase if val Fmax stagnates while train rises |

### Optional: two-stage fine-tuning

For better convergence from a pretrained TopoFormer checkpoint:

**Stage 1** (10 epochs, frozen topology encoder):
```bash
... --freeze_topo_encoder true --num_train_epochs 10 --output_dir ./runs/stage1
```

**Stage 2** (full fine-tuning, lower LR):
```bash
... --resume_from_checkpoint ./runs/stage1/checkpoint-best \
    --freeze_topo_encoder false \
    --learning_rate 3e-5 \
    --num_train_epochs 90 \
    --output_dir ./runs/stage2
```

---

## Step 6: Monitor training

### TensorBoard

```bash
tensorboard --logdir /mnt/research/woldring_lab/TopoFormer-MF/runs
```

Key metrics:
- `eval/fmax` — primary model selection metric (higher is better)
- `eval/aupr` — secondary metric
- `train/loss` vs `eval/loss` — watch for overfitting gap

### Symptoms and fixes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Loss NaN from epoch 1 | LR too high | Halve `--learning_rate` |
| `eval/fmax` ≤ 0.10 after 20 epochs | Topo features all zeros | Check `--topo_dir` paths and array shapes |
| `eval/fmax` flat at ~0.35, train → 0.70 | Overfitting | Increase dropout, reduce hidden size |
| CUDA OOM | Batch too large | Halve `--per_device_train_batch_size` |
| `transformers` import error | Wrong version | `pip install "transformers==4.24.0"` |

---

## Step 7: Evaluate on the held-out test set

Use the test set only after model configuration is finalized.

### Built-in evaluation

```bash
python protein_function/training/train_mf_prediction.py \
    ... \
    --do_train false \
    --do_eval  true \
    --do_predict true \
    --test_ids_file /mnt/research/woldring_lab/TopoFormer-MF/datasets/test_ids.txt \
    --resume_from_checkpoint /mnt/research/woldring_lab/TopoFormer-MF/runs/mf_model_run1/checkpoint-best
```

Outputs:
- `test_results.json` — Fmax, AUPR, per-term AP
- `test_predictions.npy` — raw logit matrix `[N_test, N_terms]`

### Publication-grade evaluation

For comparison to published methods, also evaluate using the external CAFA metric scripts.
A validation Fmax of **0.50–0.60** is expected on Swiss-Prot MF with good splits and
~50k training proteins.

---

## Step 8: Run single-protein inference from a PDB ensemble

### What is needed

The model directory must contain:
- model weights (`pytorch_model.bin` or HF checkpoint)
- `go_terms.txt`
- `topo_scaler.pkl`

### Command

```bash
python protein_function/predict.py \
    --model_dir         /mnt/research/woldring_lab/TopoFormer-MF/runs/mf_model_run1 \
    --protein_id        P99999 \
    --pdb_dir           /mnt/research/woldring_lab/TopoFormer-MF/inference/pdbs/P99999 \
    --nma_pca_dir       /mnt/research/woldring_lab/TopoFormer-MF/inference/nma_pca/P99999 \
    --esm_npy           /mnt/research/woldring_lab/TopoFormer-MF/inference/esm/P99999.npy \
    --prottrans_npy     /mnt/research/woldring_lab/TopoFormer-MF/inference/prottrans/P99999.npy \
    --threshold         0.3 \
    --top_k             10
```

---

## Quality-control checklist before launching a long training run

### Data checks

- [ ] every train/val/test ID has a matching FASTA entry
- [ ] every train/val/test ID has a `pdbs/<id>/` directory with ≥1 PDB file
- [ ] every train/val/test ID has `nma_pca/<id>/anm_gnm_results.npz` and `pca_results.npz`
- [ ] every train/val/test ID has at least one retained MF annotation
- [ ] no ID appears in more than one split

### Feature checks

- [ ] topology `.npy` arrays have shape `(12, 200, 121)`
- [ ] no topology arrays contain `inf` or `NaN`
- [ ] ESM `.npy` arrays have shape `(1280,)`
- [ ] ProtTrans `.npy` arrays have shape `(1024,)`
- [ ] counts of topology / ESM / ProtTrans files match expected protein counts

### Training checks

- [ ] `--topo_feature_mode ensemble_motion` is set
- [ ] output directory is empty or intentionally continued
- [ ] scaler is fit from training data only
- [ ] test set remains untouched until final model comparison

---

## Common failure modes

| Failure | Cause | Fix |
|---------|-------|-----|
| Topology all zeros | Missing NMA-PCA `.npz` files | Run Ensemble\_NMA-PCA pipeline first (Step 2) |
| Topology all zeros | PDB has only `HETATM`, no `ATOM` records | Use standard protein PDB format |
| Wrong topo shape `(6, 200, 15)` | Ran `--mode protein_only` instead of `ensemble_motion` | Rerun with `--mode ensemble_motion` |
| Shape mismatch during training | Mixing `protein_only` and `ensemble_motion` `.npy` files | Keep feature directories separate per mode |
| MISSING\_PDB warning | PDB directory not found or no `.pdb` files in it | Check naming: `pdbs/<protein_id>/conformation.pdb` |
| Warning: "No motion profiles found" | NMA-PCA files absent; falls back to all-residue topology | Run NMA-PCA pipeline; extractor still works but loses motion-guidance |
| `transformers` import error | Newer transformers installed | `pip install "transformers==4.24.0"` |
| HF cache on shared filesystem | Slow I/O during GPU jobs | Export `HF_HOME` to job-local `$TMPDIR` |
| Loss NaN | Inf values in topology features | Run QA check script above |

---

## What a successful output directory contains

After training, archive these together for reproducibility:

```
runs/mf_model_run1/
├── pytorch_model.bin       (or checkpoint-best/)
├── config.json
├── go_terms.txt
├── topo_scaler.pkl
├── training_args.json
├── trainer_state.json
├── test_results.json
└── runs/                   (TensorBoard logs)
```

Also archive externally:
- exact `train_ids.txt`, `val_ids.txt`, `test_ids.txt`
- exact annotation file used
- repo commit hash: `git rev-parse HEAD`
- SLURM batch scripts used

---

## Recommended first experiment (ablation)

For a practical first run, train and compare three variants:

| Model | Topology mode | Expected val Fmax |
|-------|--------------|------------------|
| Sequence only (ESM + ProtTrans) | None | 0.42–0.48 |
| Single-structure topology + sequence | `protein_only` | 0.46–0.54 |
| **Ensemble motion topology + sequence** | `ensemble_motion` | **0.50–0.60** |

This ablation tells the lab whether the motion-guided ensemble topology is adding
real MF prediction value over sequence embeddings alone.
