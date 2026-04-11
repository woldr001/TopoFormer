# How-To: Train, Validate, and Test TopoFormer-MF on the MSU HPCC

This guide explains how to use the `protein_function/` pipeline in the `protein-topology-integration-ALbIE` branch to predict **Molecular Function (MF)** Gene Ontology terms from a **protein PDB structure**.

It is written for our lab’s workflow on the **MSU HPCC** and is intended to help a new team member go from:

**PDB files + MF annotations -> precomputed topology + sequence features -> trained model -> validated/tested model -> single-protein prediction**

---

## 1. What this branch does

The branch adds a new `protein_function/` package that adapts the original TopoFormer protein-ligand pipeline into a **protein-only** function-prediction workflow.

The high-level idea is:

1. Read a **protein PDB**.
2. Convert the structure into a fixed-size **topological tensor** using persistent simplicial complex Laplacians.
3. Compute **sequence embeddings** from ESM and ProtTrans.
4. Fuse topology + sequence information in a new classifier.
5. Predict **multi-label MF GO terms**.

### Implemented components

- `topo_extraction/protein_only_topo_embedding.py`
  - builds **protein-only** topology tensors
  - uses intra-protein heavy-atom distances
  - outputs a tensor of shape **[6, 200, 15]**
- `scripts/precompute_topo_features.py`
  - batch precompute of topology tensors from PDBs
- `scripts/precompute_seq_features.py`
  - batch precompute of ESM + ProtTrans embeddings from FASTA
- `models/mini_topt_config.py`
  - small topology encoder config for the new tensor shape
- `models/modeling_topo_function.py`
  - fusion model: topology encoder + ESM projector + ProtTrans projector + MF classifier
- `data/protein_function_dataset.py`
  - dataset loader for `.npy` features and GO labels
- `evaluation/metrics.py`
  - Fmax and AUPR metrics
- `training/train_mf_prediction.py`
  - training pipeline using Hugging Face Trainer
- `predict.py`
  - end-to-end inference from a trained model

---

## 2. Critical implementation decision before running anything

### Standardize the ESM embedding dimension first

The current branch is **not fully self-consistent** on ESM dimensionality.

- The training/model side assumes **1152-dimensional** ESM features in several places.
- The sequence precompute script defaults to `facebook/esm2_t33_650M_UR50D`, which produces **1280-dimensional** embeddings.
- `predict.py` also computes ESM embeddings compatible with that **1280-dimensional** ESM-2 model.

### Recommended lab standard

Use the branch exactly as implemented for sequence precompute, and standardize on:

- **ESM model**: `facebook/esm2_t33_650M_UR50D`
- **ESM embedding size**: `1280`
- **ProtTrans embedding size**: `1024`

### What to change

Before production runs, update the default ESM dimension from `1152` to `1280` in:

- `training/train_mf_prediction.py`
- any docstrings/comments in `modeling_topo_function.py`
- any README text that still says the ESM branch is 1152-d

Even if you do not edit the defaults immediately, you should **always pass `--esm_dim 1280` explicitly** at training time.

---

## 3. Why each feature stream is needed

### A. Topology from PDB

The topology branch is the whole reason this adaptation exists.

It captures **3D structural organization** that pure sequence models miss, including:

- how atom groups are arranged in space
- multi-scale geometric relationships
- patterns that may relate to folds, active sites, cavities, and structural motifs tied to function

In this branch, the structure is converted into a fixed tensor **[6, 200, 15]**, which makes proteins of different sizes usable by the same encoder.

### B. ESM embeddings

The ESM branch gives a rich learned representation of protein sequence context. This helps the model use evolutionary-style sequence information that topology alone may not fully capture.

### C. ProtTrans embeddings

ProtTrans gives a second sequence representation family with a different pretraining history. In practice, this can add complementary information and improve robustness.

### D. Fusion

Molecular function depends on both **what the sequence suggests** and **how the structure is organized**. The fusion model is intended to let the classifier use both.

---

## 4. Recommended dataset strategy

For a first serious training run, use:

- **UniProtKB/Swiss-Prot** for reviewed proteins and GO annotations
- **experimental MF evidence only**
- **AlphaFold DB** structures for proteins without a solved experimental structure

### Why this is the best starting point

- Swiss-Prot is reviewed and manually curated.
- GO annotations are available in a format that is straightforward to filter.
- AlphaFold DB makes structure coverage practical at scale.

### Annotation filtering rules

Keep **Molecular Function** annotations with stronger evidence only.

Recommended evidence codes to keep:

- `EXP`
- `IDA`
- `IPI`
- `IMP`
- `IGI`
- `IEP`
- `TAS`
- `IC`

Exclude weak/electronic-only annotations such as `IEA` for the main benchmark training set.

### Term filtering

For the first model, keep only MF GO terms that appear at least **50 times** in the training set.

That matches the branch default and helps avoid training a classifier on labels with almost no support.

---

## 5. Recommended train/val/test split strategy

Do **not** use a naive random split.

Why:

If closely related proteins are split across train and test, the model may appear to generalize when it is actually seeing near-duplicates.

### Recommended options

#### Option 1: Sequence-cluster split

Cluster proteins by sequence identity, then place entire clusters into train, validation, or test.

A practical first-pass rule is:

- cluster at **30% sequence identity**
- never split a cluster across train/val/test

#### Option 2: Temporal split

Train on proteins annotated before a cutoff date and test on proteins annotated later.

This is especially useful if you want a benchmark that feels closer to a “future annotation” problem.

### Lab recommendation

For internal development:

- use **cluster-based splitting** first
- maintain a separate held-out test set that is not touched during model iteration

---

## 6. Files you need before running the pipeline

Create a project directory that contains at least the following:

```text
project_root/
├── pdbs/
│   ├── P12345.pdb
│   ├── Q8ABC1.pdb
│   └── ...
├── sequences.fasta
├── mf_annotations.tsv
├── train_ids.txt
├── val_ids.txt
├── test_ids.txt
└── runs/
```

### Required file formats

#### A. `pdbs/`
One PDB file per protein.

Recommended naming:

- `P12345.pdb`
- `Q8ABC1.pdb`

The feature scripts assume the protein ID can be matched to filenames.

#### B. `sequences.fasta`
FASTA headers should match your protein IDs exactly.

Example:

```fasta
>P12345
MSEQUENCE...
>Q8ABC1
MSEQUENCE...
```

#### C. `mf_annotations.tsv`
Use a long-format TSV with columns:

- `protein_id`
- `go_term`
- `namespace`

Example:

```tsv
protein_id	go_term	namespace
P12345	GO:0004672	molecular_function
P12345	GO:0005524	molecular_function
Q8ABC1	GO:0016491	molecular_function
```

#### D. `train_ids.txt`, `val_ids.txt`, `test_ids.txt`
One protein ID per line.

Example:

```text
P12345
Q8ABC1
...
```

---

## 7. Recommended HPCC directory layout

Use the HPCC storage tiers intentionally.

### Put here permanently

Use your lab research space for things you want to keep:

- repo clone
- final curated datasets
- trained model outputs you want to preserve
- final evaluation reports

Example:

```bash
/mnt/research/woldring_lab/TopoFormer-MF/
```

### Put heavy intermediate I/O here

Use scratch for:

- precomputed `.npy` features
- HF caches during repeated runs
- temporary working directories

Example:

```bash
/mnt/scratch/$USER/topoformer_mf/
```

### Put per-job temporary cache here

Inside a batch job, set Hugging Face and temp caches to node-local storage when possible.

Example:

```bash
export JOB_TMP=${TMPDIR:-/tmp/$USER/topoformer_mf_$SLURM_JOB_ID}
mkdir -p "$JOB_TMP"
export HF_HOME="$JOB_TMP/hf"
export TRANSFORMERS_CACHE="$JOB_TMP/hf/transformers"
export HUGGINGFACE_HUB_CACHE="$JOB_TMP/hf/hub"
```

This reduces pressure on networked storage and can make GPU jobs much smoother.

---

## 8. Create the software environment on the MSU HPCC

The branch has an `environment.yml`, but it is **not yet sufficient by itself** for the full new pipeline.

The current environment file is lightweight and does **not clearly include everything needed** by the new training pipeline, which imports at least:

- `torch`
- `transformers`
- `pandas`
- `scikit-learn`
- `scipy`
- `numpy`

### Recommended environment build

From the repo root:

```bash
module purge
module load Miniforge3

cd /mnt/research/woldring_lab/TopoFormer-MF/TopoFormer
conda env create -f environment.yml -n topoformer_mf
conda activate topoformer_mf
```

Then add the missing packages explicitly:

```bash
pip install torch torchvision torchaudio
pip install pandas accelerate sentencepiece
```

If your job will use GPUs, make sure the installed PyTorch build is compatible with the CUDA stack available on HPCC.

### Quick sanity check

Run this before launching big jobs:

```bash
python - <<'PY'
import torch, transformers, pandas, numpy, scipy, sklearn
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('pandas', pandas.__version__)
PY
```

---

## 9. Step 1: curate the dataset

This step is where most downstream problems are either prevented or created.

### Goal

Build a consistent dataset where every chosen protein has:

1. a valid protein ID
2. a PDB file
3. a FASTA sequence
4. at least one retained MF GO annotation
5. membership in exactly one of train/val/test

### Why this matters

The current dataset loader can silently degrade quality if features are missing. A clean manifest is far better than discovering later that some samples trained with zero-filled arrays.

### Suggested curation workflow

1. Download reviewed Swiss-Prot entries.
2. Keep only proteins with retained experimental MF annotations.
3. Retrieve or generate corresponding structures.
4. Remove proteins with missing sequence/PDB/annotation alignment.
5. Split proteins into train/val/test.
6. Save the four required files described above.

### Strong recommendation

Also create a single master manifest file for QA.

Example columns:

- `protein_id`
- `sequence_length`
- `pdb_path`
- `has_topology`
- `has_esm`
- `has_prottrans`
- `split`
- `n_mf_terms`

This makes debugging much easier.

---

## 10. Step 2: precompute topology features from PDBs

### Why precompute

Topology extraction is CPU-heavy and deterministic. You do not want to recompute it every training epoch.

### What the script produces

For each protein, the topology extractor creates a fixed tensor of shape:

```text
[6, 200, 15]
```

These are the structural inputs used by the topology encoder.

### Run command

```bash
python protein_function/scripts/precompute_topo_features.py \
  --pdb_dir ./pdbs \
  --output_dir ./topo_features \
  --n_workers 8
```

### Recommended HPCC production command

```bash
python protein_function/scripts/precompute_topo_features.py \
  --pdb_dir /mnt/research/woldring_lab/TopoFormer-MF/datasets/pdbs \
  --output_dir /mnt/scratch/$USER/topoformer_mf/topo_features \
  --pdb_list /mnt/research/woldring_lab/TopoFormer-MF/datasets/all_ids.txt \
  --n_workers 16
```

### What your team should verify immediately after this step

Check that:

1. the number of `.npy` outputs matches the number of expected proteins
2. all arrays have shape `(6, 200, 15)`
3. no proteins failed due to malformed PDB files

### Simple QA check

```bash
python - <<'PY'
import os, numpy as np
feat_dir = '/mnt/scratch/$USER/topoformer_mf/topo_features'
files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
print('n_files', len(files))
for f in files[:5]:
    arr = np.load(os.path.join(feat_dir, f))
    print(f, arr.shape, arr.dtype)
PY
```

---

## 11. Step 3: precompute ESM and ProtTrans sequence features

### Why precompute

These models are GPU-expensive relative to normal dataloading, but they do not need to be rerun every epoch.

Precomputing them once makes training much faster and much more reproducible.

### Default behavior

The script computes:

- one ESM embedding per protein
- one ProtTrans embedding per protein

and stores them as `.npy` files.

### Run command

```bash
python protein_function/scripts/precompute_seq_features.py \
  --fasta_file ./sequences.fasta \
  --esm_output_dir ./esm_features \
  --prottrans_output_dir ./prottrans_features \
  --device cuda
```

### Recommended HPCC production command

```bash
python protein_function/scripts/precompute_seq_features.py \
  --fasta_file /mnt/research/woldring_lab/TopoFormer-MF/datasets/sequences.fasta \
  --esm_output_dir /mnt/scratch/$USER/topoformer_mf/esm_features \
  --prottrans_output_dir /mnt/scratch/$USER/topoformer_mf/prottrans_features \
  --device cuda
```

### Important notes

- Standardize on **ESM-2 650M = 1280 dimensions**.
- ProtTrans should remain **1024 dimensions** for this branch.
- The embedding script may truncate long sequences. For very long proteins, inspect how many are exceeding the model max length and decide whether to exclude them or treat them separately.

### QA checks after this step

Check that:

1. every expected protein got an ESM `.npy`
2. every expected protein got a ProtTrans `.npy`
3. ESM shapes are consistent
4. ProtTrans shapes are consistent

Example:

```bash
python - <<'PY'
import os, numpy as np
for d in ['esm_features', 'prottrans_features']:
    files = sorted([f for f in os.listdir(d) if f.endswith('.npy')])
    print(d, 'n_files', len(files))
    for f in files[:3]:
        arr = np.load(os.path.join(d, f))
        print(' ', f, arr.shape)
PY
```

---

## 12. Step 4: train the MF predictor

### Why this script is structured this way

The training script does several important things:

1. reads train/val/test IDs
2. builds the MF label vocabulary
3. filters rare terms by minimum count
4. fits the topology scaler on the **training set only**
5. constructs datasets for each split
6. trains with validation monitoring
7. performs early stopping based on **Fmax**
8. saves the model and supporting files needed for inference

### Why fitting the scaler on training only matters

This prevents leakage from validation or test into the feature normalization process.

### Basic training command

```bash
python protein_function/training/train_mf_prediction.py \
  --topo_dir ./topo_features \
  --esm_dir ./esm_features \
  --prottrans_dir ./prottrans_features \
  --label_file ./mf_annotations.tsv \
  --train_ids ./train_ids.txt \
  --val_ids ./val_ids.txt \
  --test_ids ./test_ids.txt \
  --output_dir ./runs/mf_model_run1 \
  --esm_dim 1280 \
  --prottrans_dim 1024 \
  --num_train_epochs 50 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --learning_rate 1e-4
```

### Recommended first production training command on HPCC

```bash
python protein_function/training/train_mf_prediction.py \
  --topo_dir /mnt/scratch/$USER/topoformer_mf/topo_features \
  --esm_dir /mnt/scratch/$USER/topoformer_mf/esm_features \
  --prottrans_dir /mnt/scratch/$USER/topoformer_mf/prottrans_features \
  --label_file /mnt/research/woldring_lab/TopoFormer-MF/datasets/mf_annotations.tsv \
  --train_ids /mnt/research/woldring_lab/TopoFormer-MF/datasets/train_ids.txt \
  --val_ids /mnt/research/woldring_lab/TopoFormer-MF/datasets/val_ids.txt \
  --test_ids /mnt/research/woldring_lab/TopoFormer-MF/datasets/test_ids.txt \
  --output_dir /mnt/research/woldring_lab/TopoFormer-MF/runs/mf_model_run1 \
  --esm_dim 1280 \
  --prottrans_dim 1024 \
  --num_train_epochs 50 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --weight_decay 0.01
```

### Optional: initialize from pretrained TopoFormer weights

If you have a topology encoder checkpoint from the original TopoFormer model and the partial weight transfer utility is working as intended, use it to initialize the topology branch.

That can improve convergence, but only if the transferred layers align correctly.

Do a controlled comparison:

- run one training job **with** transferred weights
- run one job **without** transferred weights
- compare validation Fmax and AUPR

---

## 13. Step 5: validate during training

Validation is not optional.

### Why it matters

MF prediction is multi-label and can overfit in ways that a loss curve alone does not reveal.

The training script uses:

- **Fmax** as the main model-selection metric
- **AUPR** as a secondary metric
- **early stopping** to stop wasting GPU time once validation stops improving

### What your team should log every run

At minimum, keep a run sheet with:

- run name
- dataset version
- split version
- min annotation count
- ESM model and embedding dimension
- ProtTrans model and embedding dimension
- batch size
- learning rate
- whether pretrained topology weights were used
- best validation Fmax
- best validation AUPR
- test Fmax
- test AUPR

---

## 14. Step 6: evaluate on the held-out test set

Use the held-out test set only after the model configuration is finalized.

### Goal

Measure how well the trained model generalizes to proteins not used during fitting or validation-based model selection.

### Built-in evaluation

The training script supports optional test evaluation and writes a `test_results.txt`-style summary to the model output directory.

### Publication-grade evaluation recommendation

The in-branch metrics are useful for routine training, but for final reporting it is a good idea to also evaluate with an external CAFA-style evaluator.

That gives your lab a cleaner bridge to the broader protein-function-prediction literature.

---

## 15. Step 7: run single-protein inference from a PDB

Once the model is trained, inference can be done from a protein structure.

### What is needed

The model directory should contain at least:

- trained model weights
- `go_terms.txt`
- `topo_scaler.pkl`

### Example inference command

```bash
python protein_function/predict.py \
  --model_dir /mnt/research/woldring_lab/TopoFormer-MF/runs/mf_model_run1 \
  --protein_pdb /mnt/research/woldring_lab/TopoFormer-MF/inference/example.pdb \
  --sequence MSEQUENCE... \
  --threshold 0.5
```

If you already precomputed sequence embeddings for that protein, use those instead of recomputing them on the fly.

### Why this matters

This is the final practical use case for the branch:

**given a new protein structure, predict candidate MF GO terms**.

---

## 16. Recommended MSU HPCC batch scripts

## A. CPU job for topology precompute

Save as `sbatch_topology_precompute.sh`:

```bash
#!/bin/bash --login
#SBATCH --job-name=mf_topo_precompute
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/mnt/scratch/%u/topoformer_mf/logs/%x_%j.out
#SBATCH --error=/mnt/scratch/%u/topoformer_mf/logs/%x_%j.err

set -euo pipefail

module purge
module load Miniforge3
conda activate topoformer_mf

mkdir -p /mnt/scratch/$USER/topoformer_mf/logs
mkdir -p /mnt/scratch/$USER/topoformer_mf/topo_features

python /mnt/research/woldring_lab/TopoFormer-MF/TopoFormer/protein_function/scripts/precompute_topo_features.py \
  --pdb_dir /mnt/research/woldring_lab/TopoFormer-MF/datasets/pdbs \
  --output_dir /mnt/scratch/$USER/topoformer_mf/topo_features \
  --pdb_list /mnt/research/woldring_lab/TopoFormer-MF/datasets/all_ids.txt \
  --n_workers 16
```

## B. GPU job for sequence feature precompute

Save as `sbatch_seq_precompute.sh`:

```bash
#!/bin/bash --login
#SBATCH --job-name=mf_seq_precompute
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mnt/scratch/%u/topoformer_mf/logs/%x_%j.out
#SBATCH --error=/mnt/scratch/%u/topoformer_mf/logs/%x_%j.err

set -euo pipefail

module purge
module load Miniforge3
conda activate topoformer_mf

mkdir -p /mnt/scratch/$USER/topoformer_mf/logs
mkdir -p /mnt/scratch/$USER/topoformer_mf/esm_features
mkdir -p /mnt/scratch/$USER/topoformer_mf/prottrans_features

JOB_TMP=${TMPDIR:-/tmp/$USER/topoformer_mf_$SLURM_JOB_ID}
mkdir -p "$JOB_TMP"
export HF_HOME="$JOB_TMP/hf"
export TRANSFORMERS_CACHE="$JOB_TMP/hf/transformers"
export HUGGINGFACE_HUB_CACHE="$JOB_TMP/hf/hub"

python /mnt/research/woldring_lab/TopoFormer-MF/TopoFormer/protein_function/scripts/precompute_seq_features.py \
  --fasta_file /mnt/research/woldring_lab/TopoFormer-MF/datasets/sequences.fasta \
  --esm_output_dir /mnt/scratch/$USER/topoformer_mf/esm_features \
  --prottrans_output_dir /mnt/scratch/$USER/topoformer_mf/prottrans_features \
  --device cuda
```

## C. GPU job for training

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

module purge
module load Miniforge3
conda activate topoformer_mf

mkdir -p /mnt/scratch/$USER/topoformer_mf/logs
mkdir -p /mnt/research/woldring_lab/TopoFormer-MF/runs

JOB_TMP=${TMPDIR:-/tmp/$USER/topoformer_mf_$SLURM_JOB_ID}
mkdir -p "$JOB_TMP"
export HF_HOME="$JOB_TMP/hf"
export TRANSFORMERS_CACHE="$JOB_TMP/hf/transformers"
export HUGGINGFACE_HUB_CACHE="$JOB_TMP/hf/hub"

python /mnt/research/woldring_lab/TopoFormer-MF/TopoFormer/protein_function/training/train_mf_prediction.py \
  --topo_dir /mnt/scratch/$USER/topoformer_mf/topo_features \
  --esm_dir /mnt/scratch/$USER/topoformer_mf/esm_features \
  --prottrans_dir /mnt/scratch/$USER/topoformer_mf/prottrans_features \
  --label_file /mnt/research/woldring_lab/TopoFormer-MF/datasets/mf_annotations.tsv \
  --train_ids /mnt/research/woldring_lab/TopoFormer-MF/datasets/train_ids.txt \
  --val_ids /mnt/research/woldring_lab/TopoFormer-MF/datasets/val_ids.txt \
  --test_ids /mnt/research/woldring_lab/TopoFormer-MF/datasets/test_ids.txt \
  --output_dir /mnt/research/woldring_lab/TopoFormer-MF/runs/mf_model_run1 \
  --esm_dim 1280 \
  --prottrans_dim 1024 \
  --num_train_epochs 50 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --learning_rate 1e-4
```

### Notes for HPCC users

- Adjust `--gpus`, memory, and time upward if your dataset is large.
- If your lab has access to specific GPU types, request them explicitly.
- Use scratch for large intermediate feature directories.
- Keep final models and curated inputs in research space.

---

## 17. Quality-control checklist before launching a long training run

Do not skip this.

### Data checks

- [ ] every train/val/test ID has a matching FASTA entry
- [ ] every train/val/test ID has a matching PDB
- [ ] every train/val/test ID has at least one retained MF annotation
- [ ] no ID appears in more than one split
- [ ] no split file contains extra whitespace or duplicate IDs

### Feature checks

- [ ] every ID has a topology `.npy`
- [ ] every ID has an ESM `.npy`
- [ ] every ID has a ProtTrans `.npy`
- [ ] topology arrays have shape `(6, 200, 15)`
- [ ] ESM arrays are consistently `(1280,)`
- [ ] ProtTrans arrays are consistently `(1024,)`

### Training checks

- [ ] `--esm_dim 1280` is explicitly set
- [ ] output directory is empty or intentionally reused
- [ ] scaler is fit from training only
- [ ] validation IDs are not used to tune the train split itself
- [ ] test set remains untouched until final model comparison

---

## 18. Common failure modes and how to avoid them

### Failure 1: shape mismatch during training

Cause:

- ESM features were precomputed at 1280-d, but training assumed 1152-d.

Fix:

- pass `--esm_dim 1280`
- update code defaults to 1280
- delete and regenerate inconsistent features if needed

### Failure 2: silent zero-filled samples

Cause:

- missing `.npy` features or unmatched IDs

Fix:

- verify feature counts and file naming before training
- create a manifest and explicitly check completeness

### Failure 3: poor apparent test performance from label noise

Cause:

- weak GO evidence or bad namespace filtering

Fix:

- train on experimental MF annotations only
- confirm namespace is `molecular_function`

### Failure 4: unstable results across runs

Cause:

- changing splits, caches, or preprocessing between runs

Fix:

- version your splits
- keep a run sheet
- do not overwrite previous run directories

### Failure 5: long HPCC jobs slowed by networked cache use

Cause:

- Hugging Face cache living on shared network storage

Fix:

- set cache directories to job-local temporary storage inside each batch script

---

## 19. What a successful output directory should contain

After training, the output directory should include the items needed to reproduce inference.

At minimum, expect artifacts such as:

- model weights
- trainer/config files
- `go_terms.txt`
- `topo_scaler.pkl`
- training logs
- test results summary

Your team should archive the following together:

1. model output directory
2. exact split files
3. exact annotation file used
4. exact commit hash of the repo branch
5. batch script used to launch the run

---

## 20. Minimum recommended first experiment

For a practical first run on HPCC:

1. curate a clean MF dataset of moderate size
2. precompute topology on CPU
3. precompute sequence embeddings on GPU
4. train with `--esm_dim 1280`
5. compare:
   - topology + ESM + ProtTrans
   - ESM + ProtTrans only
   - topology only
6. evaluate on the held-out test set

This ablation is important because it tells the lab whether the topology branch is actually adding MF prediction value.

---

## 21. Final lab recommendation

For the first benchmark-quality version of this workflow, do the following in order:

1. **fix the ESM dimensionality mismatch**
2. **freeze the dataset and splits**
3. **precompute all features once**
4. **train with explicit versioned output directories**
5. **compare with and without topology**
6. **report both validation and held-out test performance**
7. **use the trained model + scaler + GO vocabulary together at inference time**

That will give the lab a reproducible and defensible pipeline for predicting Molecular Function directly from a protein PDB structure.
