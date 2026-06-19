# TopoFormer Protein Function Prediction — How-To Guide

**Branch:** `protein-motion-topology`
**Task:** Predict Molecular Function (MF) Gene Ontology terms from protein 3D structures using **side-chain centroid ensemble topology** + sequence embeddings.

> **This guide focuses on the side-chain centroid topology configuration.**
> Two other topology configurations exist and share the same sequence-embedding,
> training, and evaluation machinery (see `README.md` for their specifics):
> - **motion-guided ensemble** — atom-level cross-pair topology → `[12, 200, 121]`
> - **protein-only** — single-structure atom-level topology → `[6, 200, 15]`

---

## Table of Contents

1. [Overview and Architecture](#1-overview-and-architecture)
2. [Environment Setup](#2-environment-setup)
3. [Data Acquisition](#3-data-acquisition)
4. [Annotation File Format](#4-annotation-file-format)
5. [Train / Val / Test Splits](#5-train--val--test-splits)
6. [Pre-computing Side-chain Topology Features](#6-pre-computing-side-chain-topology-features)
7. [Pre-computing Sequence Embeddings (ESM + ProtTrans)](#7-pre-computing-sequence-embeddings-esm--prottrans)
8. [Training the Model](#8-training-the-model)
9. [Monitoring and Early Stopping](#9-monitoring-and-early-stopping)
10. [Evaluation](#10-evaluation)
11. [Inference on New Proteins](#11-inference-on-new-proteins)
12. [Troubleshooting](#12-troubleshooting)
13. [Expected Results and Benchmarks](#13-expected-results-and-benchmarks)
14. [Repository Layout](#14-repository-layout)
15. [Quick-Start Checklist](#15-quick-start-checklist)

---

## 1. Overview and Architecture

This tool predicts which **Molecular Function (MF) Gene Ontology terms** a protein performs, given:
- A set of 3D structural conformations (PDB files, e.g. from aSAM ensembles, AlphaFold, or MD simulation)
- Protein amino acid sequence (for ESM and ProtTrans embeddings)

The **side-chain centroid** configuration works at **residue resolution**. Instead of
the positions of individual heavy atoms (C, N, O, S), each residue is collapsed to a
single point — the **geometric centroid of its side-chain heavy atoms** (all heavy atoms
excluding the backbone N, Cα, C, O). Glycine, which has no side-chain heavy atoms, falls
back to its Cα coordinate.

Each residue is assigned one of four chemical classes, and topology is computed over a
**flat list of 15 class combinations** (every non-empty subset of the four classes).
This requires **no NMA-PCA motion profiles** — it runs directly on the PDB ensembles.

### Residue chemical classes

| Class | Label | Residues |
|-------|-------|----------|
| 0 | nonpolar / hydrophobic | G A V L I M P |
| 1 | aromatic | F W Y |
| 2 | polar / uncharged | S T C N Q |
| 3 | polar / charged | D E K R H |

### Side-chain-type combinations (flat list of 15)

4 singles + 6 pairs + 4 triples + 1 quadruple = **15 combinations**. For each combination,
the centroids of all residues whose class is in the subset are collected and an all-pairs
Persistent Simplicial Complex Laplacian (dim-0) filtration is run over them.

### Architecture

```
PDB ensemble (10 conformations per protein)
        │
        └─► sidechain_topo_embedding.py
                │
                ├── Reduce each residue to its side-chain centroid (Gly → Cα)
                ├── Assign each residue a chemical class (0-3)
                ├── For each of 15 class combinations: all-pairs distance matrix
                │   over the selected centroids → Persistent Laplacian dim-0
                │   over 200 filtration steps (0–40 Å, 0.2 Å)
                ├── Stack per-conformation [6, 200, 15] tensors
                └── Aggregate → mean + std → [12, 200, 15] saved as .npy

Protein sequence
        │
        ├─► ESM-2 (650M)  → mean-pool residues → [1280]  saved as .npy
        └─► ProtTrans T5  → mean-pool residues → [1024]  saved as .npy

                                ┌──────────────────────────────────────┐
                                │         TopoFunctionModel             │
   topo [B, 12, 200, 15]  ─────►│  Mini TopT encoder → CLS → [B, 256]  │
   esm  [B, 1280]         ─────►│  ESM proj → [B, 512]                 │
   prot [B, 1024]         ─────►│  ProtTrans proj → [B, 512]           │
                                │  concat → [B, 1280]                  │
                                │  FC(1280→512) → GELU → FC(512→N_mf)  │
                                │  BCEWithLogitsLoss                   │
                                └──────────────────────────────────────┘
                                              │
                                    MF GO term predictions
```

### Key design choices

| Feature | Value | Rationale |
|---------|-------|-----------|
| Topological points | side-chain centroids (Gly → Cα) | Residue-resolution; captures side-chain packing without atom-level cost |
| Side-chain-type combos | 15 flat subsets of 4 chemical classes | Encodes how hydrophobic / aromatic / polar / charged residues co-locate |
| Filtration range | 0–40 Å, 200 steps of 0.2 Å | A few hundred centroids span larger distances than atoms; 40 Å captures domain-scale contacts |
| Ensemble aggregation | mean + std across 10 conformers | Mean = average topology; std = conformational variability hypothesised to carry functional signal |
| NMA-PCA required? | **No** | This configuration is not motion-guided |
| Topo encoder size | 4 layers, hidden=256 | ~3M params; appropriate for 30k–100k protein training sets to avoid overfitting |
| ESM model | ESM-2 650M (1280-dim) | Strong sequence representation |

### Why residue resolution is cheaper

The Persistent Laplacian eigenvalue computation scales as O(N³) in the number of points N.
At atom resolution a 300-residue protein has ~2,000+ heavy atoms; at residue resolution it
has only ~300 centroids. The side-chain configuration is therefore **much faster** than the
atom-level configurations, and the 200-step / 40 Å filtration is comfortably affordable.

---

## 2. Environment Setup

### 2.1 Clone and install

```bash
# Clone the repository (if not already)
git clone https://github.com/woldr001/TopoFormer.git
cd TopoFormer
git checkout protein-motion-topology

# Create a dedicated conda environment
conda create -n topoformer python=3.8
conda activate topoformer

# Core dependencies — pinned together in requirements_mf.txt to avoid a
# numpy/scipy/prody/matplotlib resolver conflict (see the note below)
pip install -r protein_function/requirements_mf.txt
pip install torch torchvision
pip install datasets accelerate
pip install biopython   # optional, for PDB utilities

# Sequence embedding models
pip install fair-esm        # for ESM-2
pip install sentencepiece   # for ProtTrans tokenizer
```

> **Always install core scientific packages from `requirements_mf.txt`,
> not standalone `pip install numpy`/`pip install matplotlib` commands.**
> `scipy==1.7.3` and `prody==2.4.1` both require `numpy<1.23`, but
> `matplotlib>=3.6` requires `numpy>=1.23` — installing any one of these
> packages independently can silently upgrade numpy and break the others.
> `requirements_mf.txt` pins a mutually compatible set
> (`numpy==1.22.4`, `matplotlib<3.6`).

> **Why `transformers==4.24.0`?**
> `modeling_topt.py` imports `find_pruneable_heads_and_indices` from `transformers.pytorch_utils`, a symbol removed in newer releases. Pin to 4.24.0 until the import is updated.

On the **MSU HPCC**, the `topoformer_mf` environment is pre-created. Activate it by prepending its `bin/` (the system Miniforge3 module breaks `conda activate`):

```bash
export PATH="/mnt/home/woldring/.conda/envs/topoformer_mf/bin:$PATH"
which python   # should show .../topoformer_mf/bin/python
```

### 2.2 Verify installation

```bash
python -c "
import torch
from code_pkg.top_transformer.modeling_topt import TopTModel
from protein_function.models.mini_topt_config import get_sidechain_topt_config
from protein_function.models.modeling_topo_function import TopoFunctionModel

config = get_sidechain_topt_config()        # [12, 200, 15]
model = TopoFunctionModel(config, num_mf_labels=100)
topo = torch.randn(2, 12, 200, 15)
esm  = torch.randn(2, 1280)
prot = torch.randn(2, 1024)
out  = model(topo, esm, prot)
print(out.logits.shape)   # → torch.Size([2, 100])
print('Installation OK')
"
```

---

## 3. Data Acquisition

### 3.1 Recommended datasets

| Dataset | Size | Source | Notes |
|---------|------|--------|-------|
| **Swiss-Prot (reviewed)** | ~570k proteins | [UniProt](https://www.uniprot.org/uniprotkb?query=reviewed:true) | High-quality manual annotations; exclude IEA (electronically inferred) |
| **CAFA3 benchmark** | ~66k proteins | [CAFA3](https://www.biofunctionprediction.org/cafa/) | Standard benchmark; official train/test splits available |
| **PDB-mmCIF structures** | millions | [RCSB PDB](https://www.rcsb.org/) | Experimental structures |
| **AlphaFold DB** | ~200M | [AlphaFold DB](https://alphafold.ebi.ac.uk/) | Predicted structures for most UniProt entries |

### 3.2 Minimum training set size

For MF GO term prediction:
- **Minimum recommended**: ~5,000 proteins covering ≥50 MF terms each with ≥50 annotations
- **Better**: 30,000–100,000 proteins (Swiss-Prot scale)
- Terms with fewer than 50 training examples are dropped automatically by `build_mf_label_matrix(min_annotation_count=50)`

### 3.3 Structural conformations

Each protein should have **10 structural conformations** for the ensemble. Good sources:
- aSAM ensembles (this project's primary source)
- AlphaFold2 run with different random seeds or `--num_models 5 --num_seeds 2`
- Short MD simulations (10–100 ns) sampled at equal time intervals
- Normal-mode-generated conformations from ProDy

The side-chain extractor searches for conformation PDBs in this order under `<pdb_dir>/<protein_id>/`:
1. `clustering.nodes_pipeline_v1/raw.seed_*.pdb`  ← aSAM ensemble layout (this project)
2. `cluster_repr_*.pdb`
3. `<protein_id>_conf_*.pdb`
4. Any `*.pdb` file (sorted, up to `n_conformers`)

### 3.4 Directory structure expected

Note: **no `nma_pca/` directory is required** for the side-chain configuration.

```
/data/
├── pdbs/
│   └── P12345/
│       └── clustering.nodes_pipeline_v1/
│           ├── raw.seed_0.pdb
│           ├── raw.seed_1.pdb
│           └── ... (10 files)
├── topo_features_sidechain/
│   └── P12345.npy   ← [12, 200, 15], written by precompute script
├── esm_features/
│   └── P12345.npy   ← [1280], written by precompute script
├── prottrans_features/
│   └── P12345.npy   ← [1024], written by precompute script
└── annotations/
    └── mf_annotations.tsv
```

---

## 4. Annotation File Format

### 4.1 Long format (preferred)

Tab-separated with header:

```
protein_id	go_term	namespace
P12345	GO:0003677	molecular_function
P12345	GO:0046872	molecular_function
Q99999	GO:0003824	molecular_function
```

Load with:
```python
from protein_function.data.protein_function_dataset import build_mf_label_matrix
label_df, go_terms = build_mf_label_matrix(
    "annotations/mf_annotations.tsv",
    min_annotation_count=50,
    sep="\t",
    protein_col="protein_id",
    go_col="go_term",
    namespace_col="namespace",
    namespace_value="molecular_function",
)
```

### 4.2 Wide format

CSV with `protein_id` index column and one column per GO term (0/1 values):

```
protein_id,GO:0003677,GO:0046872,GO:0003824
P12345,1,1,0
Q99999,0,0,1
```

The loader auto-detects format. Pass `namespace_col=None` when using wide format.

### 4.3 Downloading from UniProt

```bash
# Download reviewed Swiss-Prot entries with MF annotations
wget "https://rest.uniprot.org/uniprotkb/stream?\
  query=reviewed:true+AND+go_evidence:manual&\
  fields=accession,go_id,go_namespace&\
  format=tsv" -O mf_annotations.tsv

# Remove IEA (electronic) annotations
grep -v "IEA" mf_annotations.tsv > mf_annotations_manual.tsv
```

---

## 5. Train / Val / Test Splits

### 5.1 Recommended split strategy

For a rigorous benchmark, use **temporal** or **sequence-identity** splits to avoid data leakage:

**Option A — Temporal split (CAFA-style)**
- Training: proteins with annotations before a cutoff date (e.g. 2018)
- Test: proteins newly annotated after the cutoff
- Use the official CAFA3 splits if benchmarking against published results

**Option B — Sequence identity split (recommended for custom datasets)**
```bash
# Cluster sequences at 30% identity with CD-HIT
cd-hit -i all_sequences.fasta -o clusters.fasta -c 0.30 -n 2 -T 8 -M 8000
# Split cluster representatives: 70% train / 15% val / 15% test
python scripts/split_by_clusters.py clusters.clstr --train 0.70 --val 0.15 --test 0.15
```

On this project, `preprocess_training_data.py` scans the sampling directory for proteins
with complete aSAM ensembles, filters annotations/FASTA to UniProt accessions, and writes a
random 70/15/15 split (`train_ids.txt`, `val_ids.txt`, `test_ids.txt`) plus `mf_annotations.tsv`,
`sequences.fasta`, and `manifest.csv`.

### 5.2 Split files format

Plain text, one protein ID per line:

```
# train_ids.txt
P12345
Q99999
...
```

### 5.3 Class imbalance

MF GO terms are heavily imbalanced (most proteins have 1–5 terms, common terms have thousands of annotations). The BCEWithLogitsLoss handles this inherently. If extreme imbalance is a problem, compute per-term positive weight:

```python
pos_weight = (label_df == 0).sum() / (label_df == 1).sum().clip(lower=1)
# Pass to BCEWithLogitsLoss via the training script's --pos_weight flag
```

---

## 6. Pre-computing Side-chain Topology Features

This is the core topology step for this configuration. It needs **only the PDB ensembles** —
no NMA-PCA profiles — so it can run directly after preprocessing.

### 6.1 What the extractor produces

For each protein: one `<protein_id>.npy` file of shape **`[12, 200, 15]`** (float32).

```
[12, 200, 15]
 │    │   │
 │    │   └── side-chain-type combination index  (15 flat subsets)
 │    └─────── filtration distance step          (0.0 → 39.8 Å, 200 steps of 0.2 Å)
 └──────────── channel                           (6 spectral stats × mean + 6 × std = 12)
```

### 6.2 Single-protein test run

```bash
export PATH="/mnt/home/woldring/.conda/envs/topoformer_mf/bin:$PATH"
python protein_function/topo_extraction/sidechain_topo_embedding.py \
    --protein_id    A0A010 \
    --pdb_dir       /mnt/research/nodes/giacomo/asam_ensembles/protein_function_prediction/v0/sampling \
    --output_folder /mnt/research/woldring_lab/TopoFormer-MF/topo_features_sidechain
```

### 6.3 Batch extraction (multiprocessing, single node)

```bash
python protein_function/scripts/precompute_topo_features.py \
    --mode                  sidechain_centroid \
    --pdb_dir               /data/pdbs \
    --output_dir            /data/topo_features_sidechain \
    --n_conformers          10 \
    --dis_start             0.0 \
    --dis_cutoff            40.0 \
    --dis_step              0.2 \
    --ensemble_aggregation  mean_std \
    --n_workers             8
```

### 6.4 Batch extraction (SLURM array job — recommended)

```bash
# Submits a chunked array job (5 proteins per task, 5 CPUs each), writing to
# /mnt/research/woldring_lab/TopoFormer-MF/topo_features_sidechain/
bash protein_function/scripts/submit_topo_features_sidechain.sh
```

To scale to larger datasets, increase the chunk size:
```bash
CHUNK_SIZE=20 bash protein_function/scripts/submit_topo_features_sidechain.sh
```

### 6.5 Key arguments explained

| Argument | Default | Meaning |
|----------|---------|---------|
| `--mode` | `protein_only` | Use `sidechain_centroid` for this configuration |
| `--n_conformers` | 10 | Max number of PDB conformations to use per protein |
| `--dis_start` / `--dis_cutoff` / `--dis_step` | 0.0 / 40.0 / 0.2 | Filtration range → 200 steps |
| `--ensemble_aggregation` | `mean_std` | `mean_std` → 12 channels; `mean_only` → 6 channels |
| `--n_workers` | 4 | Parallel worker processes |
| `--overwrite` | False | Re-compute even if output file exists |

### 6.6 QA check after completion

```python
import os, numpy as np
feat_dir = '/data/topo_features_sidechain'
files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
print(f'n_files: {len(files)}')
bad = []
for f in files:
    arr = np.load(os.path.join(feat_dir, f))
    if arr.shape != (12, 200, 15):
        bad.append((f, arr.shape))
    elif not np.isfinite(arr).all():
        bad.append((f, 'has inf/nan'))
print(f'Bad files: {len(bad)}')
for b in bad[:10]:
    print(' ', b)
```

### 6.7 Fit and save the topology scaler

The topology features must be normalised with a `StandardScaler + MinMaxScaler` pipeline fitted **only on the training set**:

```python
from protein_function.data.protein_function_dataset import fit_topo_scaler

train_ids = open("train_ids.txt").read().split()
scaler = fit_topo_scaler(
    protein_ids=train_ids,
    topo_dir="/data/topo_features_sidechain",
    save_path="/data/topo_scaler.pkl",
)
```

> If you omit `--topo_scaler_path` at training time, the training script fits and saves a
> scaler on the training set automatically (to `<output_dir>/topo_scaler.pkl`).

---

## 7. Pre-computing Sequence Embeddings (ESM + ProtTrans)

### 7.1 Prepare a FASTA file

```
>P12345
MKTIIALSYIFCLVFA...
>Q99999
MAEGEITTFTALTEKF...
```

### 7.2 Run batch extraction

```bash
python protein_function/scripts/precompute_seq_features.py \
    --fasta_file           /data/sequences.fasta \
    --esm_output_dir       /data/esm_features \
    --prottrans_output_dir /data/prottrans_features \
    --esm_model            facebook/esm2_t33_650M_UR50D \
    --prottrans_model      Rostlab/prot_t5_xl_uniref50 \
    --batch_size           8 \
    --device               cuda
```

On HPCC, use the provided sbatch script which handles HuggingFace cache placement and GPU allocation:
```bash
sbatch protein_function/scripts/sbatch_seq_features.sh
```

Output: `<protein_id>.npy` files of shape `[1280]` (ESM) and `[1024]` (ProtTrans).

### 7.3 GPU memory requirements

| Model | Parameters | GPU VRAM (batch 16) |
|-------|-----------|---------------------|
| ESM-2 650M | 650M | ~6 GB |
| ProtTrans T5 | 3B | ~12 GB |

For limited VRAM, reduce `--batch_size` to 4 or 1. Both models can run on CPU (slow: ~30 s/protein for ProtTrans).

### 7.4 ESM embedding dimension

The default ESM model (`esm2_t33_650M_UR50D`) outputs 1280-dim embeddings. Pass `--esm_dim 1280`
to the training script (its built-in default is lower). If you use a different ESM variant:

| Model | Dim |
|-------|-----|
| `esm2_t6_8M_UR50D` | 320 |
| `esm2_t12_35M_UR50D` | 480 |
| `esm2_t30_150M_UR50D` | 640 |
| `esm2_t33_650M_UR50D` | 1280 |
| `esm2_t36_3B_UR50D` | 2560 |

---

## 8. Training the Model

### 8.1 Directory prerequisites

Before training, verify you have:
```
/data/topo_features_sidechain/  # .npy files [12, 200, 15]
/data/esm_features/             # .npy files [1280]
/data/prottrans_features/       # .npy files [1024]
/data/annotations/mf_annotations.tsv
/data/train_ids.txt
/data/val_ids.txt
/data/test_ids.txt
```

### 8.2 Launch training

The topology input shape is passed explicitly so the encoder is built for `[12, 200, 15]`:

```bash
python protein_function/training/train_mf_prediction.py \
    --topo_dir            /data/topo_features_sidechain \
    --esm_dir             /data/esm_features \
    --prottrans_dir       /data/prottrans_features \
    --label_file          /data/annotations/mf_annotations.tsv \
    --train_ids           /data/train_ids.txt \
    --val_ids             /data/val_ids.txt \
    --topo_channels       12 \
    --topo_n_filtrations  200 \
    --topo_n_combinations 15 \
    --esm_dim             1280 \
    --output_dir          /data/function_model_sidechain \
    --num_train_epochs    50 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size  64 \
    --learning_rate       1e-4 \
    --weight_decay        0.01 \
    --warmup_ratio        0.05 \
    --lr_scheduler_type   cosine \
    --evaluation_strategy epoch \
    --save_strategy       epoch \
    --load_best_model_at_end true \
    --metric_for_best_model fmax \
    --fp16
```

> **Topology shape flags are required for this configuration.** The script's built-in
> defaults (`--topo_channels 6 --topo_n_combinations 15`) describe the protein-only mode.
> The side-chain features are 12-channel, so `--topo_channels 12` is essential — otherwise
> the patch-embedding Conv2d expects 6 input channels and training fails with a shape error.
> Width (`--topo_n_combinations 15`) already matches the protein-only default, but pass it
> explicitly for clarity.

### 8.3 Hyperparameter recommendations

| Hyperparameter | Recommended range | Notes |
|---------------|-------------------|-------|
| `learning_rate` | 5e-5 – 2e-4 | Lower if loss oscillates; higher if convergence is slow |
| `per_device_train_batch_size` | 32–128 | Larger batches → more stable gradients for multi-label tasks |
| `num_train_epochs` | 50–100 | Early stopping at patience=10 prevents overfitting |
| `warmup_ratio` | 0.05–0.10 | Longer warmup helps with the topology encoder |
| `weight_decay` | 0.01 | AdamW regularisation |
| `fusion_dropout` | 0.2–0.4 | Increase if val Fmax plateaus while train Fmax rises |

### 8.4 Multi-GPU training

```bash
torchrun --nproc_per_node=4 protein_function/training/train_mf_prediction.py \
    [... same args ...]
```

### 8.5 Two-stage fine-tuning (optional)

For better convergence when starting from a pretrained TopoFormer checkpoint:

**Stage 1** (5–10 epochs): Freeze the topo encoder, train only the projection + fusion heads.
```bash
python protein_function/training/train_mf_prediction.py \
    ... \
    --freeze_topo_encoder true \
    --num_train_epochs 10 \
    --output_dir /data/stage1_model
```

**Stage 2** (40–90 epochs): Unfreeze all layers, use a lower learning rate.
```bash
python protein_function/training/train_mf_prediction.py \
    ... \
    --resume_from_checkpoint /data/stage1_model/checkpoint-best \
    --freeze_topo_encoder false \
    --learning_rate 3e-5 \
    --num_train_epochs 90 \
    --output_dir /data/stage2_model
```

### 8.6 Loading pretrained TopoFormer weights

If you have a pretrained `pytorch_model.bin` from the original TopoFormer:

```python
from protein_function.models.mini_topt_config import (
    get_sidechain_topt_config,
    load_pretrained_topt_weights,
)
from code_pkg.top_transformer.modeling_topt import TopTModel

config = get_sidechain_topt_config()        # [12, 200, 15]
encoder = TopTModel(config)
skipped = load_pretrained_topt_weights(encoder, "/path/to/pretrained_checkpoint/")
print(f"Skipped keys (incompatible shapes): {skipped}")
```

The patch embedding Conv2d and positional embeddings are re-initialised (shape incompatibility); all compatible attention/LayerNorm weights are sliced to fit the smaller hidden dimension (256 vs 768).

---

## 9. Monitoring and Early Stopping

### 9.1 TensorBoard

```bash
tensorboard --logdir /data/function_model_sidechain/runs
```

Key metrics to monitor:
- `eval/fmax` (primary — higher is better; should reach 0.45–0.60 on Swiss-Prot MF)
- `eval/aupr` (secondary)
- `train/loss` vs `eval/loss` (check for overfitting gap)

### 9.2 Early stopping

The trainer uses `EarlyStoppingCallback` with `patience=10` on `eval/fmax`. If val Fmax does not improve for 10 consecutive epochs, training stops and the best checkpoint is restored.

To adjust patience:
```bash
... --early_stopping_patience 15 ...
```

### 9.3 Signs of trouble

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Loss NaN from epoch 1 | Learning rate too high | Halve `--learning_rate` |
| Val Fmax ≤ 0.10 after 20 epochs | Topo features all-zeros (missing .npy files) | Check `--topo_dir` paths |
| `RuntimeError` channel/shape mismatch on first batch | Wrong `--topo_channels` | Side-chain features are 12-channel; set `--topo_channels 12` |
| Val Fmax stagnates at ~0.35 while train Fmax → 0.70 | Overfitting | Increase `--fusion_dropout 0.4`, reduce `hidden_size` |
| CUDA OOM | Batch too large | Halve `--per_device_train_batch_size` |
| `transformers` import error | Wrong transformers version | `pip install "transformers==4.24.0"` |

---

## 10. Evaluation

### 10.1 Standard CAFA metrics

```python
from protein_function.evaluation.metrics import compute_fmax, compute_aupr
import numpy as np

# Load test predictions and ground truth
y_true  = np.load("test_labels.npy")    # [N_proteins, N_terms]
y_score = np.load("test_logits.npy")    # [N_proteins, N_terms], raw logits
y_prob  = 1 / (1 + np.exp(-y_score))   # sigmoid → probabilities

fmax, optimal_threshold = compute_fmax(y_true, y_prob)
aupr = compute_aupr(y_true, y_prob)
print(f"Fmax: {fmax:.4f} at threshold {optimal_threshold:.2f}")
print(f"AUPR: {aupr:.4f}")
```

### 10.2 Run evaluation on a saved checkpoint

```bash
python protein_function/training/train_mf_prediction.py \
    ... \
    --do_train false \
    --do_eval  true \
    --do_predict true \
    --test_ids /data/test_ids.txt \
    --resume_from_checkpoint /data/function_model_sidechain/checkpoint-best
```

Outputs:
- `test_results.json` — Fmax, AUPR, per-term AP
- `test_predictions.npy` — raw logit matrix `[N_test, N_terms]`

### 10.3 Per-term analysis

```python
from sklearn.metrics import average_precision_score

per_term_ap = [
    average_precision_score(y_true[:, i], y_prob[:, i])
    for i in range(y_true.shape[1])
]
# Top 10 best-predicted terms
top_terms = np.argsort(per_term_ap)[::-1][:10]
for i in top_terms:
    print(f"{go_terms[i]}: AP={per_term_ap[i]:.3f}, n_pos={y_true[:,i].sum():.0f}")
```

### 10.4 Comparing topology configurations (ablation)

Because all three topology configurations feed the same fusion model, you can train each on
the same splits and compare directly:

| Configuration | Topology dir | `--topo_channels` | `--topo_n_combinations` |
|---------------|--------------|-------------------|--------------------------|
| Side-chain centroid | `topo_features_sidechain` | 12 | 15 |
| Motion-guided ensemble | `topo_features` | 12 | 121 |
| Protein-only (single PDB) | `topo_features_single` | 6 | 15 |

Keep each topology directory separate (all files in a run must share one shape).

---

## 11. Inference on New Proteins

For the side-chain configuration, the simplest robust route is **precompute features, then
run a forward pass**. (The `predict.py` CLI's built-in extractor uses the single-structure
protein-only path; for ensemble side-chain inference, precompute the `[12, 200, 15]` tensor
first.)

### 11.1 Step 1 — compute features for the new protein

```bash
# Side-chain topology → /data/infer/topo/P99999.npy  ([12, 200, 15])
python protein_function/topo_extraction/sidechain_topo_embedding.py \
    --protein_id    P99999 \
    --pdb_dir       /data/infer/pdbs \
    --output_folder /data/infer/topo

# ESM + ProtTrans → /data/infer/esm/P99999.npy, /data/infer/prottrans/P99999.npy
python protein_function/scripts/precompute_seq_features.py \
    --fasta_file           /data/infer/P99999.fasta \
    --esm_output_dir       /data/infer/esm \
    --prottrans_output_dir /data/infer/prottrans \
    --esm_model            facebook/esm2_t33_650M_UR50D \
    --prottrans_model      Rostlab/prot_t5_xl_uniref50
```

### 11.2 Step 2 — load the model and predict

```python
import numpy as np, torch
from protein_function.predict import load_model

# Reconstruct the model with the side-chain topology shape (12 channels, 15 combos)
model, go_terms, topo_scaler = load_model(
    model_dir="/data/function_model_sidechain/checkpoint-best",
    device="cuda",
    topo_channels=12,
    topo_n_filtrations=200,
    topo_n_combinations=15,
)

# Load precomputed features
topo = np.load("/data/infer/topo/P99999.npy")          # [12, 200, 15]
esm  = np.load("/data/infer/esm/P99999.npy")           # [1280]
prot = np.load("/data/infer/prottrans/P99999.npy")     # [1024]

# Scale topology features with the training-fitted scaler, then run the model
topo_scaled = topo_scaler.transform(topo.reshape(1, -1)).reshape(topo.shape)
topo_t = torch.tensor(topo_scaled, dtype=torch.float32).unsqueeze(0).to("cuda")
esm_t  = torch.tensor(esm,  dtype=torch.float32).unsqueeze(0).to("cuda")
prot_t = torch.tensor(prot, dtype=torch.float32).unsqueeze(0).to("cuda")

probs = model.predict_proba(topo_t, esm_t, prot_t)[0].cpu().numpy()  # [N_mf]

# Report terms above threshold
threshold = 0.3
hits = sorted(
    [(go_terms[i], float(p)) for i, p in enumerate(probs) if p >= threshold],
    key=lambda x: -x[1],
)
for go_term, prob in hits:
    print(f"{go_term}: {prob:.3f}")
```

> The exact scaler reshape may differ depending on how `fit_topo_scaler` flattens features;
> match whatever `ProteinFunctionDataset` does at training time (it applies the same scaler),
> so features are normalised identically for inference.

---

## 12. Troubleshooting

### 12.1 Import errors

**`ImportError: cannot import name 'find_pruneable_heads_and_indices'`**
```bash
pip install "transformers==4.24.0"
```

**`ModuleNotFoundError: No module named 'numpy'`** (or `prody`/`matplotlib`)
```bash
pip install -r protein_function/requirements_mf.txt
```

**`ERROR: pip's dependency resolver ... numpy X is incompatible`** (after
installing `prody` or `matplotlib` standalone)
```bash
# Re-pin everything to the compatible set instead of installing packages
# one at a time — prody/scipy need numpy<1.23, matplotlib>=3.6 needs numpy>=1.23
pip install -r protein_function/requirements_mf.txt --force-reinstall
```

**`ModuleNotFoundError: No module named 'protein_function'`**
Run all scripts from the repository root (`/home/user/TopoFormer`), not from subdirectories.

### 12.2 Feature extraction errors

**`MISSING_PDB P12345`** in precompute log
The script could not find any `.pdb` file for this protein. Check that:
- `<pdb_dir>/P12345/` exists and contains conformation `.pdb` files
- The protein ID in the IDs file matches the directory name exactly

**Topology features are all zeros**
For a combination this happens when fewer than two residues of the selected classes are
present (no edges can form). For a whole protein, check:
- PDB files contain `ATOM` records (not only `HETATM`)
- Residue names are standard three-letter codes (non-standard residues are skipped)
- At least some residues have resolvable side-chain heavy atoms or a Cα

**A residue's centroid looks wrong**
The centroid is the geometric mean of side-chain heavy atoms (backbone N, Cα, C, O, OXT and
all hydrogens excluded). Glycine and any residue with no side-chain heavy atoms fall back to
Cα. Only the blank/`A` alternate-location conformer is used.

### 12.3 Training issues

**`RuntimeError: Given groups=1, weight of size [256, 6, ...], expected input ... 12 channels`**
You trained without `--topo_channels 12`. Side-chain features are 12-channel; pass it.

**Loss is NaN from the first batch**
Check for:
- Infinite values in topology `.npy` files (from failed Laplacian computation)
- Labels file has NaN values

```python
import numpy as np, os
for f in os.listdir("/data/topo_features_sidechain"):
    arr = np.load(f"/data/topo_features_sidechain/{f}")
    if not np.isfinite(arr).all():
        print(f"Bad file: {f}")
```

**`RuntimeError: Expected input batch_size (X) to match target batch_size (Y)`**
The `collate_fn` requires all features to have the same shape. Ensure every `.npy` file in
`--topo_dir` has shape `[12, 200, 15]` (don't mix configurations in one directory).

### 12.4 Memory and runtime

The side-chain configuration is light: a few hundred centroids per protein means the
Persistent Laplacian is fast even at 200 filtration steps. If you still need to trim runtime
for very large proteins, reduce `--n_conformers` or `--dis_cutoff`.

---

## 13. Expected Results and Benchmarks

### 13.1 Swiss-Prot MF prediction

Expected performance on a held-out test set (30% sequence identity split):

| Method | Fmax | AUPR |
|--------|------|------|
| Sequence-only (ESM+ProtTrans) | 0.42–0.48 | 0.30–0.38 |
| **Side-chain topology + ESM + ProtTrans** | **0.48–0.58** | **0.38–0.48** |
| Motion-guided ensemble topology + sequence | 0.50–0.60 | 0.40–0.50 |
| DeepFRI (state-of-the-art, 2021) | 0.56 | 0.44 |

These are rough estimates for a well-tuned model trained on ~50k proteins. The residue-level
side-chain configuration trades some atom-level resolution for much faster feature extraction;
the ablation in §10.4 tells you the actual trade-off on your data.

### 13.2 Training time

| Hardware | Proteins | Epochs | Time |
|----------|----------|--------|------|
| 1× RTX 3090 (24 GB) | 10,000 | 50 | ~4 hours |
| 1× A100 (40 GB) | 50,000 | 50 | ~12 hours |
| 4× A100 | 100,000 | 50 | ~10 hours |

Topology feature extraction for the side-chain configuration typically runs in a few minutes
per protein on a single CPU core — far faster than the atom-level configurations.

---

## 14. Repository Layout

```
TopoFormer/
├── code_pkg/                          ← original TopoFormer core (do not modify)
│   ├── top_embedding/
│   │   └── SimplicialComplex_laplacian.py
│   ├── top_transformer/
│   │   ├── modeling_topt.py
│   │   └── configuration_topt.py
│   └── main_potein_ligand_topo_embedding.py
│
└── protein_function/                  ← protein function prediction code
    ├── GUIDE.md                       ← this guide (side-chain configuration)
    ├── README.md                      ← MSU HPCC step-by-step guide (all configs)
    ├── predict.py                     ← inference helpers (load_model, predict_function)
    ├── topo_extraction/
    │   ├── protein_only_topo_embedding.py     ← single PDB → [6, 200, 15]
    │   ├── ensemble_motion_topo_embedding.py  ← ensemble + motion → [12, 200, 121]
    │   └── sidechain_topo_embedding.py        ← side-chain centroids → [12, 200, 15]
    ├── data/
    │   └── protein_function_dataset.py        ← Dataset, scaler, label matrix
    ├── models/
    │   ├── mini_topt_config.py                ← TopTConfig factories
    │   │     get_mini_topt_config           → [6, 200, 15]
    │   │     get_ensemble_motion_topt_config → [12, 200, 121]
    │   │     get_sidechain_topt_config       → [12, 200, 15]
    │   └── modeling_topo_function.py          ← TopoFunctionModel
    ├── training/
    │   └── train_mf_prediction.py             ← HF Trainer subclass + CLI
    ├── evaluation/
    │   └── metrics.py                         ← Fmax, AUPR
    └── scripts/
        ├── preprocess_training_data.py        ← build splits + annotation/FASTA files
        ├── precompute_topo_features.py        ← batch topology extraction (all modes)
        ├── submit_topo_features_sidechain.sh  ← SLURM array launcher (side-chain)
        ├── sbatch_topo_features_sidechain.sh  ← SLURM array worker (side-chain)
        ├── submit_topo_features.sh            ← SLURM array launcher (motion ensemble)
        ├── sbatch_topo_features_array.sh      ← SLURM array worker (motion ensemble)
        ├── precompute_seq_features.py         ← batch ESM-2 + ProtTrans extraction
        └── sbatch_seq_features.sh             ← SLURM GPU job (sequence embeddings)
```

---

## 15. Quick-Start Checklist

- [ ] Environment set up with `transformers==4.24.0`
- [ ] PDB conformations available (10 per protein, in `<pdb_dir>/<protein_id>/`)
- [ ] Side-chain topology features pre-computed via `bash protein_function/scripts/submit_topo_features_sidechain.sh` → `[12, 200, 15]`
- [ ] Topology QA check passed (all shapes `(12, 200, 15)`, no inf/NaN)
- [ ] Topo scaler fitted and saved (or left to the training script to fit on the train set)
- [ ] ESM + ProtTrans features pre-computed: `python protein_function/scripts/precompute_seq_features.py ...`
- [ ] MF annotations in TSV format with `protein_id`, `go_term`, `namespace` columns
- [ ] Train/val/test splits at ≤30% sequence identity
- [ ] Training launched with `--topo_channels 12 --topo_n_combinations 15 --esm_dim 1280`
- [ ] Best checkpoint evaluated on the test set

---

*Guide written for branch `protein-motion-topology`, side-chain centroid topology configuration. For questions or issues, open a GitHub issue on `woldr001/TopoFormer`.*
