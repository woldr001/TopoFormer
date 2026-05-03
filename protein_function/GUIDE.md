# TopoFormer Protein Function Prediction — How-To Guide

**Branch:** `claude/protein-topology-integration-ALbIE`  
**Task:** Predict Molecular Function (MF) Gene Ontology terms from protein 3D structures using motion-guided ensemble topology + sequence embeddings.

---

## Table of Contents

1. [Overview and Architecture](#1-overview-and-architecture)  
2. [Environment Setup](#2-environment-setup)  
3. [Data Acquisition](#3-data-acquisition)  
4. [Annotation File Format](#4-annotation-file-format)  
5. [Train / Val / Test Splits](#5-train--val--test-splits)  
6. [Running the Ensemble_NMA-PCA Pipeline](#6-running-the-ensemble_nma-pca-pipeline)  
7. [Pre-computing Topology Features](#7-pre-computing-topology-features)  
8. [Pre-computing Sequence Embeddings (ESM + ProtTrans)](#8-pre-computing-sequence-embeddings-esm--prottrans)  
9. [Training the Model](#9-training-the-model)  
10. [Monitoring and Early Stopping](#10-monitoring-and-early-stopping)  
11. [Evaluation](#11-evaluation)  
12. [Inference on New Proteins](#12-inference-on-new-proteins)  
13. [Troubleshooting](#13-troubleshooting)  
14. [Expected Results and Benchmarks](#14-expected-results-and-benchmarks)  
15. [Repository Layout](#15-repository-layout)  

---

## 1. Overview and Architecture

This tool predicts which **Molecular Function (MF) Gene Ontology terms** a protein performs, given:
- A set of 3D structural conformations (PDB files, e.g. from AlphaFold or MD simulation)
- Optional: pre-computed NMA/PCA motion profiles (from the Ensemble_NMA-PCA pipeline)
- Protein amino acid sequence (for ESM and ProtTrans embeddings)

### Architecture

```
PDB ensemble (10 conformations per protein)
        │
        ├─► Ensemble_NMA-PCA pipeline ──► GNM/ANM/PCA profiles (.npz)
        │
        └─► ensemble_motion_topo_embedding.py
                │
                ├── Identify top-20% motion residues from GNM/PCA scores
                ├── Treat motion residues as "ligand", static residues as "protein"
                ├── Compute cross-distances (motion↔static) only, sentinel for intra pairs
                ├── Run Persistent Laplacian dim-0 on 121 element combos × 200 filtration steps
                ├── Stack per-conformation [6, 200, 121] tensors
                └── Aggregate → mean + std → [12, 200, 121] saved as .npy

Protein sequence
        │
        ├─► ESM-2 (650M)  → mean-pool residues → [1280]  saved as .npy
        └─► ProtTrans T5  → mean-pool residues → [1024]  saved as .npy

                                ┌──────────────────────────────────────┐
                                │         TopoFunctionModel            │
   topo [B, 12, 200, 121] ─────►│  Mini TopT encoder → CLS → [B, 256] │
   esm  [B, 1280]         ─────►│  ESM proj → [B, 512]                │
   prot [B, 1024]         ─────►│  ProtTrans proj → [B, 512]          │
                                │  concat → [B, 1280]                 │
                                │  FC(1280→512) → GELU → FC(512→N_mf) │
                                │  BCEWithLogitsLoss                  │
                                └──────────────────────────────────────┘
                                              │
                                    MF GO term predictions
```

### Key design choices

| Feature | Value | Rationale |
|---------|-------|-----------|
| Motion-guided cross-pairs | 11 × 11 = 121 element combos | Mirrors protein-ligand scheme; encodes how mobile regions interact with the static scaffold |
| Filtration range | 0–20 Å, 200 steps of 0.1 Å | Captures backbone (~1.5 Å), H-bonds (~3 Å), secondary structure (~5–8 Å), domain contacts (~10–15 Å) |
| Ensemble aggregation | mean + std across 10 conformers | Mean = average topology; std = geometric variability hypothesised to carry functional signal |
| Topo encoder size | 4 layers, hidden=256 | ~3M params; appropriate for 30k–100k protein training sets to avoid overfitting |
| ESM model | ESM-2 650M (1280-dim) | Strong sequence representation without requiring GPU-intensive ESM-C 600M |

---

## 2. Environment Setup

### 2.1 Clone and install

```bash
# Clone the repository (if not already)
git clone https://github.com/woldr001/TopoFormer.git
cd TopoFormer
git checkout claude/protein-topology-integration-ALbIE

# Create a dedicated conda environment
conda create -n topoformer python=3.8
conda activate topoformer

# Core dependencies
pip install numpy scipy scikit-learn pandas torch torchvision
pip install "transformers==4.24.0"   # CRITICAL: newer versions break modeling_topt.py
pip install datasets accelerate
pip install biopython   # optional, for PDB utilities

# Sequence embedding models
pip install fair-esm   # for ESM-2
pip install sentencepiece   # for ProtTrans tokenizer
```

> **Why `transformers==4.24.0`?**  
> `modeling_topt.py` imports `find_pruneable_heads_and_indices` from `transformers.pytorch_utils`, a symbol removed in newer releases. Pin to 4.24.0 until the import is updated.

### 2.2 Verify installation

```bash
python -c "
import torch
from code_pkg.top_transformer.modeling_topt import TopTModel
from protein_function.models.mini_topt_config import get_ensemble_motion_topt_config
from protein_function.models.modeling_topo_function import TopoFunctionModel

config = get_ensemble_motion_topt_config()
model = TopoFunctionModel(config, num_mf_labels=100)
topo = torch.randn(2, 12, 200, 121)
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

A practical starting point is the **Swiss-Prot proteins with PDB/AlphaFold structures** — approximately 200,000 proteins with high-quality MF annotations, of which about half have experimental structures.

### 3.3 Structural conformations

Each protein should have **10 structural conformations** for the ensemble mode. Good sources:
- AlphaFold2 can be run with different random seeds or with `--num_models 5 --num_seeds 2`
- Short MD simulations (10–100 ns) sampled at equal time intervals
- Normal mode-generated conformations from ProDy (`prody nma`)
- Clustering of crystal structure depositions (multiple crystal forms)

If only a single PDB is available, use `protein_only` mode (outputs `[6, 200, 15]`) and set `--mode protein_only` in the precompute script.

### 3.4 Directory structure expected

```
/data/
├── pdbs/
│   └── P12345/
│       ├── cluster_repr_0.pdb
│       ├── cluster_repr_1.pdb
│       └── ... (10 files)
├── nma_pca/
│   └── P12345/
│       ├── anm_gnm_results.npz
│       └── pca_results.npz
├── topo_features/
│   └── P12345.npy   ← [12, 200, 121], written by precompute script
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

## 6. Running the Ensemble_NMA-PCA Pipeline

This step computes the GNM/ANM/PCA motion profiles that drive residue selection in the topology extractor.

### 6.1 Install Ensemble_NMA-PCA

```bash
git clone https://github.com/woldr001/Ensemble_NMA-PCA.git
cd Ensemble_NMA-PCA
pip install -r requirements.txt
```

### 6.2 Run on a single protein

```bash
python run_nma_pca.py \
    --pdb_dir    /data/pdbs/P12345 \
    --output_dir /data/nma_pca/P12345 \
    --n_modes    10 \
    --n_pcs      5
```

Expected outputs in `/data/nma_pca/P12345/`:
```
anm_gnm_results.npz   ← contains 'gnm_flucts_mean' [N_modes, N_residues] and 'resids' [N_residues]
pca_results.npz       ← contains 'pc1_profile' [N_residues], 'pc2_profile' [N_residues], 'resids'
```

### 6.3 Batch processing

```bash
# Process all proteins in parallel
ls /data/pdbs | xargs -I{} -P 16 python run_nma_pca.py \
    --pdb_dir    /data/pdbs/{} \
    --output_dir /data/nma_pca/{} \
    --n_modes    10 --n_pcs 5
```

### 6.4 What if NMA-PCA files are missing?

The topology extractor falls back to **all residues equally weighted** if the `.npz` files are absent — it will warn and proceed, using the top 20% of residues by backbone B-factor (from PDB ATOM records) as a proxy for motion. The model will still train, but the motion-guided signal will be absent.

---

## 7. Pre-computing Topology Features

### 7.1 Ensemble mode (recommended)

```bash
python protein_function/scripts/precompute_topo_features.py \
    --mode            ensemble_motion \
    --pdb_dir         /data/pdbs \
    --nma_pca_dir     /data/nma_pca \
    --output_dir      /data/topo_features \
    --n_conformers    10 \
    --top_motion_pct  0.20 \
    --fluctuation_sources gnm pca \
    --n_slow_modes    3 \
    --dis_start       0.0 \
    --dis_cutoff      20.0 \
    --dis_step        0.1 \
    --ensemble_aggregation mean_std \
    --n_workers       8
```

Output: one `<protein_id>.npy` file per protein, shape `[12, 200, 121]`, float32.

### 7.2 Single-structure mode (fallback)

```bash
python protein_function/scripts/precompute_topo_features.py \
    --mode       protein_only \
    --pdb_dir    /data/pdbs \
    --output_dir /data/topo_features_single \
    --n_workers  8
```

Output: shape `[6, 200, 15]` per protein.

### 7.3 Key arguments explained

| Argument | Default | Meaning |
|----------|---------|---------|
| `--top_motion_pct` | 0.20 | Top 20% of residues by GNM+PCA score are treated as "motion" (ligand-like) |
| `--fluctuation_sources gnm pca` | both | Which profile to use for identifying motion residues |
| `--n_slow_modes` | 3 | Number of slowest GNM modes to sum (more modes → smoother motion score) |
| `--motion_field` | None | If set, restrict static atoms to within N Å of any motion atom (speeds up large proteins) |
| `--ensemble_aggregation` | mean_std | `mean_std` → 12 channels; `mean_only` → 6 channels; `all` → 6×N channels |
| `--n_conformers` | 10 | Max number of PDB files to use per protein |
| `--overwrite` | False | Re-compute even if output file exists |

### 7.4 PDB file naming conventions

The script searches for conformation PDB files in this order under `<pdb_dir>/<protein_id>/`:
1. `cluster_repr_*.pdb`
2. `conf_*.pdb`
3. `model_*.pdb`
4. `frame_*.pdb`
5. Any `*.pdb` file (sorted, up to `n_conformers`)

### 7.5 Expected runtime

On a single CPU core: ~15–60 seconds per protein (depending on protein size and n_conformers).  
With 8 workers: ~3–8 minutes for 1,000 proteins.

### 7.6 Fit and save the topology scaler

The topology features must be normalised with a `StandardScaler + MinMaxScaler` pipeline fitted **only on the training set**:

```python
from protein_function.data.protein_function_dataset import fit_topo_scaler

train_ids = open("train_ids.txt").read().split()
scaler = fit_topo_scaler(
    protein_ids=train_ids,
    topo_dir="/data/topo_features",
    save_path="/data/topo_scaler.pkl",
)
```

---

## 8. Pre-computing Sequence Embeddings (ESM + ProtTrans)

### 8.1 Prepare a FASTA file

```
>P12345
MKTIIALSYIFCLVFA...
>Q99999
MAEGEITTFTALTEKF...
```

### 8.2 Run batch extraction

```bash
python protein_function/scripts/precompute_seq_features.py \
    --fasta           /data/sequences.fasta \
    --esm_output_dir  /data/esm_features \
    --pt_output_dir   /data/prottrans_features \
    --esm_model       facebook/esm2_t33_650M_UR50D \
    --pt_model        Rostlab/prot_t5_xl_uniref50 \
    --batch_size      16 \
    --device          cuda
```

Output: `<protein_id>.npy` files of shape `[1280]` (ESM) and `[1024]` (ProtTrans).

### 8.3 GPU memory requirements

| Model | Parameters | GPU VRAM (batch 16) |
|-------|-----------|---------------------|
| ESM-2 650M | 650M | ~6 GB |
| ProtTrans T5 | 3B | ~12 GB |

For limited VRAM, reduce `--batch_size` to 4 or 1. Both models can run on CPU (slow: ~30 s/protein for ProtTrans).

### 8.4 ESM embedding dimension

The default ESM model (`esm2_t33_650M_UR50D`) outputs 1280-dim embeddings. The `TopoFunctionModel` is initialised with `esm_dim=1280` by default. If you use a different ESM variant:

| Model | Dim |
|-------|-----|
| `esm2_t6_8M_UR50D` | 320 |
| `esm2_t12_35M_UR50D` | 480 |
| `esm2_t30_150M_UR50D` | 640 |
| `esm2_t33_650M_UR50D` | 1280 |
| `esm2_t36_3B_UR50D` | 2560 |

Pass the correct dim when constructing the model: `TopoFunctionModel(config, num_mf_labels=N, esm_dim=1280)`.

---

## 9. Training the Model

### 9.1 Directory prerequisites

Before training, verify you have:
```
/data/topo_features/     # .npy files [12, 200, 121]
/data/esm_features/      # .npy files [1280]
/data/prottrans_features/ # .npy files [1024]
/data/topo_scaler.pkl    # fitted scaler
/data/annotations/mf_annotations.tsv
/data/train_ids.txt
/data/val_ids.txt
/data/test_ids.txt
```

### 9.2 Launch training

```bash
python protein_function/training/train_mf_prediction.py \
    --topo_dir          /data/topo_features \
    --esm_dir           /data/esm_features \
    --prottrans_dir     /data/prottrans_features \
    --label_file        /data/annotations/mf_annotations.tsv \
    --topo_scaler_path  /data/topo_scaler.pkl \
    --train_ids_file    /data/train_ids.txt \
    --val_ids_file      /data/val_ids.txt \
    --topo_feature_mode ensemble_motion \
    --output_dir        /data/function_model \
    --num_train_epochs  50 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size  64 \
    --learning_rate     1e-4 \
    --weight_decay      0.01 \
    --warmup_ratio      0.05 \
    --lr_scheduler_type cosine \
    --eval_strategy     epoch \
    --save_strategy     epoch \
    --load_best_model_at_end true \
    --metric_for_best_model fmax \
    --fp16
```

### 9.3 Hyperparameter recommendations

| Hyperparameter | Recommended range | Notes |
|---------------|-------------------|-------|
| `learning_rate` | 5e-5 – 2e-4 | Lower if loss oscillates; higher if convergence is slow |
| `batch_size` | 32–128 | Larger batches → more stable gradients for multi-label tasks |
| `num_train_epochs` | 50–100 | Early stopping at patience=10 prevents overfitting |
| `warmup_ratio` | 0.05–0.10 | Longer warmup helps with the topology encoder |
| `weight_decay` | 0.01 | AdamW regularisation |
| `fusion_dropout` | 0.2–0.4 | Increase if val Fmax plateaus while train Fmax rises |

### 9.4 Multi-GPU training

```bash
torchrun --nproc_per_node=4 protein_function/training/train_mf_prediction.py \
    [... same args ...]
```

### 9.5 Two-stage fine-tuning (optional)

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

### 9.6 Loading pretrained TopoFormer weights

If you have a pretrained `pytorch_model.bin` from the original TopoFormer:

```python
from protein_function.models.mini_topt_config import (
    get_ensemble_motion_topt_config,
    load_pretrained_topt_weights,
)
from code_pkg.top_transformer.modeling_topt import TopTModel

config = get_ensemble_motion_topt_config()
encoder = TopTModel(config)
skipped = load_pretrained_topt_weights(encoder, "/path/to/pretrained_checkpoint/")
print(f"Skipped keys (incompatible shapes): {skipped}")
```

The patch embedding Conv2d and positional embeddings will be re-initialised (shape incompatibility); all compatible attention/LayerNorm weights are sliced to fit the smaller hidden dimension (256 vs 768).

---

## 10. Monitoring and Early Stopping

### 10.1 TensorBoard

```bash
tensorboard --logdir /data/function_model/runs
```

Key metrics to monitor:
- `eval/fmax` (primary — higher is better; should reach 0.45–0.60 on Swiss-Prot MF)
- `eval/aupr` (secondary)
- `train/loss` vs `eval/loss` (check for overfitting gap)

### 10.2 Early stopping

The trainer uses `EarlyStoppingCallback` with `patience=10` on `eval/fmax`. If val Fmax does not improve for 10 consecutive epochs, training stops and the best checkpoint is restored.

To adjust patience:
```bash
... --early_stopping_patience 15 ...
```

### 10.3 Signs of trouble

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Loss NaN from epoch 1 | Learning rate too high | Halve `--learning_rate` |
| Val Fmax ≤ 0.10 after 20 epochs | Topo features all-zeros (missing .npy files) | Check `--topo_dir` paths |
| Val Fmax stagnates at ~0.35 while train Fmax → 0.70 | Overfitting | Increase `--fusion_dropout 0.4`, reduce `hidden_size` |
| CUDA OOM | Batch too large | Halve `--per_device_train_batch_size` |
| `transformers` import error | Wrong transformers version | `pip install "transformers==4.24.0"` |

---

## 11. Evaluation

### 11.1 Standard CAFA metrics

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

### 11.2 Run evaluation on a saved checkpoint

```bash
python protein_function/training/train_mf_prediction.py \
    ... \
    --do_train false \
    --do_eval  true \
    --do_predict true \
    --test_ids_file /data/test_ids.txt \
    --resume_from_checkpoint /data/function_model/checkpoint-best
```

Outputs:
- `test_results.json` — Fmax, AUPR, per-term AP
- `test_predictions.npy` — raw logit matrix `[N_test, N_terms]`

### 11.3 Per-term analysis

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

---

## 12. Inference on New Proteins

### 12.1 Using the predict script

```bash
python protein_function/predict.py \
    --model_dir        /data/function_model/checkpoint-best \
    --protein_id       P99999 \
    --pdb_dir          /data/pdbs/P99999 \
    --nma_pca_dir      /data/nma_pca/P99999 \
    --esm_npy          /data/esm_features/P99999.npy \
    --prottrans_npy    /data/prottrans_features/P99999.npy \
    --threshold        0.3 \
    --top_k            10
```

### 12.2 Python API

```python
from protein_function.predict import load_model, predict_function

model, go_terms, topo_scaler = load_model(
    model_dir="/data/function_model/checkpoint-best",
    device="cuda",
)

predictions = predict_function(
    model=model,
    go_terms=go_terms,
    topo_scaler=topo_scaler,
    protein_pdb_dir="/data/pdbs/P99999",
    nma_pca_dir="/data/nma_pca/P99999",
    esm_npy="/data/esm_features/P99999.npy",
    prottrans_npy="/data/prottrans_features/P99999.npy",
    threshold=0.3,
)

for go_term, prob in predictions:
    print(f"{go_term}: {prob:.3f}")
```

### 12.3 Batch inference

```bash
python protein_function/predict.py \
    --model_dir   /data/function_model/checkpoint-best \
    --batch_file  /data/test_ids.txt \
    --pdb_dir     /data/pdbs \
    --nma_pca_dir /data/nma_pca \
    --esm_dir     /data/esm_features \
    --prottrans_dir /data/prottrans_features \
    --output_csv  /data/predictions.csv \
    --threshold   0.3
```

---

## 13. Troubleshooting

### 13.1 Import errors

**`ImportError: cannot import name 'find_pruneable_heads_and_indices'`**
```bash
pip install "transformers==4.24.0"
```

**`ModuleNotFoundError: No module named 'numpy'`**
```bash
pip install numpy scipy scikit-learn
```

**`ModuleNotFoundError: No module named 'protein_function'`**  
Run all scripts from the repository root (`/home/user/TopoFormer`), not from subdirectories.

### 13.2 Feature extraction errors

**`MISSING_PDB P12345`** in precompute log  
The script could not find any `.pdb` file for this protein. Check that:
- `<pdb_dir>/P12345/` exists and contains `*.pdb` files
- The protein ID in `pdb_list.txt` matches the directory name exactly

**Topology features are all zeros**  
This happens when all atoms of the selected element combination are absent, or when no motion residues remain after filtering. Check:
- PDB files contain `ATOM` records (not `HETATM`)
- PDB has standard elements (C, N, O, S) in column 76-78 or parseable atom names
- NMA-PCA `.npz` files exist in `nma_pca_dir`

**`WARNING: [topo] anm_gnm_results.npz not found`**  
The extractor falls back to equal motion weighting. Run the Ensemble_NMA-PCA pipeline first (Section 6), or pass `--fluctuation_sources contact_map` as a fallback.

### 13.3 Training issues

**Loss is NaN from the first batch**  
Check for:
- Infinite values in topology `.npy` files (from failed Laplacian computation)
- Labels file has NaN values

```python
import numpy as np, os
for f in os.listdir("/data/topo_features"):
    arr = np.load(f"/data/topo_features/{f}")
    if not np.isfinite(arr).all():
        print(f"Bad file: {f}")
```

**`RuntimeError: Expected input batch_size (X) to match target batch_size (Y)`**  
The `collate_fn` in `protein_function_dataset.py` requires all features to have the same shape. Ensure all `.npy` files have shape `[12, 200, 121]` (ensemble mode) or `[6, 200, 15]` (single mode).

### 13.4 Memory issues

For large proteins (>1000 residues), the persistent Laplacian can consume significant memory. Mitigations:
- Use `--motion_field 15.0` to restrict static atoms to within 15 Å of motion atoms
- Reduce `--n_conformers` to 5
- Use `--dis_cutoff 15.0` for faster extraction (captures most relevant interactions)

---

## 14. Expected Results and Benchmarks

### 14.1 Swiss-Prot MF prediction

Expected performance on a held-out test set (30% sequence identity split):

| Method | Fmax | AUPR |
|--------|------|------|
| Sequence-only (ESM+ProtTrans) | 0.42–0.48 | 0.30–0.38 |
| Topology-only (single PDB) | 0.30–0.38 | 0.22–0.30 |
| **This model (ensemble topo + ESM + ProtTrans)** | **0.50–0.60** | **0.40–0.50** |
| DeepFRI (state-of-the-art, 2021) | 0.56 | 0.44 |

These are rough estimates for a well-tuned model trained on ~50k proteins. Results vary significantly with training set size, structural quality, and hyperparameter tuning.

### 14.2 CAFA3 benchmark

The official CAFA3 MF benchmark uses ~450 test proteins. Expected:
- Fmax ≈ 0.52–0.62 (comparable to CAFA3 top methods if trained on the full Swiss-Prot)

### 14.3 Training time

| Hardware | Proteins | Epochs | Time |
|----------|----------|--------|------|
| 1× RTX 3090 (24 GB) | 10,000 | 50 | ~4 hours |
| 1× A100 (40 GB) | 50,000 | 50 | ~12 hours |
| 4× A100 | 100,000 | 50 | ~10 hours |

---

## 15. Repository Layout

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
└── protein_function/                  ← new protein function prediction code
    ├── GUIDE.md                       ← this file
    ├── predict.py                     ← end-to-end inference entry point
    ├── topo_extraction/
    │   ├── protein_only_topo_embedding.py    ← single PDB → [6, 200, 15]
    │   └── ensemble_motion_topo_embedding.py ← ensemble + motion → [12, 200, 121]
    ├── data/
    │   └── protein_function_dataset.py       ← Dataset, scaler, label matrix
    ├── models/
    │   ├── mini_topt_config.py               ← TopTConfig factories
    │   └── modeling_topo_function.py         ← TopoFunctionModel
    ├── training/
    │   └── train_mf_prediction.py            ← HF Trainer subclass + CLI
    ├── evaluation/
    │   └── metrics.py                        ← Fmax, AUPR
    └── scripts/
        ├── precompute_topo_features.py       ← batch topology extraction
        └── precompute_seq_features.py        ← batch ESM + ProtTrans extraction
```

---

## Quick-Start Checklist

- [ ] Environment set up with `transformers==4.24.0`
- [ ] PDB conformations available (10 per protein, in `<pdb_dir>/<protein_id>/`)
- [ ] Ensemble_NMA-PCA pipeline run; `anm_gnm_results.npz` + `pca_results.npz` ready
- [ ] Topology features pre-computed: `python protein_function/scripts/precompute_topo_features.py --mode ensemble_motion ...`
- [ ] Topo scaler fitted and saved: `fit_topo_scaler(train_ids, topo_dir, save_path)`
- [ ] ESM + ProtTrans features pre-computed: `python protein_function/scripts/precompute_seq_features.py ...`
- [ ] MF annotations in TSV format with `protein_id`, `go_term`, `namespace` columns
- [ ] Train/val/test splits at ≤30% sequence identity
- [ ] Training launched: `python protein_function/training/train_mf_prediction.py ...`
- [ ] Best checkpoint evaluated on test set

---

*Guide written for branch `claude/protein-topology-integration-ALbIE`. For questions or issues, open a GitHub issue on `woldr001/topoformer`.*
