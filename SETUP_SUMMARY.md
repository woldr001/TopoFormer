# TopoFormer Setup Guide — MSU HPCC (AMD24-H200 Nodes)

This guide documents how to set up and run TopoFormer for protein-ligand binding affinity
prediction on the MSU HPCC. It was written based on a working installation on `dev-amd24-h200`
in April 2025 and captures every pitfall encountered along the way.

---

## Overview

TopoFormer predicts the binding affinity (pKd/pKi) of a protein-ligand complex from 3D
structural coordinates. It works in two steps:

1. **Feature extraction**: Convert a protein PDB file + ligand MOL2/SDF file into a
   topological feature array (`.npy`) using persistent simplicial complex Laplacians.
2. **Prediction**: Feed the feature array through a fine-tuned Transformer model to get
   a binding affinity score.

This branch (`claude/topoformer-binding-affinity-fvoUx`) adds:
- `run_topo.sh` — SLURM job script for feature extraction
- `predict_single.py` — prediction script for a single complex
- `environment.yml` — minimal conda environment spec
- `SETUP_SUMMARY.md` — this file

---

## Prerequisites

- MSU HPCC account with access to AMD24-H200 compute nodes
- Your protein structure as a **PDB file** (binding pocket, not full protein)
- Your ligand structure as a **MOL2 or SDF file**

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/woldr001/TopoFormer.git
cd TopoFormer
git fetch origin
git checkout claude/topoformer-binding-affinity-fvoUx
```

---

## Step 2: Create the Conda Environment

**Important**: Do not use your base `~/anaconda3/bin/conda` directly on AMD24-H200 nodes —
it is too old and will crash due to OpenSSL conflicts with the system software stack.
Always use `module load Conda/3` first.

### Option A — From `environment.yml` (recommended)

```bash
module load Conda/3
conda env create -f environment.yml
conda activate topoformer
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### Option B — Manual install (if Option A causes solver conflicts)

```bash
module load Conda/3
conda create -n topoformer python=3.9 -y
conda activate topoformer
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.24.0 numpy==1.21.5 scipy==1.7.3 scikit-learn==1.0.2
```

### Verify the installation

```bash
python -c "import torch, transformers, numpy; print(torch.__version__, transformers.__version__, numpy.__version__)"
```

Expected output: `2.6.0+cu124  4.24.0  1.21.5`

> **Note on CUDA version**: The commands above use CUDA 12.4 (`cu124`), which matches
> the AMD24-H200 nodes. If you are on different nodes, adjust accordingly:
> - CUDA 12.1 → `https://download.pytorch.org/whl/cu121`
> - CUDA 11.8 → `https://download.pytorch.org/whl/cu118`

---

## Step 3: Download the Fine-Tuned Models

Three example CASF-2016 scoring models are provided by the Wei Lab:

```bash
wget https://weilab.math.msu.edu/Downloads/TopoFormer/finetuned_casf2016_3models.tar.gz
tar -xzf finetuned_casf2016_3models.tar.gz
```

This creates:
```
shared_finetuned_model_casf_2016_at_11212025/
  model_cls_0/
  model_cls_1/
  model_cls_2/
```

Each model directory is **self-contained** — it includes both the pre-trained encoder
weights and the fine-tuned regression head. No separate pre-trained model download is
needed for inference.

The scaler file needed for inference is already in the repo at:
`code_pkg/pretrain_data_standard_minmax_6channel_large.sav`

---

## Step 4: Generate Topological Features

Edit `run_topo.sh` to point to your protein and ligand files:

```bash
# In run_topo.sh, update these two lines:
--protein_file "/path/to/your_protein_pocket.pdb" \
--ligand_file  "/path/to/your_ligand.mol2" \
```

Then submit the job **from inside your TopoFormer directory**:

```bash
cd ~/TopoFormer
sbatch run_topo.sh
```

This produces `./output/<protein_id>.npy` (shape: `[6, 100, 143]`).

**Do not run this on the dev/login node** — the Laplacian eigenvalue computation is
memory-intensive and will be killed by the node's OOM limits.

> **Why `cd ~/TopoFormer` before `sbatch`?** The script uses `$SLURM_SUBMIT_DIR` (the
> directory where you called `sbatch`) as its working directory. This avoids a subtle HPC
> pitfall: `~/` resolves to different NFS mounts on dev nodes (`/mnt/home/...`) vs. compute
> nodes (`/mnt/ffs24/home/...`), so hardcoding `cd ~/TopoFormer` inside the script causes
> `mkdir -p ./output` and the Python process to land in different directories. Calling
> `sbatch` from the correct directory sidesteps this entirely.

The output feature file name is derived automatically from your protein filename:
the script strips everything from `_protein` onward. For example,
`1a4k_protein_mono.pdb` → `output/1a4k.npy`.

---

## Step 5: Predict Binding Affinity

Edit `predict_single.py` to set your paths:

```python
FEATURE_PATH = "./output/1a4k.npy"           # output from Step 4
SCALER_PATH  = "./code_pkg/pretrain_data_standard_minmax_6channel_large.sav"
MODEL_DIR    = "/path/to/shared_finetuned_model_casf_2016_at_11212025"
MODEL_NAMES  = ["model_cls_0", "model_cls_1", "model_cls_2"]
```

Then run:

```bash
module load Conda/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate topoformer
python predict_single.py
```

> **Why `source conda.sh`?** `module load Conda/3` puts the `conda` binary on your PATH,
> but does not install the shell hooks that make `conda activate` switch environments.
> Without sourcing `conda.sh`, `conda activate` appears to succeed but the previously-active
> environment remains in effect. This is the same fix applied inside `run_topo.sh`.

Example output:
```
  model_cls_0: 7.412
  model_cls_1: 7.389
  model_cls_2: 7.501

Ensemble mean (pKd/pKi): 7.434
```

### Interpreting the output

- Units: **log Ka** (pKd or pKi). Higher = tighter binding.
- The CASF-2016 test set spans roughly 2–12 pKd units.
- Typical drug-like compounds bind in the 7–9 range (Kd ~1–100 nM).
- The ensemble mean (average of 3 models) is more reliable than any single model.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Killed` during feature extraction | OOM on dev/login node | Use `sbatch run_topo.sh` |
| `AttributeError: module 'lib' has no attribute 'X509_V_FLAG_CB_ISSUER_CHECK'` | Base `~/anaconda3` conflicts with HPC system SSL libs | Run `module load Conda/3` before any conda command |
| `conda: command not found` | Conda not in PATH on this node | Run `module load Conda/3` |
| `Could not find conda environment: topoformer` | Environment not initialized in batch job | Add `module load Conda/3` before `conda activate` in your SLURM script |
| `numpy==1.21.5` install fails with Python version error | Conda created the env with Python ≥3.11 instead of 3.9 | Recreate: `conda remove -n topoformer --all -y && conda create -n topoformer python=3.9 -y` |
| `FileNotFoundError: ./output/1a4k.npy` in batch job despite `mkdir -p ./output` in script | Dev and compute nodes mount home at different paths (`/mnt/home` vs `/mnt/ffs24/home`), so the script's working directory differed from where Python ran | Always `cd ~/TopoFormer` before running `sbatch run_topo.sh` so `$SLURM_SUBMIT_DIR` resolves correctly |
| Wrong conda env used in batch job (e.g., `foundry_clean` instead of `topoformer`) | `module load Conda/3` adds `conda` to PATH but does not install shell hooks; `conda activate` silently fails and keeps the prior env | Add `source "$(conda info --base)/etc/profile.d/conda.sh"` after `module load Conda/3` and before `conda activate` — already present in `run_topo.sh` |
| Conda YAML env create fails with solver conflicts | Full-export YAML (200+ pinned packages) is machine-specific | Use `environment.yml` in this repo (minimal spec) or the manual install in Option B above |
| `torch` not found after `conda env create -f environment.yml` | PyTorch requires a custom index URL, excluded from YAML | Run `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124` after activating the env |

---

## File Reference

| File | Purpose |
|------|---------|
| `run_topo.sh` | SLURM script: generate topological features for one complex |
| `predict_single.py` | Run ensemble prediction on a single complex |
| `environment.yml` | Minimal conda environment spec (Python 3.9 + deps, no torch) |
| `code_pkg/main_potein_ligand_topo_embedding.py` | Feature extraction entry point |
| `code_pkg/top_transformer/modeling_topt.py` | Transformer model architecture |
| `code_pkg/pretrain_data_standard_minmax_6channel_large.sav` | Scaler for inference (in repo) |
| `examples/protein_ligand_complex/1a1e/` | Example protein/ligand input files |

---

## Citation

If you use TopoFormer in your work, please cite:

> Chen, Dong, Jian Liu, and Guo-Wei Wei. "Multiscale topology-enabled structure-to-sequence
> transformer for protein-ligand interaction predictions." *Nature Machine Intelligence* (2024).
