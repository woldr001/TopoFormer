#!/bin/bash --login
#SBATCH --job-name=mf_seq_features
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mnt/research/woldring_lab/TopoFormer-MF/logs/seq_features_%j.out
#SBATCH --error=/mnt/research/woldring_lab/TopoFormer-MF/logs/seq_features_%j.err

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO=/mnt/research/woldring_lab/TopoFormer-MF/TopoFormer
DATASETS=/mnt/research/woldring_lab/TopoFormer-MF/datasets
ESM_DIR=/mnt/research/woldring_lab/TopoFormer-MF/esm_features
PT_DIR=/mnt/research/woldring_lab/TopoFormer-MF/prottrans_features

# ── Environment ──────────────────────────────────────────────────────────────
export PATH="/mnt/home/woldring/.conda/envs/topoformer_mf/bin:$PATH"

# Redirect HuggingFace cache to node-local temp (avoids slow networked storage)
JOB_TMP=${TMPDIR:-/tmp/woldring_mf_${SLURM_JOB_ID}}
mkdir -p "$JOB_TMP"
export HF_HOME="$JOB_TMP/hf"
export TRANSFORMERS_CACHE="$JOB_TMP/hf/transformers"
export HUGGINGFACE_HUB_CACHE="$JOB_TMP/hf/hub"

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$ESM_DIR" "$PT_DIR" /mnt/research/woldring_lab/TopoFormer-MF/logs

echo "========================================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started:  $(date)"
echo "========================================================"

cd "$REPO"

python protein_function/scripts/precompute_seq_features.py \
    --fasta_file           "$DATASETS/sequences.fasta" \
    --esm_output_dir       "$ESM_DIR" \
    --prottrans_output_dir "$PT_DIR" \
    --esm_model            facebook/esm2_t33_650M_UR50D \
    --prottrans_model      Rostlab/prot_t5_xl_uniref50 \
    --batch_size           8 \
    --device               cuda

echo "========================================================"
echo "Finished: $(date)"

# ── Quick QA ─────────────────────────────────────────────────────────────────
echo ""
echo "ESM features written:      $(ls "$ESM_DIR"/*.npy 2>/dev/null | wc -l)"
echo "ProtTrans features written: $(ls "$PT_DIR"/*.npy 2>/dev/null | wc -l)"

python - <<'PY'
import os, numpy as np
for d, label, expected in [
    ('/mnt/research/woldring_lab/TopoFormer-MF/esm_features',       'ESM',      1280),
    ('/mnt/research/woldring_lab/TopoFormer-MF/prottrans_features', 'ProtTrans', 1024),
]:
    files = [f for f in os.listdir(d) if f.endswith('.npy')]
    bad = [f for f in files if np.load(os.path.join(d, f)).shape != (expected,)]
    print(f"{label}: {len(files)} files, {len(bad)} wrong shape (expected ({expected},))")
PY
