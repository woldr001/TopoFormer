#!/bin/bash --login
#SBATCH --job-name=mf_topo_features
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/mnt/research/woldring_lab/TopoFormer-MF/logs/topo_features_%j.out
#SBATCH --error=/mnt/research/woldring_lab/TopoFormer-MF/logs/topo_features_%j.err

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO=/mnt/research/woldring_lab/TopoFormer-MF/TopoFormer
DATASETS=/mnt/research/woldring_lab/TopoFormer-MF/datasets
PDB_DIR=/mnt/research/nodes/giacomo/asam_ensembles/protein_function_prediction/v0/sampling
NMA_PCA_DIR=/mnt/research/woldring_lab/TopoFormer-MF/nma_pca   # set to your NMA-PCA output dir
                                                                 # (or leave as non-existent path to
                                                                 # use fallback: all residues = motion)
TOPO_DIR=/mnt/research/woldring_lab/TopoFormer-MF/topo_features
N_WORKERS=32

# ── Environment ──────────────────────────────────────────────────────────────
export PATH="/mnt/home/woldring/.conda/envs/topoformer_mf/bin:$PATH"

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$TOPO_DIR" /mnt/research/woldring_lab/TopoFormer-MF/logs

# Build a single ID list from all three splits (process train + val + test)
ALL_IDS="$DATASETS/all_ids.txt"
cat "$DATASETS/train_ids.txt" "$DATASETS/val_ids.txt" "$DATASETS/test_ids.txt" \
    | sort -u > "$ALL_IDS"
echo "Processing $(wc -l < "$ALL_IDS") unique proteins."

echo "========================================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "CPUs:     $N_WORKERS"
echo "Started:  $(date)"
echo "========================================================"

cd "$REPO"

python protein_function/scripts/precompute_topo_features.py \
    --mode                  ensemble_motion \
    --pdb_dir               "$PDB_DIR" \
    --nma_pca_dir           "$NMA_PCA_DIR" \
    --output_dir            "$TOPO_DIR" \
    --pdb_list              "$ALL_IDS" \
    --n_conformers          10 \
    --top_motion_pct        0.20 \
    --fluctuation_sources   gnm pca \
    --n_slow_modes          3 \
    --dis_start             0.0 \
    --dis_cutoff            20.0 \
    --dis_step              0.1 \
    --ensemble_aggregation  mean_std \
    --n_workers             "$N_WORKERS"

echo "========================================================"
echo "Finished: $(date)"

# ── Quick QA ─────────────────────────────────────────────────────────────────
echo ""
echo "Topology features written: $(ls "$TOPO_DIR"/*.npy 2>/dev/null | wc -l)"

python - <<'PY'
import os, numpy as np
d = '/mnt/research/woldring_lab/TopoFormer-MF/topo_features'
files = [f for f in os.listdir(d) if f.endswith('.npy')]
bad_shape = [f for f in files
             if np.load(os.path.join(d, f)).shape != (12, 200, 121)]
bad_vals  = [f for f in files
             if not np.isfinite(np.load(os.path.join(d, f))).all()]
print(f"Topology: {len(files)} files")
print(f"  wrong shape (expect (12,200,121)): {len(bad_shape)}")
print(f"  contains inf/nan:                  {len(bad_vals)}")
PY
