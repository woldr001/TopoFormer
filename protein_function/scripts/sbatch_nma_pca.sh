#!/bin/bash --login
#SBATCH --job-name=mf_nma_pca
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=1-2441          # updated automatically by the launcher below
#SBATCH --output=/mnt/research/woldring_lab/TopoFormer-MF/logs/nma_pca_%A_%a.out
#SBATCH --error=/mnt/research/woldring_lab/TopoFormer-MF/logs/nma_pca_%A_%a.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/mnt/research/woldring_lab/TopoFormer-MF/TopoFormer
DATASETS=/mnt/research/woldring_lab/TopoFormer-MF/datasets
PDB_ROOT=/mnt/research/nodes/giacomo/asam_ensembles/protein_function_prediction/v0/sampling
NMA_PCA_ROOT=/mnt/research/woldring_lab/TopoFormer-MF/nma_pca
ASAM_SUBDIR=clustering.nodes_pipeline_v1

# ── Environment ───────────────────────────────────────────────────────────────
export PATH="/mnt/home/woldring/.conda/envs/topoformer_mf/bin:$PATH"

# ── Pick this task's protein ──────────────────────────────────────────────────
ID_FILE="$DATASETS/all_ids.txt"
PROTEIN_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ID_FILE")

if [[ -z "$PROTEIN_ID" ]]; then
    echo "No protein ID for task $SLURM_ARRAY_TASK_ID — exiting."
    exit 0
fi

PDB_DIR="$PDB_ROOT/$PROTEIN_ID/$ASAM_SUBDIR"
OUT_DIR="$NMA_PCA_ROOT/$PROTEIN_ID"

echo "Task $SLURM_ARRAY_TASK_ID | Protein: $PROTEIN_ID"
echo "PDB dir: $PDB_DIR"
echo "Output:  $OUT_DIR"

# ── Run ───────────────────────────────────────────────────────────────────────
python "$REPO/protein_function/scripts/run_nma_pca.py" \
    --pdb_dir      "$PDB_DIR" \
    --output_dir   "$OUT_DIR" \
    --n_conformers 10 \
    --n_slow_modes 3 \
    --n_pcs        2 \
    --gnm_cutoff   7.5
