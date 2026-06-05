#!/bin/bash --login
#SBATCH --job-name=mf_nma_pca
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/mnt/research/woldring_lab/TopoFormer-MF/logs/nma_pca_%A_%a.out
#SBATCH --error=/mnt/research/woldring_lab/TopoFormer-MF/logs/nma_pca_%A_%a.err
# NOTE: --array is set dynamically by submit_nma_pca.sh — do not set it here.

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/mnt/research/woldring_lab/TopoFormer-MF/TopoFormer
DATASETS=/mnt/research/woldring_lab/TopoFormer-MF/datasets
PDB_ROOT=/mnt/research/nodes/giacomo/asam_ensembles/protein_function_prediction/v0/sampling
NMA_PCA_ROOT=/mnt/research/woldring_lab/TopoFormer-MF/nma_pca
ASAM_SUBDIR=clustering.nodes_pipeline_v1

# Each array task processes CHUNK_SIZE proteins sequentially.
# submit_nma_pca.sh calculates the number of array tasks accordingly.
CHUNK_SIZE=${CHUNK_SIZE:-5}

# ── Environment ───────────────────────────────────────────────────────────────
export PATH="/mnt/home/woldring/.conda/envs/topoformer_mf/bin:$PATH"

# ── Process this task's chunk of proteins ─────────────────────────────────────
ID_FILE="$DATASETS/all_ids.txt"
TOTAL=$(wc -l < "$ID_FILE")

START=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK_SIZE + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( END > TOTAL ? TOTAL : END ))

echo "Task $SLURM_ARRAY_TASK_ID | Lines $START–$END of $TOTAL | $(date)"

for LINE in $(seq "$START" "$END"); do
    PROTEIN_ID=$(sed -n "${LINE}p" "$ID_FILE")
    [[ -z "$PROTEIN_ID" ]] && continue

    PDB_DIR="$PDB_ROOT/$PROTEIN_ID/$ASAM_SUBDIR"
    OUT_DIR="$NMA_PCA_ROOT/$PROTEIN_ID"

    echo "  [$LINE/$TOTAL] $PROTEIN_ID"

    python "$REPO/protein_function/scripts/run_nma_pca.py" \
        --pdb_dir      "$PDB_DIR" \
        --output_dir   "$OUT_DIR" \
        --n_conformers 10 \
        --n_slow_modes 3 \
        --n_pcs        2 \
        --gnm_cutoff   7.5 || echo "  WARNING: $PROTEIN_ID failed, continuing."
done

echo "Task $SLURM_ARRAY_TASK_ID done | $(date)"
