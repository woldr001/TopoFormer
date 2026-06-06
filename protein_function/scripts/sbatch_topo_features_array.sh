#!/bin/bash --login
#SBATCH --job-name=mf_topo
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=25G
#SBATCH --output=/mnt/research/woldring_lab/TopoFormer-MF/logs/topo_%A_%a.out
#SBATCH --error=/mnt/research/woldring_lab/TopoFormer-MF/logs/topo_%A_%a.err
# NOTE: --array is set dynamically by submit_topo_features.sh — do not set it here.

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/mnt/research/woldring_lab/TopoFormer-MF/TopoFormer
DATASETS=/mnt/research/woldring_lab/TopoFormer-MF/datasets
PDB_DIR=/mnt/research/nodes/giacomo/asam_ensembles/protein_function_prediction/v0/sampling
NMA_PCA_DIR=/mnt/research/woldring_lab/TopoFormer-MF/nma_pca
TOPO_DIR=/mnt/research/woldring_lab/TopoFormer-MF/topo_features

# Each array task processes CHUNK_SIZE proteins in parallel (n_workers=CHUNK_SIZE).
# submit_topo_features.sh sets CHUNK_SIZE and calculates the number of tasks.
CHUNK_SIZE=${CHUNK_SIZE:-5}

# ── Environment ───────────────────────────────────────────────────────────────
export PATH="/mnt/home/woldring/.conda/envs/topoformer_mf/bin:$PATH"

mkdir -p "$TOPO_DIR" /mnt/research/woldring_lab/TopoFormer-MF/logs

# ── Select this task's chunk of proteins ──────────────────────────────────────
ID_FILE="$DATASETS/all_ids.txt"
TOTAL=$(wc -l < "$ID_FILE")

START=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK_SIZE + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( END > TOTAL ? TOTAL : END ))

echo "Task $SLURM_ARRAY_TASK_ID | Lines $START–$END of $TOTAL | $(date)"
echo "Node: $(hostname) | CPUs: $CHUNK_SIZE"

# Write the chunk's protein IDs to a task-local temp file
TMP_IDS=$(mktemp /tmp/topo_ids_${SLURM_ARRAY_TASK_ID}_XXXX.txt)
trap "rm -f $TMP_IDS" EXIT
sed -n "${START},${END}p" "$ID_FILE" > "$TMP_IDS"

echo "Proteins in this chunk:"
cat "$TMP_IDS"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
python "$REPO/protein_function/scripts/precompute_topo_features.py" \
    --mode                  ensemble_motion \
    --pdb_dir               "$PDB_DIR" \
    --nma_pca_dir           "$NMA_PCA_DIR" \
    --output_dir            "$TOPO_DIR" \
    --pdb_list              "$TMP_IDS" \
    --n_conformers          10 \
    --top_motion_pct        0.20 \
    --fluctuation_sources   gnm pca \
    --n_slow_modes          3 \
    --dis_start             0.0 \
    --dis_cutoff            20.0 \
    --dis_step              0.1 \
    --ensemble_aggregation  mean_std \
    --n_workers             "$CHUNK_SIZE"

echo "Task $SLURM_ARRAY_TASK_ID done | $(date)"
