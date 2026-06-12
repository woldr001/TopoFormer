#!/bin/bash --login
#SBATCH --job-name=mf_topo_sc
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --output=/mnt/research/woldring_lab/TopoFormer-MF/logs/topo_sc_%A_%a.out
#SBATCH --error=/mnt/research/woldring_lab/TopoFormer-MF/logs/topo_sc_%A_%a.err
# NOTE: --array is set dynamically by submit_topo_features_sidechain.sh — do not set it here.
#
# Side-chain centroid configuration:
#   Each residue → geometric centroid of its side-chain heavy atoms (Gly → Cα)
#   15 side-chain-type combos (flat list): 4 singles + 6 pairs + 4 triples + 1 quad
#   --dis_start 0.0 --dis_cutoff 40.0 --dis_step 0.2  → 200 filtration steps
#   --n_conformers 10                                 → all 10 conformations
#   mean + std ensemble aggregation                   → 12 channels
#
# Output shape: [12, 200, 15] per protein
# No NMA-PCA profiles are required for this mode.

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/mnt/research/woldring_lab/TopoFormer-MF/TopoFormer
DATASETS=/mnt/research/woldring_lab/TopoFormer-MF/datasets
PDB_DIR=/mnt/research/nodes/giacomo/asam_ensembles/protein_function_prediction/v0/sampling
TOPO_DIR=/mnt/research/woldring_lab/TopoFormer-MF/topo_features_sidechain

# Each array task processes CHUNK_SIZE proteins in parallel (n_workers=CHUNK_SIZE).
# submit_topo_features_sidechain.sh sets CHUNK_SIZE and calculates the number of tasks.
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
TMP_IDS=$(mktemp /tmp/topo_sc_ids_${SLURM_ARRAY_TASK_ID}_XXXX.txt)
trap "rm -f $TMP_IDS" EXIT
sed -n "${START},${END}p" "$ID_FILE" > "$TMP_IDS"

echo "Proteins in this chunk:"
cat "$TMP_IDS"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
python "$REPO/protein_function/scripts/precompute_topo_features.py" \
    --mode                  sidechain_centroid \
    --pdb_dir               "$PDB_DIR" \
    --output_dir            "$TOPO_DIR" \
    --pdb_list              "$TMP_IDS" \
    --n_conformers          10 \
    --dis_start             0.0 \
    --dis_cutoff            40.0 \
    --dis_step              0.2 \
    --ensemble_aggregation  mean_std \
    --n_workers             "$CHUNK_SIZE"

echo "Task $SLURM_ARRAY_TASK_ID done | $(date)"
