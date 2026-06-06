#!/bin/bash
# Submit the topology-feature array job with the correct --array size.
# Run from the repo root:
#   bash protein_function/scripts/submit_topo_features.sh
#
# Optional environment overrides:
#   CHUNK_SIZE=10  bash protein_function/scripts/submit_topo_features.sh

REPO=/mnt/research/woldring_lab/TopoFormer-MF/TopoFormer
DATASETS=/mnt/research/woldring_lab/TopoFormer-MF/datasets
LOGS=/mnt/research/woldring_lab/TopoFormer-MF/logs

ALL_IDS="$DATASETS/all_ids.txt"
if [[ ! -f "$ALL_IDS" ]]; then
    cat "$DATASETS/train_ids.txt" "$DATASETS/val_ids.txt" "$DATASETS/test_ids.txt" \
        | sort -u > "$ALL_IDS"
    echo "Created $ALL_IDS"
fi

N=$(wc -l < "$ALL_IDS")
mkdir -p "$LOGS"

# 5 proteins per task → each task uses 5 CPUs (n_workers=5) running in parallel.
# Adjust CHUNK_SIZE if you hit QOS limits or want different granularity.
CHUNK_SIZE=${CHUNK_SIZE:-5}
N_TASKS=$(( (N + CHUNK_SIZE - 1) / CHUNK_SIZE ))

echo "Submitting topo-features array job for $N proteins as $N_TASKS tasks (CHUNK_SIZE=$CHUNK_SIZE) ..."

sbatch \
    --array="1-${N_TASKS}" \
    --export=ALL,CHUNK_SIZE=${CHUNK_SIZE} \
    "$REPO/protein_function/scripts/sbatch_topo_features_array.sh"
