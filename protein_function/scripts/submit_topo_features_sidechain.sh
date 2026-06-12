#!/bin/bash
# Submit the side-chain-centroid topology-feature array job.
# Run from the repo root:
#   bash protein_function/scripts/submit_topo_features_sidechain.sh
#
# Optional environment overrides:
#   CHUNK_SIZE=10  bash protein_function/scripts/submit_topo_features_sidechain.sh
#
# Outputs go to /mnt/research/woldring_lab/TopoFormer-MF/topo_features_sidechain/
# (separate directory from the other configurations).

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

CHUNK_SIZE=${CHUNK_SIZE:-5}
N_TASKS=$(( (N + CHUNK_SIZE - 1) / CHUNK_SIZE ))

echo "Submitting side-chain topo-features array job for $N proteins as $N_TASKS tasks (CHUNK_SIZE=$CHUNK_SIZE) ..."

sbatch \
    --array="1-${N_TASKS}" \
    --export=ALL,CHUNK_SIZE=${CHUNK_SIZE} \
    "$REPO/protein_function/scripts/sbatch_topo_features_sidechain.sh"
