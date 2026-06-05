#!/bin/bash
# Submit the NMA-PCA array job with the correct --array size.
# Run from the repo root:
#   bash protein_function/scripts/submit_nma_pca.sh

REPO=/mnt/research/woldring_lab/TopoFormer-MF/TopoFormer
DATASETS=/mnt/research/woldring_lab/TopoFormer-MF/datasets
LOGS=/mnt/research/woldring_lab/TopoFormer-MF/logs

# Ensure all_ids.txt exists (combines train + val + test)
ALL_IDS="$DATASETS/all_ids.txt"
if [[ ! -f "$ALL_IDS" ]]; then
    cat "$DATASETS/train_ids.txt" "$DATASETS/val_ids.txt" "$DATASETS/test_ids.txt" \
        | sort -u > "$ALL_IDS"
    echo "Created $ALL_IDS"
fi

N=$(wc -l < "$ALL_IDS")
mkdir -p "$LOGS"

# Each array task processes CHUNK_SIZE proteins; keep total tasks within QOS limits.
CHUNK_SIZE=5
N_TASKS=$(( (N + CHUNK_SIZE - 1) / CHUNK_SIZE ))

echo "Submitting NMA-PCA array job for $N proteins as $N_TASKS tasks (CHUNK_SIZE=$CHUNK_SIZE) ..."

sbatch \
    --array="1-${N_TASKS}" \
    --export=ALL,CHUNK_SIZE=${CHUNK_SIZE} \
    "$REPO/protein_function/scripts/sbatch_nma_pca.sh"
