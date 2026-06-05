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

echo "Submitting NMA-PCA array job for $N proteins ..."

sbatch \
    --array="1-${N}" \
    "$REPO/protein_function/scripts/sbatch_nma_pca.sh"
