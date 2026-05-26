#!/bin/bash --login
#SBATCH --job-name=filter_pdb_500aa
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/mnt/scratch/%u/topoformer_mf/logs/filter_pdb_%j.out
#SBATCH --error=/mnt/scratch/%u/topoformer_mf/logs/filter_pdb_%j.err

set -euo pipefail

module purge
module load Miniforge3
conda activate topoformer_mf

REPO=/mnt/research/woldring_lab/TopoFormer-MF/TopoFormer
SOURCE=/mnt/research/nodes/trainingset/afdatabase
DEST=/mnt/research/nodes/trainingset/afdatabase/sub_500aa

mkdir -p /mnt/scratch/$USER/topoformer_mf/logs
mkdir -p "$DEST"

python "$REPO/protein_function/scripts/filter_pdb_by_length.py" \
    --source_dir "$SOURCE" \
    --dest_dir   "$DEST" \
    --max_aa     500 \
    --n_workers  16

echo "Done. Copied files:"
ls "$DEST"/*.pdb 2>/dev/null | wc -l
