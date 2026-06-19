#!/bin/bash
#SBATCH --job-name=sc_topo_gpu
#SBATCH --account=YOUR_PROJECT        # <-- set your OLCF project allocation
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus=1                      # one MI250X GCD is plenty per task
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#
# GPU-batched side-chain centroid topology features for OLCF Frontier (MI250X / ROCm).
#
# This is a TEMPLATE — adjust the account, paths, and module names to your
# Frontier environment. Unlike the CPU SLURM scripts, GPU runs are
# single-process per task (the eigvalsh batching already provides the
# parallelism), so each array task processes a chunk of proteins SEQUENTIALLY
# on one GPU rather than fanning out across CPU cores.
#
# Output shape per protein: [12, 200, 15]

set -euo pipefail

# ── Paths (EDIT THESE) ────────────────────────────────────────────────────────
REPO=/lustre/orion/proj-shared/YOUR_PROJECT/TopoFormer
PDB_DIR=/lustre/orion/proj-shared/YOUR_PROJECT/asam_ensembles/v0/sampling
ID_FILE=$REPO/datasets/all_ids.txt
TOPO_DIR=/lustre/orion/proj-shared/YOUR_PROJECT/topo_features_sidechain_gpu

# Proteins per array task (processed sequentially on the GPU).
CHUNK_SIZE=${CHUNK_SIZE:-200}

# ── Environment (EDIT module names to match Frontier) ─────────────────────────
# Frontier provides PyTorch via modules or a conda env with a ROCm build.
# Example (adjust to your setup):
#   module load PrgEnv-gnu rocm
#   source /lustre/orion/proj-shared/YOUR_PROJECT/envs/topoformer_rocm/bin/activate
# Verify torch sees the GPU:
#   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

mkdir -p "$TOPO_DIR"

# ── Select this task's chunk of proteins ──────────────────────────────────────
TOTAL=$(wc -l < "$ID_FILE")
START=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK_SIZE + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( END > TOTAL ? TOTAL : END ))

echo "Task $SLURM_ARRAY_TASK_ID | Lines $START–$END of $TOTAL | $(date)"
echo "Node: $(hostname)"
python -c "import torch; print('GPU:', torch.cuda.is_available(), \
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

TMP_IDS=$(mktemp)
trap "rm -f $TMP_IDS" EXIT
sed -n "${START},${END}p" "$ID_FILE" > "$TMP_IDS"

echo "Proteins in this chunk:"
cat "$TMP_IDS"
echo ""

# ── Run (single GPU process; --n_workers is forced to 1 by --use_gpu) ─────────
srun -n1 python "$REPO/protein_function/scripts/precompute_topo_features.py" \
    --mode                  sidechain_centroid \
    --use_gpu \
    --device                cuda \
    --gpu_dtype             float64 \
    --pdb_dir               "$PDB_DIR" \
    --output_dir            "$TOPO_DIR" \
    --pdb_list              "$TMP_IDS" \
    --n_conformers          10 \
    --dis_start             0.0 \
    --dis_cutoff            40.0 \
    --dis_step              0.2 \
    --ensemble_aggregation  mean_std \
    --max_batch_matrices    4096

echo "Task $SLURM_ARRAY_TASK_ID done | $(date)"
