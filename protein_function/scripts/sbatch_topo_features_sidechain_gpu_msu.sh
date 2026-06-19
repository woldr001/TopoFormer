#!/bin/bash --login
#SBATCH --job-name=sc_topo_gpu
#SBATCH --time=04:00:00
#SBATCH --gpus=1                 # request 1 GPU; to pin a type use e.g. --gpus=v100:1
#SBATCH --cpus-per-task=4        # a few CPUs for PDB parsing / data loading
#SBATCH --mem=32G
#SBATCH --output=/mnt/research/nodes/Woldring/logs/topo_sc_gpu_%A_%a.out
#SBATCH --error=/mnt/research/nodes/Woldring/logs/topo_sc_gpu_%A_%a.err
# NOTE: --array is set on the command line (see submit example at the bottom).
#
# GPU side-chain centroid topology features on the MSU HPCC (ICER) NVIDIA GPUs.
#
# Unlike the CPU SLURM script (which fans out across CPU cores with a
# multiprocessing Pool), GPU runs are SINGLE-PROCESS: --use_gpu forces
# --n_workers 1 and each array task processes its chunk of proteins
# SEQUENTIALLY on one GPU, because the eigvalsh batching already provides the
# parallelism. Oversubscribing one GPU with multiple processes is slower.
#
# Output shape per protein: [12, 200, 15]
# No NMA-PCA profiles are required for this mode.

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/mnt/research/nodes/Woldring/TopoFormer
PDB_DIR=/mnt/research/nodes/giacomo/asam_ensembles/protein_function_prediction/v0/sampling
TOPO_DIR=/mnt/research/nodes/Woldring/topo_features_sidechain_gpu
LOG_DIR=/mnt/research/nodes/Woldring/logs

# Protein ID list (one ID per line). Default: your 20-protein test set.
# Point this at datasets/all_ids.txt to run the full dataset.
ID_FILE=${ID_FILE:-/mnt/research/nodes/Woldring/protein_ids_20.txt}

# Proteins per array task (processed sequentially on the GPU).
CHUNK_SIZE=${CHUNK_SIZE:-20}

# ── Environment ───────────────────────────────────────────────────────────────
# The system Miniforge3 module breaks `conda activate`, so prepend the env's bin.
export PATH="/mnt/home/woldring/.conda/envs/topoformer_mf/bin:$PATH"

# CUDA PyTorch must be installed in this env (once):
#   pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# (match the CUDA version of the GPU node; cu124 verified on the amd24-h200 nodes)

mkdir -p "$TOPO_DIR" "$LOG_DIR"

# ── Sanity: confirm the GPU is visible to torch ───────────────────────────────
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU visible'; \
    print('GPU:', torch.cuda.get_device_name(0))"

# ── Select this task's chunk of proteins ──────────────────────────────────────
TOTAL=$(wc -l < "$ID_FILE")
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
START=$(( (TASK_ID - 1) * CHUNK_SIZE + 1 ))
END=$(( TASK_ID * CHUNK_SIZE ))
END=$(( END > TOTAL ? TOTAL : END ))

echo "Task $TASK_ID | Lines $START–$END of $TOTAL | $(date)"
echo "Node: $(hostname) | GPU job"

TMP_IDS=$(mktemp /tmp/topo_sc_gpu_ids_${TASK_ID}_XXXX.txt)
trap "rm -f $TMP_IDS" EXIT
sed -n "${START},${END}p" "$ID_FILE" > "$TMP_IDS"

echo "Proteins in this chunk:"
cat "$TMP_IDS"
echo ""

# ── Run (single GPU process; --n_workers forced to 1 by --use_gpu) ────────────
python "$REPO/protein_function/scripts/precompute_topo_features.py" \
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

echo "Task $TASK_ID done | $(date)"

# ──────────────────────────────────────────────────────────────────────────────
# Submit examples:
#
#   # 20-protein test set in a single GPU task (CHUNK_SIZE=20 covers all 20):
#   sbatch --array=1-1 protein_function/scripts/sbatch_topo_features_sidechain_gpu_msu.sh
#
#   # Full dataset, 200 proteins per GPU task:
#   N=$(wc -l < /mnt/research/nodes/Woldring/TopoFormer/datasets/all_ids.txt)
#   NTASKS=$(( (N + 199) / 200 ))
#   sbatch --array=1-${NTASKS} \
#       --export=ALL,ID_FILE=/mnt/research/nodes/Woldring/TopoFormer/datasets/all_ids.txt,CHUNK_SIZE=200 \
#       protein_function/scripts/sbatch_topo_features_sidechain_gpu_msu.sh
# ──────────────────────────────────────────────────────────────────────────────
