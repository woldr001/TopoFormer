#!/bin/bash
#SBATCH --job-name=sc_topo_gpu
#SBATCH --account=bip294              # OLCF project ID
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#
# GPU-batched side-chain centroid topology features for OLCF Frontier
# (4x MI250X = 8 GCDs per node / ROCm).
#
# Account and paths are set for project bip294; verify the module-load lines
# below match what `module avail` shows on Frontier before submitting.
#
# IMPORTANT: a single GCD uses only 1/8 of a Frontier node, and you are billed
# per whole node, so this script PACKS the node: it splits each array task's
# chunk of proteins into GPUS_PER_NODE shards and runs one process per GCD
# concurrently (each process is single-GPU; --use_gpu forces --n_workers 1).
#
# Read protein_function/README.md "Performance notes: CPU vs GPU" first — for
# this small-eigendecomposition workload the GPU is only ~1.2-2x faster than a
# CPU core, so the CPU multiprocessing path is usually the better tool. Use
# Frontier GPUs only when that is the allocation you have to spend.
#
# Output shape per protein: [12, 200, 15]

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
# Frontier Lustre layout is /lustre/orion/<project>/proj-shared/... — your team's
# shared root is /lustre/orion/bip294/proj-shared.
# NOTE: PDB_DIR must be on Frontier's Lustre — copy the ensembles over with
# Globus first (Frontier cannot read MSU's /mnt/research filesystem).
SHARED=/lustre/orion/bip294/proj-shared
REPO=$SHARED/protein_function/TopoFormer
PDB_DIR=$SHARED/asam_ensembles/protein_function_prediction/v0/sampling
ID_FILE=${ID_FILE:-$REPO/datasets/all_ids.txt}
TOPO_DIR=$SHARED/topo_features_sidechain_gpu

# GCDs to use per node (Frontier has 8). Proteins per array task = CHUNK_SIZE.
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CHUNK_SIZE=${CHUNK_SIZE:-1600}

# ── Environment ───────────────────────────────────────────────────────────────
# Verified working on a frontier compute node (torch 2.6.0+rocm6.1, 8 GCDs).
# The compute-node shell does NOT inherit an interactive `conda activate`, and
# `salloc`/module loads print "Deactivating conda environments" — so the job
# MUST (re)activate the env itself here. PYTHONNOUSERSITE=1 is REQUIRED: a stray
# torch in ~/.local/.../python3.10 (missing libmagma.so) will otherwise shadow
# the env's torch and crash every worker with an ImportError.
ENV_PREFIX=$SHARED/protein_function/envs/topoformer_rocm
module load PrgEnv-gnu rocm miniforge3
# A batch shell doesn't source ~/.bashrc, so the `conda` shell function may be
# undefined — source the hook explicitly before activating.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export PYTHONNOUSERSITE=1

# CPU deps were installed into the env once with:
#   pip install torch --index-url https://download.pytorch.org/whl/rocm6.1
#   pip install -r $REPO/protein_function/requirements_mf.txt

mkdir -p "$TOPO_DIR"

# ── Select this array task's chunk of proteins ────────────────────────────────
TOTAL=$(wc -l < "$ID_FILE")
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
START=$(( (TASK_ID - 1) * CHUNK_SIZE + 1 ))
END=$(( TASK_ID * CHUNK_SIZE ))
END=$(( END > TOTAL ? TOTAL : END ))

echo "Task $TASK_ID | Lines $START–$END of $TOTAL | $(date)"
echo "Node: $(hostname) | packing $GPUS_PER_NODE GCDs"
# Confirm the env's ROCm torch loaded (not a stray ~/.local torch) and 8 GCDs
# are visible — bail early with a clear message if not.
python -c "import torch, sys; print('torch', torch.__version__, '|', torch.__file__); \
    n = torch.cuda.device_count(); print('GPUs visible:', n); \
    sys.exit(0 if n == ${GPUS_PER_NODE} else 1)" \
    || { echo 'ERROR: expected '"$GPUS_PER_NODE"' GCDs / wrong torch — check env activation'; exit 1; }

WORKDIR=$(mktemp -d)
trap "rm -rf $WORKDIR" EXIT
sed -n "${START},${END}p" "$ID_FILE" > "$WORKDIR/chunk_ids.txt"

# Split this chunk into one shard per GCD (split -n l/N divides by line count).
split -n "l/${GPUS_PER_NODE}" -d --additional-suffix=.txt \
    "$WORKDIR/chunk_ids.txt" "$WORKDIR/shard_"

# ── Launch one process per GCD, each pinned to its GCD via ROCR_VISIBLE_DEVICES ─
pids=()
gcd=0
for shard in "$WORKDIR"/shard_*.txt; do
    [ -s "$shard" ] || { gcd=$((gcd + 1)); continue; }   # skip empty shard
    echo "  GCD $gcd → $(wc -l < "$shard") proteins ($shard)"
    ROCR_VISIBLE_DEVICES=$gcd \
    python "$REPO/protein_function/scripts/precompute_topo_features.py" \
        --mode                  sidechain_centroid \
        --use_gpu \
        --device                cuda \
        --gpu_dtype             float64 \
        --pdb_dir               "$PDB_DIR" \
        --output_dir            "$TOPO_DIR" \
        --pdb_list              "$shard" \
        --n_conformers          10 \
        --dis_start             0.0 \
        --dis_cutoff            40.0 \
        --dis_step              0.2 \
        --ensemble_aggregation  mean_std \
        --max_batch_matrices    4096 \
        > "$WORKDIR/gcd_${gcd}.log" 2>&1 &
    pids+=($!)
    gcd=$((gcd + 1))
done

# Wait for all GCD workers; fail the job if any worker failed.
rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done

# Surface each worker's log into the main job output.
for log in "$WORKDIR"/gcd_*.log; do
    echo "──── $log ────"
    cat "$log"
done

echo "Task $TASK_ID done (rc=$rc) | $(date)"
exit $rc

# ──────────────────────────────────────────────────────────────────────────────
# Submit examples (from $REPO):
#
#   # One node, all 8 GCDs, 1600 proteins (200 per GCD) in a single array task:
#   sbatch --array=1-1 protein_function/scripts/sbatch_topo_features_sidechain_gpu_frontier.sh
#
#   # Full dataset across many nodes, 1600 proteins per node-task:
#   N=$(wc -l < /lustre/orion/bip294/proj-shared/TopoFormer/datasets/all_ids.txt)
#   NTASKS=$(( (N + 1599) / 1600 ))
#   sbatch --array=1-${NTASKS} protein_function/scripts/sbatch_topo_features_sidechain_gpu_frontier.sh
# ──────────────────────────────────────────────────────────────────────────────
