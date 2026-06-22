# CPU (MSU HPCC) vs GPU (OLCF Frontier): Side-Chain Topology Embeddings

A guide to the changes that let us compute the **side-chain centroid topology
embeddings** for protein ensembles on **Frontier's AMD GPUs**, alongside the
original **CPU pipeline on the MSU HPCC**. Written so a newcomer can follow the
"why," with enough specifics that an expert can audit the "how."

---

## 1. The one-paragraph summary

The feature we compute per protein is a `[12, 200, 15]` array of numbers that
summarize the *shape* (topology) of each protein's side chains across 10
conformations. The math is the same on both machines. The **only** thing that
changed for Frontier is **how the heavy linear-algebra step is executed**: the
CPU version does it as ~30,000 tiny calculations one after another; the GPU
version stacks those same calculations into a few large batched operations that
an AMD GPU can do in parallel. **The numbers that come out are identical** (to 5
decimal places). We did *not* change the science — we changed the plumbing.

---

## 2. What stays the same vs what changes

| Aspect | CPU (HPCC) | GPU (Frontier) |
|---|---|---|
| The feature definition (what is computed) | ✅ identical | ✅ identical |
| Output shape per protein | `[12, 200, 15]` float32 `.npy` | `[12, 200, 15]` float32 `.npy` |
| PDB parsing / side-chain centroid logic | shared code | **same shared code** (imported, not re-written) |
| The heavy step: eigenvalues of small matrices | NumPy, one matrix at a time | PyTorch, **batched** many matrices at once |
| Hardware | many CPU cores (multiprocessing) | AMD MI250X GPUs (8 GCDs/node) |
| Numerical result | reference | **matches CPU to ~1e-5** |

The key idea: for each protein we must compute the eigenvalues of
`200 filtration steps × 15 chemical-class combinations × 10 conformations ≈
30,000` small symmetric matrices. On CPU these are a sequential loop. On GPU we
**batch** them — that batching, not the GPU by itself, is where any speedup
comes from.

---

## 3. Files added or modified

Everything below is on the **`protein-motion-topology`** branch.

### New files (the GPU path)

| File | Purpose |
|---|---|
| `protein_function/topo_extraction/sidechain_topo_embedding_gpu.py` | The GPU re-implementation. Batches all thresholds/conformations into `torch.linalg.eigvalsh` calls. Produces output identical to the CPU module. |
| `protein_function/scripts/sbatch_topo_features_sidechain_gpu_frontier.sh` | SLURM job script for Frontier. Packs all **8 GCDs** of a node, one process per GCD. |
| `protein_function/scripts/sbatch_topo_features_sidechain_gpu_msu.sh` | SLURM job script for MSU's NVIDIA GPUs (single GPU per task). |
| `protein_function/scripts/verify_gpu_parity.py` | Runs CPU and GPU on the same protein and reports the max numerical difference + speedup. This is the correctness check. |
| `protein_function/scripts/profile_sidechain_bottleneck.py` | Per-combination profiler used to investigate why the GPU speedup is modest. |
| `protein_function/requirements_mf.txt` | Pinned, mutually-compatible CPU dependency set (numpy/scipy/prody/matplotlib). torch is installed separately per accelerator. |

### Modified files

| File | What changed |
|---|---|
| `protein_function/scripts/precompute_topo_features.py` | The shared batch entry point gained `--use_gpu`, `--device`, `--gpu_dtype`, `--max_batch_matrices`. When `--use_gpu` is set it dispatches to the GPU module; otherwise it runs the CPU path exactly as before. |
| `protein_function/README.md` | Added a "GPU acceleration" section and a "Performance notes: CPU vs GPU" section documenting the measured speedups. |

### Unchanged (deliberately reused, not copied)

| File | Why it matters |
|---|---|
| `protein_function/topo_extraction/sidechain_topo_embedding.py` (CPU module) | The GPU module **imports** its PDB parsing, chemical-class definitions, and the 15 combinations from here, so the two implementations cannot silently drift apart. |
| `code_pkg/top_embedding/SimplicialComplex_laplacian.py` | The original CPU eigenvalue/statistics reference. Never modified — it is the ground truth the GPU output is validated against. |

---

## 4. Libraries and modules

| | CPU (HPCC) | GPU (Frontier) |
|---|---|---|
| Linear algebra | **NumPy** `np.linalg.eigvalsh` | **PyTorch** `torch.linalg.eigvalsh` (ROCm build) |
| Distances | SciPy `cdist` | `torch.cdist` |
| Parallelism | Python `multiprocessing.Pool` across CPU cores | GPU tensor batching (+ one OS process per GCD) |
| PDB parsing | ProDy (via the shared CPU module) | **same** (parsing runs on CPU, then coordinates are moved to the GPU) |
| Accelerator install | n/a | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.1` |

On MSU's NVIDIA GPUs the same GPU code runs with the CUDA wheel
(`--index-url .../cu124`) instead of the ROCm wheel — the Python code is
accelerator-agnostic (`device='cuda'` works for both NVIDIA and AMD/ROCm).

---

## 5. The core code change, explained

### CPU reference (sequential)
In `sidechain_topo_embedding.py`, for every filtration threshold the code builds
one Laplacian matrix and calls NumPy on it, one at a time:

```python
lap_features = scl.persistent_simplicialComplex_laplacian_dim0(...)  # loops internally,
                                                                     # one np.linalg.eigvalsh per step
```

### GPU version (batched)
In `sidechain_topo_embedding_gpu.py`, all thresholds for a combination are
stacked into a single 3-D tensor and solved in one call (chunked for memory):

```python
adj = dist.unsqueeze(1) <= thr.view(1, -1, 1, 1)   # [g, tc, N, N] adjacency for many thresholds at once
adj = adj & ~diag_mask                              # zero the diagonal
lap = torch.diag_embed(adj_f.sum(-1)) - adj_f       # L = diag(rowsum(A)) - A, batched
eig = torch.linalg.eigvalsh(lap_flat)               # eigenvalues of the WHOLE batch in one kernel
stats = _stats_from_eigenvalues(eig)                # the 6 spectral statistics, vectorized
```

That is the entire conceptual change: **a Python loop of tiny NumPy calls becomes
a handful of large batched PyTorch calls.** Everything around it (which residues,
which class combos, how the 6 statistics are defined, how conformations are
averaged) is reproduced exactly.

### How identical output is guaranteed
The GPU module preserves the CPU semantics line-for-line, including the subtle
parts that are easy to get wrong:

- **Adjacency rule:** `A = (distance ≤ threshold)` with the diagonal forced to 0.
- **Rounding:** eigenvalues are rounded to **5 decimals** before any statistic
  (`_round5`), matching `np.round(x, 5)`.
- **The 6 statistics, in order:** `[count_zero, max, sum, nonzero_mean,
  nonzero_std, nonzero_min]`, with `std` as population std (ddof=0) over the
  *non-zero-after-rounding* eigenvalues.
- **Empty combos:** combinations with fewer than 2 selected residues are left as
  zeros (same as CPU).
- **Float32-before-averaging:** the CPU stores each conformation's stats as
  float32 *before* averaging across the 10 conformations. The GPU casts to
  float32 at the same point (`result...astype(np.float32)` before
  `mean`/`std`) — without this, float64 accumulation would diverge from CPU at
  ~1e-3 on the large-magnitude `sum` statistic. This is the one non-obvious
  parity detail and it is commented in the code.

`verify_gpu_parity.py` checks all of this empirically: it runs both paths on the
same protein and asserts the max absolute difference is below tolerance
(default `1e-2`); in practice it is ~`1e-5`.

---

## 6. The `precompute_topo_features.py` integration

This is the single batch entry point both machines call. The change is additive
and backward-compatible:

- **Without `--use_gpu`:** behaves exactly as the original CPU pipeline —
  `multiprocessing.Pool` across `--n_workers` CPU cores. torch is never even
  imported.
- **With `--use_gpu`:** dispatches each protein to
  `generate_sidechain_lap_features_gpu`. torch is imported lazily (only when
  needed), so CPU-only environments don't need it installed.

Two guard rails were added:
- `--use_gpu` is only valid for `--mode sidechain_centroid` (the only mode with
  a GPU implementation); otherwise it errors out clearly.
- If `--use_gpu` is combined with `--n_workers > 1`, it **forces
  `--n_workers 1`** with a warning — the GPU path already parallelizes via
  batching, and running multiple processes against one GPU would oversubscribe
  it.

New flags: `--device` (`cuda`/`cpu`), `--gpu_dtype` (`float64` default to match
CPU, or `float32` for speed), `--max_batch_matrices` (lower it if a large
protein hits out-of-memory).

---

## 7. The SLURM / job-script differences

| | CPU script (HPCC) | GPU script (Frontier) |
|---|---|---|
| Resource request | many CPU cores | a whole node (8 GCDs) |
| Work distribution | one `Pool` over cores | the node's protein chunk is **split into 8 shards**, one per GCD |
| GPU pinning | n/a | each shard's process pinned with `ROCR_VISIBLE_DEVICES=$gcd` (the ROCm equivalent of `CUDA_VISIBLE_DEVICES`) |
| Environment | conda env on HPCC | `module load PrgEnv-gnu rocm miniforge3`, then activate the ROCm env |
| Billing-driven design | per-core | Frontier bills **per whole node**, so a single-GCD job would waste 7/8 of the allocation — the script deliberately packs all 8 GCDs |

Two Frontier-specific gotchas that are now baked into the script (each cost real
debugging time):

1. **The compute-node shell does not inherit your interactive conda
   activation.** `salloc`/module loads print "Deactivating conda environments,"
   so the job script must re-run `module load ... && conda activate ...` itself.
2. **`PYTHONNOUSERSITE=1` is required.** A stray torch in
   `~/.local/.../python3.10` (missing `libmagma.so`) will otherwise shadow the
   correct env torch and crash every worker with an `ImportError`. The script
   sets this and includes a startup check that confirms the *right* torch loaded
   and that all 8 GCDs are visible before doing any work.

The data must also live on Frontier's Lustre filesystem (`/lustre/orion/...`);
Frontier cannot read MSU's `/mnt/research`, so the ensembles are transferred via
Globus first.

---

## 8. Performance — the honest result

We measured this carefully (full write-up in `README.md` → "Performance notes").
The short version:

| Hardware | CPU (full protein) | GPU (full protein) | Speedup |
|---|---|---|---|
| V100 node (older CPU) | 129.2 s | 59.7 s | 2.2× |
| H200 node | 68.0 s | 52.6 s | **1.24×** |

**Why so modest?** The matrices are small (N ≤ ~400). Small combos are actually
*slower* on GPU (kernel-launch overhead dominates); only the largest combos win
~2×; the net is ~1.24× over a **single** CPU core. Because a CPU node has many
cores running in parallel via multiprocessing, the CPU path usually wins on
total throughput and on allocation cost.

**Practical guidance for the team:** prefer the **CPU multiprocessing path on
HPCC** for bulk production. Use the **Frontier GPU path** when burning a GPU
allocation is the goal, or for very large proteins where a single GPU's batched
solve beats a single CPU core. Two hypotheses for a *larger* hidden GPU win were
tested and **disproven** (CPU adjacency-matrix caching: only ~8% of steps
skipped; BLAS multithreading flattering the CPU: negligible effect) — documented
so nobody re-investigates them.

---

## 9. How to run each path (quick reference)

**CPU (MSU HPCC), bulk, multi-core:**
```bash
python protein_function/scripts/precompute_topo_features.py \
    --mode sidechain_centroid \
    --pdb_dir <ensembles> --output_dir <out> \
    --pdb_list <ids.txt> --n_conformers 10 \
    --n_workers 8
```

**GPU (single protein, either accelerator):**
```bash
python protein_function/scripts/precompute_topo_features.py \
    --mode sidechain_centroid --use_gpu --device cuda --gpu_dtype float64 \
    --pdb_dir <ensembles> --output_dir <out> --pdb_list <ids.txt> --n_conformers 10
```

**GPU (Frontier, full node, all 8 GCDs):**
```bash
sbatch --array=1-1 protein_function/scripts/sbatch_topo_features_sidechain_gpu_frontier.sh
```

**Verify CPU and GPU agree:**
```bash
python protein_function/scripts/verify_gpu_parity.py \
    --protein_id <id> --pdb_dir <ensembles>
```

---

*Branch: `protein-motion-topology`. The CPU and GPU modules share all parsing
and feature-definition code by import, so any future change to the feature
definition automatically applies to both.*
