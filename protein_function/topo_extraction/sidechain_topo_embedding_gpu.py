"""GPU-batched side-chain centroid ensemble topology feature extraction.

This is a tensor-batched re-implementation of
``sidechain_topo_embedding.py`` (the CPU reference) designed to run on GPUs
such as the AMD MI250X cards on OLCF Frontier (PyTorch ROCm build).  It
produces **bit-for-bit comparable** output to the CPU module — the same
``[12, 200, 15]`` float32 tensor per protein — up to floating-point rounding
at the 5th decimal place.

Why this is faster
------------------
The CPU reference computes, per protein,
``200 filtration steps x 15 combos x 10 conformations`` = 30,000 *sequential*
eigendecompositions of small symmetric Laplacians (``np.linalg.eigvalsh``).
That is an embarrassingly parallel batch of tiny linear-algebra problems —
exactly what GPUs accelerate when the work is **batched** rather than looped.

This module restructures the computation so that, for each of the 15 class
combinations, all filtration thresholds and all conformations are stacked into
a single batched ``torch.linalg.eigvalsh`` call (chunked to bound memory).
That collapses 30,000 sequential CPU calls into ~15 batched GPU kernels.

Critically, the speedup comes from **batching**, not from the GPU alone: a
naive per-step ``.eigvalsh()`` on tiny matrices would be slower on GPU than on
CPU because of kernel-launch overhead.  Always keep matrices batched.

Semantics preserved from the CPU reference
------------------------------------------
* Adjacency at threshold ``t``: ``A = (D <= t)`` with the diagonal forced to 0.
* dim-0 graph Laplacian: ``L = diag(rowsum(A)) - A``.
* Eigenvalues rounded to 5 decimals before any statistic is computed.
* The 6 spectral statistics, in order:
  ``[count_zero, max, sum, nonzero_mean, nonzero_std, nonzero_min]``
  where ``nonzero_*`` are over eigenvalues that are non-zero *after* rounding,
  and ``std`` is the population std (ddof=0, matching ``np.std``).
* Combos with fewer than 2 selected residues are left as zeros (matches CPU).
* Ensemble aggregation (mean + std across conformations) is done in NumPy to
  match the CPU reference exactly.

Precision
---------
Defaults to ``float64`` to match the CPU reference (which runs in double
precision throughout).  MI250X has strong FP64 throughput, so this is usually
the right default.  Pass ``--dtype float32`` for extra speed if you have
validated that the reduced precision does not change your downstream model.

Usage (CLI)
-----------
    python sidechain_topo_embedding_gpu.py \\
        --protein_id    P12345 \\
        --pdb_dir       /data/ensembles \\
        --output_folder /data/topo_sidechain \\
        --device        cuda

Usage (API)
-----------
    from protein_function.topo_extraction.sidechain_topo_embedding_gpu import (
        generate_sidechain_lap_features_gpu,
    )
    feat = generate_sidechain_lap_features_gpu(
        output_folder="./features",
        protein_id="P12345",
        pdb_files=["conf_0.pdb", ..., "conf_9.pdb"],
        device="cuda",
    )  # shape [12, 200, 15]
"""

import argparse
import os
import sys
import time
from typing import List, Optional

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch is a hard dependency here
    raise ImportError(
        "sidechain_topo_embedding_gpu requires PyTorch. On OLCF Frontier install "
        "a ROCm build, e.g.:\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/rocm6.1"
    ) from exc

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the parsing, class definitions, and combination list from the CPU module
# so the two implementations can never drift apart.
from protein_function.topo_extraction.sidechain_topo_embedding import (
    NUM_COMBINATIONS,
    NUM_STATISTICS,
    SIDECHAIN_COMBINATIONS,
    parse_residue_sidechain_centroids,
)
from protein_function.topo_extraction.ensemble_motion_topo_embedding import (
    find_ensemble_pdbs,
)

_ROUND_DECIMALS = 5
_ROUND_SCALE = 10 ** _ROUND_DECIMALS

# Peak memory holds several [M, N, N] copies at once (bool adjacency, float
# adjacency, Laplacian, plus eigvalsh internal workspace). Budget conservatively.
_MEM_OVERHEAD = 6
_DEFAULT_MEM_BUDGET_BYTES = 8 * (1024 ** 3)  # 8 GiB working set per chunk


# ---------------------------------------------------------------------------
# Core batched primitives
# ---------------------------------------------------------------------------

def _round5(x: torch.Tensor) -> torch.Tensor:
    """Round to 5 decimals, matching ``np.round(x, 5)`` (round-half-to-even)."""
    return torch.round(x * _ROUND_SCALE) / _ROUND_SCALE


def _stats_from_eigenvalues(eig: torch.Tensor) -> torch.Tensor:
    """Compute the 6 spectral statistics from a batch of eigenvalue vectors.

    Replicates ``code_pkg...statistic_eigvalues`` exactly (with eigenvalues
    already rounded to 5 decimals), vectorised over the leading batch dim.

    Args:
        eig: [M, N] tensor of eigenvalues (ascending; from eigvalsh).

    Returns:
        [M, 6] tensor: [count_zero, max, sum, nonzero_mean, nonzero_std, nonzero_min].
    """
    vals = _round5(eig)                                  # [M, N]
    zero_mask = vals == 0.0                              # exact zeros after rounding
    nonzero_mask = ~zero_mask
    zeros = torch.zeros(vals.shape[0], dtype=vals.dtype, device=vals.device)

    count_zero = zero_mask.sum(dim=-1).to(vals.dtype)    # [M]
    mx = vals.max(dim=-1).values                         # [M]
    sm = _round5(vals.sum(dim=-1))                       # [M]

    nz_count = nonzero_mask.sum(dim=-1)                  # [M] int
    has_nz = nz_count > 0
    nz_count_safe = nz_count.clamp(min=1).to(vals.dtype)

    nz_vals = torch.where(nonzero_mask, vals, torch.zeros_like(vals))
    nz_sum = nz_vals.sum(dim=-1)
    nz_mean = nz_sum / nz_count_safe

    diff = torch.where(nonzero_mask, vals - nz_mean.unsqueeze(-1),
                       torch.zeros_like(vals))
    nz_var = (diff * diff).sum(dim=-1) / nz_count_safe   # population variance (ddof=0)
    nz_std = torch.sqrt(nz_var.clamp(min=0))

    vals_for_min = vals.masked_fill(zero_mask, float("inf"))
    nz_min = vals_for_min.min(dim=-1).values

    nz_mean = torch.where(has_nz, _round5(nz_mean), zeros)
    nz_std = torch.where(has_nz, _round5(nz_std), zeros)
    nz_min = torch.where(has_nz, _round5(nz_min), zeros)

    return torch.stack([count_zero, mx, sm, nz_mean, nz_std, nz_min], dim=-1)  # [M, 6]


def _batched_eigvalsh(mats: torch.Tensor, max_batch: int) -> torch.Tensor:
    """``torch.linalg.eigvalsh`` over a [M, N, N] stack, chunked to bound memory.

    Args:
        mats: [M, N, N] symmetric matrices.
        max_batch: maximum number of matrices per eigvalsh call.

    Returns:
        [M, N] eigenvalues (ascending per matrix).
    """
    m = mats.shape[0]
    if m <= max_batch:
        return torch.linalg.eigvalsh(mats)
    out = []
    for start in range(0, m, max_batch):
        out.append(torch.linalg.eigvalsh(mats[start:start + max_batch]))
    return torch.cat(out, dim=0)


def _combo_stats_over_thresholds(
    dist: torch.Tensor,
    thresholds: torch.Tensor,
    eye: torch.Tensor,
    max_batch: int,
    mem_budget_bytes: int = _DEFAULT_MEM_BUDGET_BYTES,
) -> torch.Tensor:
    """Compute [g, T, 6] stats for a group of g distance matrices (same N).

    For each threshold the dim-0 Laplacian is built and its eigenvalue spectrum
    reduced to the 6 statistics.  Thresholds are processed in chunks so the
    intermediate ``[g, tc, N, N]`` adjacency tensor stays within memory budget.

    Args:
        dist: [g, N, N] distance matrices (all same N >= 2).
        thresholds: [T] filtration thresholds.
        eye: [N, N] boolean identity for masking the diagonal.
        max_batch: target number of matrices per eigvalsh call.
        mem_budget_bytes: working-set cap used to shrink the batch for large N.

    Returns:
        [g, T, 6] statistics tensor.
    """
    g, n, _ = dist.shape
    t = thresholds.shape[0]
    dtype = dist.dtype
    device = dist.device

    out = torch.zeros(g, t, NUM_STATISTICS, dtype=dtype, device=device)

    # Cap the per-chunk matrix count two ways:
    #   1. the user's max_batch (eigvalsh call size), and
    #   2. an N-aware memory bound so a huge combo (large N) auto-shrinks the
    #      batch instead of OOM-ing.
    bytes_per = torch.finfo(dtype).bits // 8
    max_by_mem = max(1, mem_budget_bytes // (n * n * bytes_per * _MEM_OVERHEAD))
    m_max = max(1, min(max_batch, int(max_by_mem)))

    # Choose how many thresholds per chunk so that g * tc <= m_max.
    tc = max(1, m_max // max(1, g))

    diag_mask = eye.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]

    for t0 in range(0, t, tc):
        thr = thresholds[t0:t0 + tc]                      # [tc]
        # Adjacency: (D <= t) with diagonal removed.
        adj = dist.unsqueeze(1) <= thr.view(1, -1, 1, 1)  # [g, tc, N, N] bool
        adj = adj & ~diag_mask
        adj_f = adj.to(dtype)
        deg = adj_f.sum(dim=-1)                            # [g, tc, N]
        lap = torch.diag_embed(deg) - adj_f                # [g, tc, N, N]

        tc_actual = thr.shape[0]
        lap_flat = lap.reshape(g * tc_actual, n, n)
        eig = _batched_eigvalsh(lap_flat, max_batch)       # [g*tc, N]
        stats = _stats_from_eigenvalues(eig)               # [g*tc, 6]
        out[:, t0:t0 + tc_actual, :] = stats.reshape(g, tc_actual, NUM_STATISTICS)

    return out


# ---------------------------------------------------------------------------
# Main ensemble feature extraction (GPU)
# ---------------------------------------------------------------------------

def generate_sidechain_lap_features_gpu(
    output_folder: str,
    protein_id: str,
    pdb_files: List[str],
    output_feature_name: Optional[str] = None,
    dis_start: float = 0.0,
    dis_cutoff: float = 40.0,
    dis_step: float = 0.2,
    ensemble_aggregation: str = "mean_std",
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float64,
    max_batch_matrices: int = 4096,
    save: bool = True,
    print_progress: bool = True,
) -> np.ndarray:
    """GPU-batched side-chain-centroid ensemble topology features for a protein.

    Mirrors ``generate_sidechain_lap_features`` (CPU) but batches all filtration
    thresholds and conformations per combination into ``torch.linalg.eigvalsh``
    calls.  Output and aggregation semantics are identical.

    Args:
        output_folder: Directory to save the ``.npy`` output.
        protein_id: Identifier used for messages and the default file name.
        pdb_files: List of PDB conformation file paths (the ensemble).
        output_feature_name: Output base name. Defaults to ``protein_id``.
        dis_start/dis_cutoff/dis_step: Filtration range in Å (default 0–40 by 0.2 → 200 steps).
        ensemble_aggregation: ``'mean_std'`` (12 ch), ``'mean_only'`` (6 ch), or ``'all'``.
        device: Torch device string ('cuda' for NVIDIA/ROCm GPUs, 'cpu'). Auto if None.
        dtype: ``torch.float64`` (default, matches CPU) or ``torch.float32``.
        max_batch_matrices: Target number of NxN matrices per eigvalsh call
            (controls memory; lower it if you hit out-of-memory on large proteins).
        save: Whether to write the ``.npy`` file.
        print_progress: Whether to print progress.

    Returns:
        Feature array of shape [C, n_filtrations, 15] float32 (C=12 for mean_std).
    """
    if not pdb_files:
        raise ValueError("pdb_files must be a non-empty list.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    # 1. Filtration parameters (identical to the CPU reference)
    filtration = np.round(np.arange(dis_start, dis_cutoff, dis_step), 6)
    n_filtrations = len(filtration)
    thresholds = torch.as_tensor(filtration, dtype=dtype, device=torch_device)

    # 2. Parse every conformation once (CPU-side PDB parsing, then move coords to GPU)
    parsed = []  # list of (classes np[int], centroids torch[N,3])
    for pdb_file in pdb_files:
        classes, centroids = parse_residue_sidechain_centroids(pdb_file)
        coords_t = (torch.as_tensor(centroids, dtype=dtype, device=torch_device)
                    if len(classes) else
                    torch.empty(0, 3, dtype=dtype, device=torch_device))
        parsed.append((classes, coords_t))

    n_conf = len(parsed)

    # Per-conformation feature buffer: [n_conf, 6, T, 15]
    result = torch.zeros(n_conf, NUM_STATISTICS, n_filtrations, NUM_COMBINATIONS,
                         dtype=dtype, device=torch_device)

    # 3. For each combo, batch all conformations (grouped by N) over all thresholds.
    for combo_idx, class_combo in enumerate(SIDECHAIN_COMBINATIONS):
        if print_progress:
            print(f"[topo-gpu] combo {combo_idx + 1}/{NUM_COMBINATIONS} "
                  f"classes={class_combo}")
        combo_set = set(class_combo)

        # Select each conformation's residues for this combo.
        sel_coords = []   # per conf: [N_c, 3] (may be empty)
        for classes, coords_t in parsed:
            if len(classes) == 0:
                sel_coords.append(coords_t[:0])
                continue
            mask = np.isin(classes, list(combo_set))
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=torch_device)
            sel_coords.append(coords_t[mask_t])

        # Group conformations by their selected residue count N (>= 2).
        groups: dict = {}
        for conf_idx, coords in enumerate(sel_coords):
            n = coords.shape[0]
            if n >= 2:
                groups.setdefault(n, []).append(conf_idx)
            # n < 2 -> leave zeros (matches CPU)

        for n, conf_idxs in groups.items():
            coords_stack = torch.stack([sel_coords[i] for i in conf_idxs], dim=0)  # [g, N, 3]
            dist = torch.cdist(coords_stack, coords_stack)                          # [g, N, N]
            eye = torch.eye(n, dtype=torch.bool, device=torch_device)
            stats = _combo_stats_over_thresholds(
                dist, thresholds, eye, max_batch_matrices,
            )  # [g, T, 6]
            # scatter into result[conf, :, :, combo] -> need [g, 6, T]
            stats_ctf = stats.permute(0, 2, 1)  # [g, 6, T]
            idx_t = torch.as_tensor(conf_idxs, dtype=torch.long, device=torch_device)
            result[idx_t, :, :, combo_idx] = stats_ctf

    # 4. Ensemble aggregation in NumPy (exactly matches the CPU reference).
    #    The CPU reference stores each conformation's stats as float32 *before*
    #    aggregating, so cast to float32 here too — otherwise mean/std would
    #    accumulate in float64 and diverge from the CPU output at ~1e-3 for the
    #    large-magnitude `sum` statistic.
    conf_stack = result.detach().cpu().numpy().astype(np.float32)  # [n_conf, 6, T, 15]

    if ensemble_aggregation == "mean_std":
        mean_feat = conf_stack.mean(axis=0)
        std_feat = conf_stack.std(axis=0)                    # ddof=0
        output = np.concatenate([mean_feat, std_feat], axis=0)   # [12, T, 15]
    elif ensemble_aggregation == "mean_only":
        output = conf_stack.mean(axis=0)                     # [6, T, 15]
    elif ensemble_aggregation == "all":
        output = conf_stack.reshape(
            n_conf * NUM_STATISTICS, n_filtrations, NUM_COMBINATIONS)
    else:
        raise ValueError(f"Unknown ensemble_aggregation: {ensemble_aggregation!r}. "
                         "Choose 'mean_std', 'mean_only', or 'all'.")

    output = output.astype(np.float32)

    if save:
        fname = output_feature_name if output_feature_name else protein_id
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, f"{fname}.npy")
        np.save(out_path, output, allow_pickle=True)
        if print_progress:
            print(f"[topo-gpu] Saved {output.shape} → {out_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DTYPE_MAP = {"float64": torch.float64, "float32": torch.float32}


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "GPU-batched side-chain-centroid Persistent Laplacian features.\n"
            "Output: [12, 200, 15] float32 numpy array (mean_std mode)."
        )
    )
    parser.add_argument("--protein_id",    required=True)
    parser.add_argument("--pdb_dir",       required=True,
                        help="Root directory containing ensemble PDB files.")
    parser.add_argument("--output_folder", default="./topo_sidechain_features_gpu")
    parser.add_argument("--n_conformers",  type=int, default=10)
    parser.add_argument("--dis_start",     type=float, default=0.0)
    parser.add_argument("--dis_cutoff",    type=float, default=40.0)
    parser.add_argument("--dis_step",      type=float, default=0.2)
    parser.add_argument("--ensemble_aggregation", default="mean_std",
                        choices=["mean_std", "mean_only", "all"])
    parser.add_argument("--device", default=None,
                        help="Torch device ('cuda' for NVIDIA/ROCm GPUs, 'cpu'). "
                             "Auto-detected if omitted.")
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"],
                        help="float64 (default, matches CPU) or float32 (faster).")
    parser.add_argument("--max_batch_matrices", type=int, default=4096,
                        help="Matrices per eigvalsh call; lower if you hit OOM.")
    return parser.parse_args(argv)


def main():
    args = _parse_args(sys.argv[1:])
    t0 = time.time()

    pdb_files = find_ensemble_pdbs(
        pdb_dir=args.pdb_dir,
        protein_id=args.protein_id,
        n_conformers=args.n_conformers,
    )
    if not pdb_files:
        print(f"[ERROR] No PDB files found for {args.protein_id} in {args.pdb_dir}")
        sys.exit(1)

    generate_sidechain_lap_features_gpu(
        output_folder=args.output_folder,
        protein_id=args.protein_id,
        pdb_files=pdb_files,
        dis_start=args.dis_start,
        dis_cutoff=args.dis_cutoff,
        dis_step=args.dis_step,
        ensemble_aggregation=args.ensemble_aggregation,
        device=args.device,
        dtype=_DTYPE_MAP[args.dtype],
        max_batch_matrices=args.max_batch_matrices,
        print_progress=True,
    )
    print(f"[topo-gpu] Finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
