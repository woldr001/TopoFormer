#!/usr/bin/env python3
"""Per-combo profiling to test the hypothesis that the CPU reference's
adjacency-matrix caching (it skips re-computing eigvalsh when the threshold
graph hasn't changed between filtration steps) explains why the GPU-batched
extractor shows little/no speedup on H200 hardware.

For one protein conformation, and for each of the 15 side-chain combos, this
reports:
  - N            : number of selected residue centroids for that combo
  - cpu_calls    : how many of the 200 filtration steps actually triggered a
                   fresh np.linalg.eigvalsh call on CPU (vs. how many were
                   served from the cached previous result)
  - cpu_time     : wall time for the CPU dim-0 persistent Laplacian filtration
  - gpu_calls    : how many underlying torch.linalg.eigvalsh invocations the
                   batched GPU path made (should be ~1-2 per combo regardless
                   of N, since it always processes all 200 thresholds at once)
  - gpu_matrices : total NxN matrices fed to eigvalsh on GPU (always ~200,
                   since there is no caching/dedup in the batched path)
  - gpu_time     : wall time for the GPU batched filtration for that combo

If the hypothesis is correct, combos with small N should show cpu_calls << 200
(heavy caching) and cpu_time << gpu_time, while combos with large N (the
triples/quadruple) should show cpu_calls close to 200 and a more competitive
or GPU-favorable time.

Usage
-----
    python protein_function/scripts/profile_sidechain_bottleneck.py \\
        --protein_id A0A017T5A5 \\
        --pdb_dir /mnt/research/nodes/giacomo/asam_ensembles/protein_function_prediction/v0/sampling \\
        --device cuda
"""

import argparse
import contextlib
import os
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from code_pkg.top_embedding import SimplicialComplex_laplacian  # noqa: E402
from protein_function.topo_extraction.ensemble_motion_topo_embedding import (  # noqa: E402
    find_ensemble_pdbs,
)
from protein_function.topo_extraction.sidechain_topo_embedding import (  # noqa: E402
    CLASS_LABELS,
    SIDECHAIN_COMBINATIONS,
    parse_residue_sidechain_centroids,
)
from protein_function.topo_extraction.sidechain_topo_embedding_gpu import (  # noqa: E402
    _combo_stats_over_thresholds,
)


# ---------------------------------------------------------------------------
# Call-counting instrumentation (no production code is modified — these are
# context managers that temporarily wrap the underlying eigensolver calls).
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _count_numpy_eigvalsh(scl):
    """Patch the *bound* ``scl.eigvalue_calculator``, not the module-level
    ``np.linalg.eigvalsh`` — ``SimplicialComplexLaplacian.__init__`` captures a
    direct reference to ``np.linalg.eigvalsh`` at construction time, so
    patching the module attribute afterward would never be seen by an
    already-constructed ``scl`` instance.
    """
    counter = {"calls": 0}
    orig = scl.eigvalue_calculator

    def wrapped(*a, **kw):
        counter["calls"] += 1
        return orig(*a, **kw)

    scl.eigvalue_calculator = wrapped
    try:
        yield counter
    finally:
        scl.eigvalue_calculator = orig


@contextlib.contextmanager
def _count_torch_eigvalsh():
    counter = {"calls": 0, "matrices": 0}
    orig = torch.linalg.eigvalsh

    def wrapped(mats, *a, **kw):
        counter["calls"] += 1
        counter["matrices"] += mats.shape[0] if mats.ndim == 3 else 1
        return orig(mats, *a, **kw)

    torch.linalg.eigvalsh = wrapped
    try:
        yield counter
    finally:
        torch.linalg.eigvalsh = orig


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein_id", required=True)
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--n_conformers", type=int, default=10)
    parser.add_argument("--conf_index", type=int, default=0,
                        help="Which conformation to profile (default: first).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument("--dis_start", type=float, default=0.0)
    parser.add_argument("--dis_cutoff", type=float, default=40.0)
    parser.add_argument("--dis_step", type=float, default=0.2)
    parser.add_argument("--max_batch_matrices", type=int, default=4096)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    np_dtype = np.float32 if args.dtype == "float32" else np.float64

    pdb_files = find_ensemble_pdbs(args.pdb_dir, args.protein_id, args.n_conformers)
    if not pdb_files:
        print(f"[ERROR] No PDB files for {args.protein_id} in {args.pdb_dir}")
        sys.exit(1)
    pdb_file = pdb_files[args.conf_index]
    print(f"Profiling {args.protein_id} | conformation: {os.path.basename(pdb_file)}\n")

    classes, centroids = parse_residue_sidechain_centroids(pdb_file)
    print(f"Total residues with a centroid: {len(classes)}\n")

    filtration = np.round(np.arange(args.dis_start, args.dis_cutoff, args.dis_step), 6)
    n_filt = len(filtration)
    thresholds_t = torch.as_tensor(filtration, dtype=dtype, device=device)

    scl = SimplicialComplex_laplacian.SimplicialComplexLaplacian()

    header = (f"{'combo':<22} {'N':>5} {'cpu_calls':>10} {'cpu_skip%':>10} "
              f"{'cpu_time':>10} {'gpu_calls':>10} {'gpu_matrices':>13} {'gpu_time':>10}")
    print(header)
    print("-" * len(header))

    total_cpu_time = 0.0
    total_gpu_time = 0.0

    for class_combo in SIDECHAIN_COMBINATIONS:
        label = "+".join(CLASS_LABELS[c] for c in class_combo)
        mask = np.isin(classes, class_combo)
        sel_xyz = centroids[mask]
        n = len(sel_xyz)

        if n < 2:
            print(f"{label:<22} {n:>5} {'--':>10} {'--':>10} {'--':>10} "
                  f"{'--':>10} {'--':>13} {'--':>10}  (skipped: N<2)")
            continue

        dist_mat = cdist(sel_xyz.astype(np_dtype), sel_xyz.astype(np_dtype),
                         metric="euclidean")

        # ---- CPU reference, instrumented ----
        t0 = time.perf_counter()
        with _count_numpy_eigvalsh(scl) as cpu_counter:
            scl.persistent_simplicialComplex_laplacian_dim0(
                input_data=dist_mat, is_distance_matrix=True, max_dim=0,
                filtration=filtration, print_by_step=False,
            )
        cpu_time = time.perf_counter() - t0
        total_cpu_time += cpu_time

        # ---- GPU batched path, instrumented ----
        coords_t = torch.as_tensor(sel_xyz, dtype=dtype, device=device).unsqueeze(0)  # [1,N,3]
        dist_t = torch.cdist(coords_t, coords_t)  # [1,N,N]
        eye = torch.eye(n, dtype=torch.bool, device=device)

        _sync(device)
        t0 = time.perf_counter()
        with _count_torch_eigvalsh() as gpu_counter:
            _combo_stats_over_thresholds(
                dist_t, thresholds_t, eye, args.max_batch_matrices,
            )
        _sync(device)
        gpu_time = time.perf_counter() - t0
        total_gpu_time += gpu_time

        skip_pct = 100.0 * (1.0 - cpu_counter["calls"] / n_filt)
        print(f"{label:<22} {n:>5} {cpu_counter['calls']:>10} {skip_pct:>9.1f}% "
              f"{cpu_time:>9.3f}s {gpu_counter['calls']:>10} "
              f"{gpu_counter['matrices']:>13} {gpu_time:>9.3f}s")

    print("-" * len(header))
    print(f"{'TOTAL (this conformation)':<22} {'':>5} {'':>10} {'':>10} "
          f"{total_cpu_time:>9.3f}s {'':>10} {'':>13} {total_gpu_time:>9.3f}s")
    print(f"\nSingle-conformation speedup: {total_cpu_time / max(total_gpu_time, 1e-9):.2f}x")
    print("(Full protein time = this x ~n_conformers, plus PDB-parsing overhead "
          "shared by both paths and not measured here.)")


if __name__ == "__main__":
    main()
