#!/usr/bin/env python3
"""Verify the GPU side-chain extractor matches the CPU reference.

Runs both ``sidechain_topo_embedding`` (CPU) and
``sidechain_topo_embedding_gpu`` (GPU) on the same protein ensemble and reports
the maximum absolute and relative difference between the two ``[12, 200, 15]``
outputs.  Also times both so you can see the speedup.

Because the two paths share PDB parsing, the class table, and the combination
list, and the GPU path replicates the CPU's eigenvalue rounding, float32
per-conformation storage, and ddof=0 aggregation, the outputs should agree to
within floating-point noise at the 5th decimal (typ. max abs diff < 1e-3,
dominated by the large-magnitude ``sum`` statistic).

Usage
-----
    python protein_function/scripts/verify_gpu_parity.py \\
        --protein_id A0A031WDA8 \\
        --pdb_dir /mnt/research/.../v0/sampling \\
        --device cuda --dtype float64
"""

import argparse
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from protein_function.topo_extraction.ensemble_motion_topo_embedding import (  # noqa: E402
    find_ensemble_pdbs,
)
from protein_function.topo_extraction.sidechain_topo_embedding import (  # noqa: E402
    generate_sidechain_lap_features,
)
from protein_function.topo_extraction.sidechain_topo_embedding_gpu import (  # noqa: E402
    generate_sidechain_lap_features_gpu,
)

_DTYPE_MAP = {"float64": torch.float64, "float32": torch.float32}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein_id", required=True)
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--n_conformers", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument("--tol", type=float, default=1e-2,
                        help="Max-abs-diff threshold for a PASS (default 1e-2).")
    args = parser.parse_args()

    pdb_files = find_ensemble_pdbs(args.pdb_dir, args.protein_id, args.n_conformers)
    if not pdb_files:
        print(f"[ERROR] No PDB files for {args.protein_id} in {args.pdb_dir}")
        sys.exit(1)
    print(f"Found {len(pdb_files)} conformations for {args.protein_id}")

    # CPU reference (always writes its .npy; we point it at a temp dir)
    t0 = time.time()
    cpu_feat = generate_sidechain_lap_features(
        output_folder="/tmp/_parity_cpu", protein_id=args.protein_id,
        pdb_files=pdb_files, print_progress=False,
    )
    cpu_time = time.time() - t0
    print(f"CPU: {cpu_feat.shape} in {cpu_time:.1f}s")

    # GPU
    t0 = time.time()
    gpu_feat = generate_sidechain_lap_features_gpu(
        output_folder="/tmp/_parity_gpu", protein_id=args.protein_id,
        pdb_files=pdb_files, device=args.device, dtype=_DTYPE_MAP[args.dtype],
        print_progress=False, save=False,
    )
    gpu_time = time.time() - t0
    print(f"GPU: {gpu_feat.shape} in {gpu_time:.1f}s  (speedup {cpu_time/max(gpu_time,1e-9):.1f}x)")

    if cpu_feat.shape != gpu_feat.shape:
        print(f"[FAIL] shape mismatch: CPU {cpu_feat.shape} vs GPU {gpu_feat.shape}")
        sys.exit(1)

    abs_diff = np.abs(cpu_feat - gpu_feat)
    denom = np.maximum(np.abs(cpu_feat), 1e-8)
    rel_diff = abs_diff / denom
    max_abs = abs_diff.max()
    max_rel = rel_diff.max()

    # Per-statistic breakdown (axis 0 channels 0-5 = stats; 6-11 = std-of-stats)
    stat_names = ["count_zero", "max", "sum", "nonzero_mean", "nonzero_std", "nonzero_min"]
    print("\nPer-statistic max abs diff (mean channels):")
    for i, name in enumerate(stat_names):
        print(f"  {name:<14} {abs_diff[i].max():.3e}")

    print(f"\nOverall max abs diff: {max_abs:.3e}")
    print(f"Overall max rel diff: {max_rel:.3e}")

    if max_abs <= args.tol:
        print(f"[PASS] within tolerance {args.tol}")
    else:
        print(f"[FAIL] exceeds tolerance {args.tol} — investigate the largest-diff "
              "statistic above (large `sum` values can legitimately differ at the "
              "float32 rounding level; tighten --dtype float64 if needed).")
        sys.exit(1)


if __name__ == "__main__":
    main()
