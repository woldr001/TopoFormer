"""Batch pre-computation of protein topology features.

Supports two modes selected with ``--mode``:

**protein_only** (default):
    Single PDB per protein.  Output shape: [6, 200, 15].
    Usage::

        python protein_function/scripts/precompute_topo_features.py \\
            --pdb_dir    /data/alphafold_pdbs \\
            --output_dir /data/topo_features \\
            --n_workers  8

**ensemble_motion**:
    Ensemble of N PDB conformations per protein + GNM/ANM/PCA profiles.
    Output shape: [12, 200, 121].
    Usage::

        python protein_function/scripts/precompute_topo_features.py \\
            --mode           ensemble_motion \\
            --pdb_dir        /data/ensembles \\
            --nma_pca_dir    /data/nma_pca \\
            --output_dir     /data/topo_ensemble \\
            --n_conformers   10 \\
            --top_motion_pct 0.20 \\
            --n_workers      4

    Each protein's PDB ensemble is expected under
    ``<pdb_dir>/<protein_id>/cluster_repr_*.pdb`` (or similar; see
    ``find_ensemble_pdbs`` for the full search order).
    NMA/PCA profiles are expected under ``<nma_pca_dir>/<protein_id>/``.
"""

import argparse
import glob
import logging
import os
import sys
import time
from multiprocessing import Pool
from functools import partial

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from protein_function.topo_extraction.protein_only_topo_embedding import (
    generate_protein_lap_features,
)
from protein_function.topo_extraction.ensemble_motion_topo_embedding import (
    generate_ensemble_motion_lap_features,
    find_ensemble_pdbs,
)


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------

def _process_one_single(
    protein_id: str,
    pdb_dir: str,
    output_dir: str,
    dis_start: float,
    dis_cutoff: float,
    dis_step: float,
    overwrite: bool,
) -> str:
    """protein_only mode: process one protein from a single PDB file."""
    out_path = os.path.join(output_dir, f"{protein_id}.npy")
    if os.path.exists(out_path) and not overwrite:
        return f"SKIP {protein_id}"

    candidates = [
        os.path.join(pdb_dir, f"{protein_id}.pdb"),
        os.path.join(pdb_dir, f"{protein_id}_protein.pdb"),
        os.path.join(pdb_dir, f"{protein_id}_pocket.pdb"),
        os.path.join(pdb_dir, protein_id, f"{protein_id}.pdb"),
    ]
    pdb_path = next((c for c in candidates if os.path.exists(c)), None)
    if pdb_path is None:
        return f"MISSING_PDB {protein_id}"

    try:
        generate_protein_lap_features(
            output_folder=output_dir,
            protein_file=pdb_path,
            output_feature_name=protein_id,
            dis_start=dis_start,
            dis_cutoff=dis_cutoff,
            dis_step=dis_step,
            print_progress=False,
        )
        return f"OK {protein_id}"
    except Exception as exc:
        return f"ERROR {protein_id}: {exc}"


def _process_one_ensemble(
    protein_id: str,
    pdb_dir: str,
    nma_pca_dir: str,
    output_dir: str,
    n_conformers: int,
    top_motion_pct: float,
    min_motion_residues: int,
    fluctuation_sources: list,
    n_slow_modes: int,
    motion_field,
    dis_start: float,
    dis_cutoff: float,
    dis_step: float,
    ensemble_aggregation: str,
    overwrite: bool,
) -> str:
    """ensemble_motion mode: process one protein from N PDB conformations."""
    out_path = os.path.join(output_dir, f"{protein_id}.npy")
    if os.path.exists(out_path) and not overwrite:
        return f"SKIP {protein_id}"

    pdb_files = find_ensemble_pdbs(pdb_dir, protein_id, n_conformers)
    if not pdb_files:
        return f"MISSING_PDB {protein_id}"

    protein_nma_dir = os.path.join(nma_pca_dir, protein_id)
    if not os.path.isdir(protein_nma_dir):
        # Try flat layout: <nma_pca_dir>/<protein_id>_*.npz
        protein_nma_dir = nma_pca_dir

    try:
        generate_ensemble_motion_lap_features(
            output_folder=output_dir,
            protein_id=protein_id,
            pdb_files=pdb_files,
            nma_pca_dir=protein_nma_dir,
            top_motion_pct=top_motion_pct,
            min_motion_residues=min_motion_residues,
            fluctuation_sources=fluctuation_sources,
            n_slow_modes=n_slow_modes,
            motion_field=motion_field,
            dis_start=dis_start,
            dis_cutoff=dis_cutoff,
            dis_step=dis_step,
            ensemble_aggregation=ensemble_aggregation,
            print_progress=False,
        )
        return f"OK {protein_id}"
    except Exception as exc:
        return f"ERROR {protein_id}: {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch topology feature extraction for protein function prediction."
    )
    parser.add_argument("--mode", default="protein_only",
                        choices=["protein_only", "ensemble_motion"],
                        help="Feature extraction mode (default: protein_only).")
    parser.add_argument("--pdb_dir", required=True,
                        help="For protein_only: directory of <protein_id>.pdb files. "
                             "For ensemble_motion: root dir where each protein has a "
                             "subdirectory of conformation PDB files.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for output .npy feature files.")
    parser.add_argument("--pdb_list", default=None,
                        help="Text file with one protein_id per line. "
                             "If omitted, protein IDs are inferred from pdb_dir.")
    parser.add_argument("--dis_start",  type=float, default=0.0)
    parser.add_argument("--dis_cutoff", type=float, default=20.0)
    parser.add_argument("--dis_step",   type=float, default=0.1)
    parser.add_argument("--n_workers",  type=int,   default=4)
    parser.add_argument("--overwrite",  action="store_true")
    # ensemble_motion-specific arguments
    parser.add_argument("--nma_pca_dir", default=None,
                        help="[ensemble_motion] Root dir with NMA/PCA .npz outputs. "
                             "Each protein expected under <nma_pca_dir>/<protein_id>/.")
    parser.add_argument("--n_conformers", type=int, default=10)
    parser.add_argument("--top_motion_pct", type=float, default=0.20)
    parser.add_argument("--min_motion_residues", type=int, default=10)
    parser.add_argument("--fluctuation_sources", nargs="+",
                        default=["gnm", "pca"],
                        choices=["gnm", "pca", "contact_map"])
    parser.add_argument("--n_slow_modes", type=int, default=3)
    parser.add_argument("--motion_field", type=float, default=None)
    parser.add_argument("--ensemble_aggregation", default="mean_std",
                        choices=["mean_std", "mean_only", "all"])
    args = parser.parse_args()

    if args.mode == "ensemble_motion" and args.nma_pca_dir is None:
        parser.error("--nma_pca_dir is required for ensemble_motion mode.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect protein IDs
    if args.pdb_list:
        with open(args.pdb_list) as f:
            protein_ids = [l.strip() for l in f if l.strip()]
    elif args.mode == "ensemble_motion":
        # Each protein has its own subdirectory
        protein_ids = sorted(
            d for d in os.listdir(args.pdb_dir)
            if os.path.isdir(os.path.join(args.pdb_dir, d))
        )
    else:
        pdb_files = glob.glob(os.path.join(args.pdb_dir, "*.pdb"))
        protein_ids = []
        for p in pdb_files:
            stem = os.path.splitext(os.path.basename(p))[0]
            for suffix in ("_protein", "_pocket"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            protein_ids.append(stem)
        protein_ids = sorted(set(protein_ids))

    logger.info(
        f"Mode: {args.mode} | Processing {len(protein_ids)} proteins "
        f"with {args.n_workers} workers ..."
    )
    t0 = time.time()

    if args.mode == "ensemble_motion":
        worker = partial(
            _process_one_ensemble,
            pdb_dir=args.pdb_dir,
            nma_pca_dir=args.nma_pca_dir,
            output_dir=args.output_dir,
            n_conformers=args.n_conformers,
            top_motion_pct=args.top_motion_pct,
            min_motion_residues=args.min_motion_residues,
            fluctuation_sources=args.fluctuation_sources,
            n_slow_modes=args.n_slow_modes,
            motion_field=args.motion_field,
            dis_start=args.dis_start,
            dis_cutoff=args.dis_cutoff,
            dis_step=args.dis_step,
            ensemble_aggregation=args.ensemble_aggregation,
            overwrite=args.overwrite,
        )
    else:
        worker = partial(
            _process_one_single,
            pdb_dir=args.pdb_dir,
            output_dir=args.output_dir,
            dis_start=args.dis_start,
            dis_cutoff=args.dis_cutoff,
            dis_step=args.dis_step,
            overwrite=args.overwrite,
        )

    ok_count, skip_count, error_count = 0, 0, 0
    if args.n_workers > 1:
        with Pool(processes=args.n_workers) as pool:
            for result in pool.imap_unordered(worker, protein_ids):
                tag = result.split()[0]
                if tag == "OK":
                    ok_count += 1
                elif tag == "SKIP":
                    skip_count += 1
                else:
                    error_count += 1
                    logger.warning(result)
    else:
        for pid in protein_ids:
            result = worker(pid)
            tag = result.split()[0]
            if tag == "OK":
                ok_count += 1
            elif tag == "SKIP":
                skip_count += 1
            else:
                error_count += 1
                logger.warning(result)

    elapsed = time.time() - t0
    logger.info(
        f"Done in {elapsed:.1f}s — "
        f"OK: {ok_count}, Skipped: {skip_count}, Errors: {error_count}"
    )


if __name__ == "__main__":
    main()
