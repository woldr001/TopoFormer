"""Batch pre-computation of protein-only Persistent Laplacian topology features.

For each protein PDB file, runs the topology extraction pipeline from
``protein_only_topo_embedding.py`` and saves the result as a
``<protein_id>.npy`` file of shape [6, 200, 15].

Supports multiprocessing for parallelisation across CPU cores.

Usage
-----
    python protein_function/scripts/precompute_topo_features.py \\
        --pdb_dir     /data/alphafold_pdbs \\
        --output_dir  /data/topo_features \\
        --dis_cutoff  20.0 \\
        --dis_step    0.1 \\
        --n_workers   8

    # Or pass an explicit file list:
    python protein_function/scripts/precompute_topo_features.py \\
        --pdb_list    /data/protein_ids.txt \\
        --pdb_dir     /data/pdbs \\
        --output_dir  /data/topo_features \\
        --n_workers   4

PDB files are expected to be named ``<protein_id>.pdb`` or
``<protein_id>_protein.pdb`` / ``<protein_id>_pocket.pdb`` inside ``pdb_dir``.
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


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _process_one(
    protein_id: str,
    pdb_dir: str,
    output_dir: str,
    dis_start: float,
    dis_cutoff: float,
    dis_step: float,
    overwrite: bool,
) -> str:
    """Process a single protein. Returns protein_id on success, or error string."""
    out_path = os.path.join(output_dir, f"{protein_id}.npy")
    if os.path.exists(out_path) and not overwrite:
        return f"SKIP {protein_id}"

    # Try common PDB naming conventions
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch topology feature extraction for protein function prediction."
    )
    parser.add_argument("--pdb_dir", required=True,
                        help="Directory containing protein PDB files.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for output .npy feature files.")
    parser.add_argument("--pdb_list", default=None,
                        help="Text file with one protein_id per line. "
                             "If omitted, all *.pdb files in pdb_dir are used.")
    parser.add_argument("--dis_start",  type=float, default=0.0)
    parser.add_argument("--dis_cutoff", type=float, default=20.0)
    parser.add_argument("--dis_step",   type=float, default=0.1)
    parser.add_argument("--n_workers",  type=int,   default=4,
                        help="Number of parallel worker processes.")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-compute features even if output file exists.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect protein IDs
    if args.pdb_list:
        with open(args.pdb_list) as f:
            protein_ids = [l.strip() for l in f if l.strip()]
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

    logger.info(f"Processing {len(protein_ids)} proteins with {args.n_workers} workers ...")
    t0 = time.time()

    worker = partial(
        _process_one,
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
