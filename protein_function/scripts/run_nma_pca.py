#!/usr/bin/env python3
"""Compute GNM slow-mode fluctuations and PCA displacement profiles from a
structural ensemble, producing the .npz files expected by the TopoFormer-MF
topology extractor.

Output files written to <output_dir>/
    anm_gnm_results.npz   keys: gnm_flucts_mean [N_slow_modes, N_residues]
                                 resids           [N_residues]
    pca_results.npz       keys: pc1_profile      [N_residues]
                                 pc2_profile      [N_residues]
                                 resids           [N_residues]

These files are read by:
    protein_function/topo_extraction/ensemble_motion_topo_embedding.load_motion_scores()

Dependencies
------------
    pip install prody numpy

Usage
-----
    python run_nma_pca.py \\
        --pdb_dir  /path/to/A0A010/clustering.nodes_pipeline_v1 \\
        --output_dir /path/to/nma_pca/A0A010 \\
        --n_slow_modes 3 \\
        --n_pcs 2
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np

try:
    import prody
    from prody import (
        parsePDB, Ensemble, GNM, PCA,
        calcSqFlucts, confProDy,
    )
    confProDy(verbosity="none")   # suppress ProDy progress spam
except ImportError:
    sys.exit(
        "ProDy is not installed.  Run:  pip install prody\n"
        "or:  conda install -c conda-forge prody"
    )


# ---------------------------------------------------------------------------
# PDB discovery
# ---------------------------------------------------------------------------

def find_pdb_files(pdb_dir: str, n_conformers: int = 10) -> list:
    """Return sorted list of PDB files under pdb_dir.

    Searches for the aSAM naming convention first, then falls back to
    any *.pdb in the directory.
    """
    for pattern in [
        os.path.join(pdb_dir, "raw.seed_*.pdb"),
        os.path.join(pdb_dir, "cluster_repr_*.pdb"),
        os.path.join(pdb_dir, "conf_*.pdb"),
        os.path.join(pdb_dir, "*.pdb"),
    ]:
        files = sorted(glob.glob(pattern))
        if files:
            return files[:n_conformers]
    return []


# ---------------------------------------------------------------------------
# GNM analysis
# ---------------------------------------------------------------------------

def compute_gnm_fluctuations(
    pdb_files: list,
    n_slow_modes: int = 3,
    cutoff: float = 7.5,
) -> tuple:
    """Compute mean GNM slow-mode square fluctuations across all conformations.

    For each PDB file:
      1. Parse Cα atoms.
      2. Build GNM Kirchhoff matrix.
      3. Calculate the n_slow_modes slowest non-trivial modes.
      4. Extract per-residue square fluctuation for each slow mode.

    Then average across conformations.

    Returns
    -------
    gnm_flucts_mean : np.ndarray, shape [n_slow_modes, N_residues]
    resids          : np.ndarray, shape [N_residues]   (PDB residue numbers)
    """
    per_conf_flucts = []
    resids = None

    for pdb_file in pdb_files:
        try:
            struct = parsePDB(pdb_file, subset="calpha")
            if struct is None or len(struct) == 0:
                warnings.warn(f"No Cα atoms in {pdb_file}, skipping.")
                continue

            gnm = GNM()
            gnm.buildKirchhoff(struct, cutoff=cutoff)
            # Calculate extra modes to skip the trivial zero mode
            gnm.calcModes(n_modes=n_slow_modes + 6, zeros=False)

            mode_flucts = []
            for i in range(min(n_slow_modes, len(gnm))):
                sq = calcSqFlucts(gnm[i])   # [N_residues]
                mode_flucts.append(sq)

            if len(mode_flucts) < n_slow_modes:
                warnings.warn(
                    f"{pdb_file}: only {len(mode_flucts)} modes computed "
                    f"(requested {n_slow_modes})."
                )
                # Pad with zeros if too few modes
                while len(mode_flucts) < n_slow_modes:
                    mode_flucts.append(np.zeros(len(struct)))

            per_conf_flucts.append(np.array(mode_flucts))  # [n_slow_modes, N_res]

            if resids is None:
                resids = struct.getResnums().astype(int)

        except Exception as exc:
            warnings.warn(f"GNM failed for {pdb_file}: {exc}")
            continue

    if not per_conf_flucts:
        raise RuntimeError("GNM: no conformations processed successfully.")

    gnm_flucts_mean = np.mean(per_conf_flucts, axis=0)   # [n_slow_modes, N_res]
    return gnm_flucts_mean.astype(np.float32), resids


# ---------------------------------------------------------------------------
# PCA analysis
# ---------------------------------------------------------------------------

def compute_pca_profiles(pdb_files: list, n_pcs: int = 2) -> tuple:
    """Compute per-residue displacement magnitudes along the top PCs.

    Builds a ProDy Ensemble from all conformations, superposes them,
    and runs PCA.  Per-residue square fluctuations along PC1 and PC2
    are returned as the displacement profiles.

    Returns
    -------
    pc_profiles : list of np.ndarray, each [N_residues]  (length = n_pcs)
    resids      : np.ndarray [N_residues]
    """
    # Parse first structure to use as reference
    ref = None
    for pdb_file in pdb_files:
        ref = parsePDB(pdb_file, subset="calpha")
        if ref is not None and len(ref) > 0:
            break
    if ref is None:
        raise RuntimeError("PCA: could not parse any PDB file.")

    n_res  = len(ref)
    resids = ref.getResnums().astype(int)

    ens = Ensemble("ensemble")
    ens.setAtoms(ref)
    ens.setCoords(ref.getCoords())

    for pdb_file in pdb_files:
        try:
            struct = parsePDB(pdb_file, subset="calpha")
            if struct is None or len(struct) != n_res:
                warnings.warn(
                    f"PCA: skipping {pdb_file} "
                    f"(residue count {len(struct) if struct else 0} ≠ {n_res})."
                )
                continue
            ens.addCoordset(struct.getCoords())
        except Exception as exc:
            warnings.warn(f"PCA: failed to add {pdb_file}: {exc}")

    if len(ens) < 2:
        raise RuntimeError(f"PCA: need ≥2 conformations, got {len(ens)}.")

    ens.superpose()

    pca = PCA()
    pca.buildCovariance(ens)
    pca.calcModes(n_modes=n_pcs)

    pc_profiles = []
    for i in range(min(n_pcs, len(pca))):
        sq = calcSqFlucts(pca[i])   # [N_residues]
        pc_profiles.append(sq.astype(np.float32))

    # Pad with zeros if too few PCs
    while len(pc_profiles) < n_pcs:
        pc_profiles.append(np.zeros(n_res, dtype=np.float32))

    return pc_profiles, resids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GNM + PCA analysis of a protein structural ensemble."
    )
    parser.add_argument(
        "--pdb_dir", required=True,
        help="Directory containing *.pdb conformation files "
             "(e.g. .../A0A010/clustering.nodes_pipeline_v1/).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where anm_gnm_results.npz and pca_results.npz are written.",
    )
    parser.add_argument(
        "--n_conformers", type=int, default=10,
        help="Maximum number of PDB conformations to use. Default: 10.",
    )
    parser.add_argument(
        "--n_slow_modes", type=int, default=3,
        help="Number of slowest GNM modes to compute. Default: 3.",
    )
    parser.add_argument(
        "--n_pcs", type=int, default=2,
        help="Number of PCA components to compute. Default: 2.",
    )
    parser.add_argument(
        "--gnm_cutoff", type=float, default=7.5,
        help="Cα-Cα distance cutoff for GNM Kirchhoff matrix (Å). Default: 7.5.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute even if output files already exist.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gnm_out = os.path.join(args.output_dir, "anm_gnm_results.npz")
    pca_out = os.path.join(args.output_dir, "pca_results.npz")

    if not args.overwrite and os.path.exists(gnm_out) and os.path.exists(pca_out):
        print(f"SKIP (outputs exist): {args.output_dir}")
        return

    pdb_files = find_pdb_files(args.pdb_dir, args.n_conformers)
    if not pdb_files:
        print(f"ERROR: no PDB files found in {args.pdb_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(pdb_files)} PDB files found in {args.pdb_dir}")

    # --- GNM ---
    try:
        gnm_flucts, gnm_resids = compute_gnm_fluctuations(
            pdb_files, n_slow_modes=args.n_slow_modes, cutoff=args.gnm_cutoff
        )
        np.savez(gnm_out, gnm_flucts_mean=gnm_flucts, resids=gnm_resids)
        print(f"  GNM saved → {gnm_out}  shape: {gnm_flucts.shape}")
    except Exception as exc:
        print(f"  ERROR (GNM): {exc}", file=sys.stderr)

    # --- PCA ---
    try:
        pc_profiles, pca_resids = compute_pca_profiles(
            pdb_files, n_pcs=args.n_pcs
        )
        save_dict = {"resids": pca_resids}
        for i, profile in enumerate(pc_profiles, 1):
            save_dict[f"pc{i}_profile"] = profile
        np.savez(pca_out, **save_dict)
        print(f"  PCA saved → {pca_out}  n_pcs: {len(pc_profiles)}")
    except Exception as exc:
        print(f"  ERROR (PCA): {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
