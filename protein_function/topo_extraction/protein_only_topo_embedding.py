"""Protein-only topological feature extraction using Persistent Simplicial Complex Laplacians.

Adapted from code_pkg/main_potein_ligand_topo_embedding.py.

Key differences from the protein-ligand version:
- No ligand file required; all protein heavy atoms (C, N, O, S) are used.
- No binding-pocket proximity filter (uses the entire protein structure).
- Distance matrix is computed as true all-pairs intra-protein distances via
  scipy.spatial.distance.cdist (instead of the cross-pair sentinel values used
  in the protein-ligand version).
- 15 element-specific combinations (single × 4, pairs × 6, triples × 4, all × 1)
  instead of 143 cross-pair combinations.
- Filtration range: 0–20 Å at 0.1 Å steps (200 steps) to capture full-protein
  contacts rather than just binding-pocket geometry.
- Output shape: [6, 200, 15] float32 numpy array.

Usage (command line):
    python protein_only_topo_embedding.py \\
        --protein_file /path/to/protein.pdb \\
        --output_folder /path/to/output \\
        --dis_start 0 --dis_cutoff 20 --dis_step 0.1

Usage (Python API):
    from protein_function.topo_extraction.protein_only_topo_embedding import (
        generate_protein_lap_features,
    )
    feat = generate_protein_lap_features(
        output_folder="./features",
        protein_file="protein.pdb",
    )  # returns np.ndarray of shape [6, 200, 15]
"""

import sys
import os
import argparse
import time
import warnings

import numpy as np
from scipy.spatial.distance import cdist

# Allow running from the repository root or as a module
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from code_pkg.top_embedding import SimplicialComplex_laplacian


# ---------------------------------------------------------------------------
# Element combinations for protein-only topology
# ---------------------------------------------------------------------------

PROTEIN_ONLY_COMBINATIONS = [
    # single-element (4)
    ("C",),
    ("N",),
    ("O",),
    ("S",),
    # two-element (6)
    ("C", "N"),
    ("C", "O"),
    ("C", "S"),
    ("N", "O"),
    ("N", "S"),
    ("O", "S"),
    # three-element (4)
    ("C", "N", "O"),
    ("C", "N", "S"),
    ("C", "O", "S"),
    ("N", "O", "S"),
    # all heavy atoms (1)
    ("C", "N", "O", "S"),
]
"""15 element-specific combinations for protein-only topology."""

NUM_COMBINATIONS = len(PROTEIN_ONLY_COMBINATIONS)   # 15
NUM_STATISTICS   = 6                                 # spectral statistics per filtration step


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

def get_data_from_protein_pdb(
    protein_pdb_file: str,
    consider_ele: list = None,
) -> tuple:
    """Parse ATOM records from a PDB file and return element labels and coordinates.

    Reuses the same logic as ``get_data_from_protein_PDB`` in
    ``main_potein_ligand_topo_embedding.py``, with an added fallback to the
    standard PDB element column (columns 76-77) to handle multi-character atom
    names (CA, CB, etc.) that would otherwise be mis-parsed.

    Args:
        protein_pdb_file: Path to the PDB file.
        consider_ele: Elements to retain. Defaults to ['C', 'N', 'O', 'S'].

    Returns:
        (elements, coordinates) where elements is a numpy array of strings and
        coordinates is a float64 array of shape [N, 3].
    """
    if consider_ele is None:
        consider_ele = ["C", "N", "O", "S"]

    selected_ele, selected_xyz = [], []
    with open(protein_pdb_file) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue

            # Try standard element column first (cols 76-77, 0-indexed 76:78)
            element_col = line[76:78].strip() if len(line) > 76 else ""
            if element_col and element_col in consider_ele:
                ele = element_col
            else:
                # Fall back to first character of atom name field (col 13, 0-indexed 12)
                atom_name = line[12:16].strip()
                ele = atom_name[0] if atom_name else ""
                if ele not in consider_ele:
                    continue

            # Extract XYZ using the '.' delimiter approach from original code
            try:
                parts = line.strip().split(".")
                x = parts[0][-4:].strip() + "." + parts[1][:3].strip()
                y = parts[1][-4:].strip() + "." + parts[2][:3].strip()
                z = parts[2][-4:].strip() + "." + parts[3][:3].strip()
                selected_ele.append(ele)
                selected_xyz.append([float(x), float(y), float(z)])
            except (IndexError, ValueError):
                warnings.warn(
                    f"Could not parse coordinates for line: {line.strip()!r}",
                    stacklevel=2,
                )
                continue

    return np.array(selected_ele), np.array(selected_xyz, dtype=np.float64)


# ---------------------------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------------------------

def generate_protein_distance_matrix(
    protein_xyz: np.ndarray,
) -> np.ndarray:
    """Compute all-pairs intra-protein Euclidean distance matrix.

    Unlike the cross-molecule version in the original code (which uses sentinel
    values of ``cutoff × 10000`` for intra-molecule pairs), this function
    computes the true pairwise distances for all atoms in the given coordinate
    set.

    Args:
        protein_xyz: Float array of shape [M, 3].

    Returns:
        Symmetric distance matrix of shape [M, M].
    """
    if len(protein_xyz) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    return cdist(protein_xyz, protein_xyz, metric="euclidean")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def generate_protein_lap_features(
    output_folder: str,
    protein_file: str,
    output_feature_name: str = None,
    dis_start: float = 0.0,
    dis_cutoff: float = 20.0,
    dis_step: float = 0.1,
    consider_ele: list = None,
    print_progress: bool = True,
) -> np.ndarray:
    """Extract protein-only Persistent Laplacian topological features.

    Runs the Persistent Simplicial Complex Laplacian (dim-0) over 15
    element-specific atom subsets of the protein, producing a feature tensor
    of shape [6, n_filtrations, 15].

    Args:
        output_folder: Directory where the ``.npy`` file will be saved.
        protein_file: Path to the protein PDB file.
        output_feature_name: Base name for the output file (without extension).
            Defaults to the protein file stem.
        dis_start: Start of the filtration range in Ångströms.
        dis_cutoff: End of the filtration range in Ångströms.
        dis_step: Filtration step size in Ångströms.
        consider_ele: Protein elements to use. Defaults to ['C', 'N', 'O', 'S'].
        print_progress: Whether to print progress messages.

    Returns:
        Feature array of shape [6, n_filtrations, 15] (float32), also saved
        as ``<output_folder>/<output_feature_name>.npy``.
    """
    if consider_ele is None:
        consider_ele = ["C", "N", "O", "S"]

    # Parse PDB
    protein_ele, protein_xyz = get_data_from_protein_pdb(protein_file, consider_ele)
    if print_progress:
        print(f"[topo] Loaded {len(protein_ele)} atoms ({', '.join(consider_ele)}) "
              f"from {os.path.basename(protein_file)}")

    if len(protein_ele) == 0:
        warnings.warn(f"No atoms found in {protein_file}. Feature will be all zeros.")

    # Filtration parameters
    filtration = np.round(np.arange(dis_start, dis_cutoff, dis_step), 6)
    n_filtrations = len(filtration)
    feature_array = np.zeros([NUM_STATISTICS, n_filtrations, NUM_COMBINATIONS], dtype=np.float32)

    scl = SimplicialComplex_laplacian.SimplicialComplexLaplacian()

    for combo_idx, element_combo in enumerate(PROTEIN_ONLY_COMBINATIONS):
        # Select atoms matching this element subset
        mask = np.array([e in element_combo for e in protein_ele])
        sel_xyz = protein_xyz[mask]

        if print_progress:
            print(f"[topo] Combination {combo_idx+1:2d}/{NUM_COMBINATIONS} "
                  f"{element_combo}: {mask.sum()} atoms")

        if len(sel_xyz) < 2:
            # Not enough atoms to form edges; leave as zeros
            continue

        dist_mat = generate_protein_distance_matrix(sel_xyz)

        laplacian_features = scl.persistent_simplicialComplex_laplacian_dim0(
            input_data=dist_mat,
            is_distance_matrix=True,
            max_dim=0,
            filtration=filtration,
            print_by_step=False,
        )

        for filt_idx, (eigvals, _) in enumerate(laplacian_features):
            stats = SimplicialComplex_laplacian.statistic_eigvalues(eigvals)
            feature_array[:, filt_idx, combo_idx] = [
                stats.count_zero,
                stats.max,
                stats.sum,
                stats.nonzero_mean,
                stats.nonzero_std,
                stats.nonzero_min,
            ]

    # Determine output file name
    if output_feature_name is None:
        stem = os.path.splitext(os.path.basename(protein_file))[0]
        # Strip common suffixes like _protein, _pocket
        for suffix in ("_protein", "_pocket"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        output_feature_name = stem

    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"{output_feature_name}.npy")
    np.save(out_path, feature_array, allow_pickle=True)
    if print_progress:
        print(f"[topo] Saved features {feature_array.shape} → {out_path}")

    return feature_array


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Extract protein-only Persistent Laplacian topological features.\n"
            "Output: [6, n_filtrations, 15] float32 numpy array saved as .npy"
        )
    )
    parser.add_argument("--protein_file", required=True,
                        help="Path to protein PDB file.")
    parser.add_argument("--output_folder", default="./topo_features",
                        help="Directory for output .npy file (default: ./topo_features).")
    parser.add_argument("--output_feature_name", default=None,
                        help="Output file base name. Defaults to PDB file stem.")
    parser.add_argument("--dis_start", type=float, default=0.0,
                        help="Filtration start in Å (default: 0.0).")
    parser.add_argument("--dis_cutoff", type=float, default=20.0,
                        help="Filtration end in Å (default: 20.0).")
    parser.add_argument("--dis_step", type=float, default=0.1,
                        help="Filtration step in Å (default: 0.1).")
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    t0 = time.time()
    generate_protein_lap_features(
        output_folder=args.output_folder,
        protein_file=args.protein_file,
        output_feature_name=args.output_feature_name,
        dis_start=args.dis_start,
        dis_cutoff=args.dis_cutoff,
        dis_step=args.dis_step,
        print_progress=True,
    )
    print(f"[topo] Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
