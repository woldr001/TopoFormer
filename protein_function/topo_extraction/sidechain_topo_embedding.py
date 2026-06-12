"""Side-chain centroid ensemble topology feature extraction for protein function prediction.

This module is a third feature configuration alongside:
  - ``protein_only_topo_embedding.py``      (single PDB, atom-level, [6, 200, 15])
  - ``ensemble_motion_topo_embedding.py``   (ensemble, motion-guided cross-pairs, [12, 200, 121])

**Side-chain centroid configuration**

Instead of treating individual heavy atoms (C, N, O, S) as the topological points,
each residue is collapsed to a single point: the **geometric centroid of its
side-chain heavy atoms** (all heavy atoms excluding the backbone N, Cα, C, O).
Glycine — which has no side-chain heavy atoms — falls back to its Cα coordinate.

The "element combinations" of the atom-level schemes become **side-chain-type
combinations**.  Each residue is assigned to one of four chemical classes:

    0. nonpolar / hydrophobic      G A V L I M P
    1. aromatic                    F W Y
    2. polar / uncharged           S T C N Q
    3. polar / charged             D E K R H

Topology is computed over a **flat list of 15 type combinations** (not a
cross-product): every non-empty subset of the four classes —
4 singles + 6 pairs + 4 triples + 1 quadruple = 15.  For each combination the
centroids of all residues whose class is in the subset are collected and a full
all-pairs Persistent Simplicial Complex Laplacian (dim-0) filtration is run over
them.

Each protein's structural ensemble (10 conformations) is processed individually
and the per-conformation tensors are aggregated as **mean + std** across the
ensemble, doubling the 6 spectral statistics to 12 channels.

Output shape
------------
``[12, 200, 15]`` float32 per protein, where:
- **12** = 6 spectral statistics × 2 ensemble aggregations (mean + std)
- **200** = filtration steps: ``np.arange(0.0, 40.0, 0.2)``
- **15** = side-chain-type combinations (flat list)

Usage (CLI)
-----------
    python sidechain_topo_embedding.py \\
        --protein_id    P12345 \\
        --pdb_dir       /data/ensembles/P12345 \\
        --output_folder /data/topo_sidechain

Usage (API)
-----------
    from protein_function.topo_extraction.sidechain_topo_embedding import (
        generate_sidechain_lap_features,
    )
    feat = generate_sidechain_lap_features(
        output_folder="./features",
        protein_id="P12345",
        pdb_files=["conf_0.pdb", ..., "conf_9.pdb"],
    )  # shape [12, 200, 15]
"""

import argparse
import itertools
import os
import sys
import time
import warnings
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from code_pkg.top_embedding import SimplicialComplex_laplacian
from protein_function.topo_extraction.ensemble_motion_topo_embedding import (
    find_ensemble_pdbs,
)


# ---------------------------------------------------------------------------
# Residue chemical-class definitions
# ---------------------------------------------------------------------------

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Class index → one-letter residues
CLASS_RESIDUES = {
    0: set("GAVLIMP"),   # nonpolar / hydrophobic
    1: set("FWY"),       # aromatic
    2: set("STCNQ"),     # polar / uncharged
    3: set("DEKRH"),     # polar / charged
}
CLASS_LABELS = {
    0: "nonpolar",
    1: "aromatic",
    2: "polar_uncharged",
    3: "polar_charged",
}
NUM_CLASSES = 4

# one-letter residue → class index
ONE_TO_CLASS = {}
for _cls, _residues in CLASS_RESIDUES.items():
    for _r in _residues:
        ONE_TO_CLASS[_r] = _cls

# Backbone atom names excluded from the side-chain centroid
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


# ---------------------------------------------------------------------------
# Side-chain-type combinations (flat list of 15)
# ---------------------------------------------------------------------------

def _build_sidechain_combinations() -> List[Tuple[int, ...]]:
    """All non-empty subsets of the 4 classes: 4 + 6 + 4 + 1 = 15, ordered by size."""
    combos: List[Tuple[int, ...]] = []
    for r in range(1, NUM_CLASSES + 1):
        combos.extend(itertools.combinations(range(NUM_CLASSES), r))
    return combos

SIDECHAIN_COMBINATIONS = _build_sidechain_combinations()  # 15 tuples of class indices
NUM_COMBINATIONS   = len(SIDECHAIN_COMBINATIONS)   # 15
NUM_STATISTICS     = 6   # spectral statistics per filtration step
NUM_ENSEMBLE_STATS = 2   # mean + std across ensemble → 12 channels


# ---------------------------------------------------------------------------
# PDB parsing → per-residue side-chain centroids and classes
# ---------------------------------------------------------------------------

def parse_residue_sidechain_centroids(
    pdb_file: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a PDB file into per-residue side-chain centroids and chemical classes.

    For each standard residue:
      - centroid = geometric mean of all side-chain heavy atoms
        (all heavy atoms excluding backbone N, Cα, C, O, OXT)
      - glycine (and any residue with no side-chain heavy atoms) falls back to
        its Cα coordinate
      - the residue is assigned one of the four chemical classes (0-3)

    Non-standard residues (not in the 20-letter table) are skipped.  Only the
    blank/``A`` alternate-location conformer is kept.

    Args:
        pdb_file: Path to a PDB file.

    Returns:
        (classes [R], centroids [R, 3]) where R is the number of residues
        successfully reduced to a centroid.  Empty arrays if none found.
    """
    # residue_key -> dict with resname, ca_xyz, sidechain_xyz list
    residues = {}
    order = []

    try:
        with open(pdb_file) as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue

                altloc = line[16]
                if altloc not in (" ", "A"):
                    continue

                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21]
                resseq = line[22:26].strip()
                icode = line[26]

                # Element: standard col 76-77, else infer from atom name
                element = line[76:78].strip() if len(line) > 76 else ""
                if not element:
                    element = atom_name[0] if atom_name else ""

                # Skip hydrogens
                if element == "H" or atom_name.startswith("H"):
                    continue

                try:
                    xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                except ValueError:
                    continue

                key = (chain, resseq, icode)
                if key not in residues:
                    residues[key] = {"resname": resname, "ca": None, "sc": []}
                    order.append(key)

                if atom_name == "CA":
                    residues[key]["ca"] = xyz
                if atom_name not in BACKBONE_ATOMS:
                    residues[key]["sc"].append(xyz)
    except OSError:
        return np.empty(0, dtype=int), np.empty((0, 3), dtype=np.float64)

    classes, centroids = [], []
    for key in order:
        info = residues[key]
        one = THREE_TO_ONE.get(info["resname"])
        if one is None:
            continue  # non-standard residue
        cls = ONE_TO_CLASS.get(one)
        if cls is None:
            continue

        sc = info["sc"]
        if sc:
            centroid = np.mean(sc, axis=0)
        elif info["ca"] is not None:
            # Glycine, or a residue with only backbone atoms resolved → use Cα
            centroid = np.array(info["ca"], dtype=np.float64)
        else:
            continue  # no usable coordinate

        classes.append(cls)
        centroids.append(centroid)

    if not centroids:
        return np.empty(0, dtype=int), np.empty((0, 3), dtype=np.float64)

    return (
        np.array(classes, dtype=int),
        np.array(centroids, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Per-conformation feature extraction
# ---------------------------------------------------------------------------

def _extract_single_conformation_features(
    pdb_file: str,
    filtration: np.ndarray,
    scl: SimplicialComplex_laplacian.SimplicialComplexLaplacian,
) -> np.ndarray:
    """Extract [6, n_filtrations, 15] side-chain-centroid features for one PDB.

    Args:
        pdb_file: Path to one PDB conformation.
        filtration: Filtration parameter array (e.g. np.arange(0, 40, 0.2)).
        scl: Reusable SimplicialComplexLaplacian instance.

    Returns:
        Feature array of shape [6, n_filtrations, 15] as float32.
    """
    n_filtrations = len(filtration)
    feature_array = np.zeros([NUM_STATISTICS, n_filtrations, NUM_COMBINATIONS],
                             dtype=np.float32)

    classes, centroids = parse_residue_sidechain_centroids(pdb_file)
    if len(classes) == 0:
        return feature_array

    for combo_idx, class_combo in enumerate(SIDECHAIN_COMBINATIONS):
        mask = np.isin(classes, class_combo)
        sel_xyz = centroids[mask]

        if len(sel_xyz) < 2:
            # Not enough centroids to form edges; leave as zeros
            continue

        dist_mat = cdist(sel_xyz, sel_xyz, metric="euclidean")

        try:
            lap_features = scl.persistent_simplicialComplex_laplacian_dim0(
                input_data=dist_mat,
                is_distance_matrix=True,
                max_dim=0,
                filtration=filtration,
                print_by_step=False,
            )
            # Each element is a dict {0: [sorted eigenvalues]} from the dim-0 Laplacian.
            for filt_idx, laplacian_eigenv in enumerate(lap_features):
                eigvals = np.array(laplacian_eigenv[0], dtype=np.float64)
                stats = SimplicialComplex_laplacian.statistic_eigvalues(eigvals)
                feature_array[:, filt_idx, combo_idx] = [
                    stats.count_zero,
                    stats.max,
                    stats.sum,
                    stats.nonzero_mean,
                    stats.nonzero_std,
                    stats.nonzero_min,
                ]
        except Exception as exc:
            warnings.warn(
                f"[topo] Laplacian failed for combo {combo_idx} in {pdb_file}: {exc}",
                stacklevel=2,
            )

    return feature_array


# ---------------------------------------------------------------------------
# Main ensemble feature extraction
# ---------------------------------------------------------------------------

def generate_sidechain_lap_features(
    output_folder: str,
    protein_id: str,
    pdb_files: List[str],
    output_feature_name: Optional[str] = None,
    dis_start: float = 0.0,
    dis_cutoff: float = 40.0,
    dis_step: float = 0.2,
    ensemble_aggregation: str = "mean_std",
    print_progress: bool = True,
) -> np.ndarray:
    """Extract side-chain-centroid ensemble topology features for a protein.

    Full pipeline:
      1. For each PDB conformation, reduce every residue to its side-chain
         centroid and assign a chemical class (0-3).
      2. For each of 15 class combinations, run the all-pairs Persistent
         Laplacian (dim-0) filtration over the selected centroids → [6, n_filt, 15].
      3. Aggregate across conformations: mean + std → [12, n_filt, 15].
      4. Save as ``<output_folder>/<output_feature_name>.npy``.

    Args:
        output_folder: Directory to save the ``.npy`` output.
        protein_id: Identifier used for messages and the default file name.
        pdb_files: List of PDB conformation file paths (length = ensemble size).
        output_feature_name: Base name for the output file. Defaults to protein_id.
        dis_start: Filtration start in Å (default 0.0).
        dis_cutoff: Filtration end in Å (default 40.0).
        dis_step: Filtration step in Å (default 0.2 → 200 steps over 0–40 Å).
        ensemble_aggregation: How to combine conformations:
            - ``'mean_std'`` (default): stack mean+std → 12 channels
            - ``'mean_only'``: average only → 6 channels
            - ``'all'``: concatenate all conformations → 6*N_conformers channels
        print_progress: Whether to print per-step progress.

    Returns:
        Feature array of shape [C, n_filtrations, 15] as float32, where C depends
        on ensemble_aggregation (12 for mean_std, 6 for mean_only).
        Also saved as ``<output_folder>/<output_feature_name>.npy``.
    """
    if not pdb_files:
        raise ValueError("pdb_files must be a non-empty list.")

    # ------------------------------------------------------------------
    # 1. Filtration parameters
    # ------------------------------------------------------------------
    filtration = np.round(np.arange(dis_start, dis_cutoff, dis_step), 6)
    n_filtrations = len(filtration)

    # ------------------------------------------------------------------
    # 2. Per-conformation feature extraction
    # ------------------------------------------------------------------
    scl = SimplicialComplex_laplacian.SimplicialComplexLaplacian()
    per_conf_features = []

    for conf_idx, pdb_file in enumerate(pdb_files):
        if print_progress:
            print(f"[topo]   conformation {conf_idx+1}/{len(pdb_files)}: "
                  f"{os.path.basename(pdb_file)}")
        feat = _extract_single_conformation_features(
            pdb_file=pdb_file,
            filtration=filtration,
            scl=scl,
        )
        per_conf_features.append(feat)

    conf_stack = np.stack(per_conf_features, axis=0)  # [N_conf, 6, n_filt, 15]

    # ------------------------------------------------------------------
    # 3. Ensemble aggregation
    # ------------------------------------------------------------------
    if ensemble_aggregation == "mean_std":
        mean_feat = conf_stack.mean(axis=0)                 # [6, n_filt, 15]
        std_feat  = conf_stack.std(axis=0)                  # [6, n_filt, 15]
        output    = np.concatenate([mean_feat, std_feat], axis=0)  # [12, n_filt, 15]
    elif ensemble_aggregation == "mean_only":
        output = conf_stack.mean(axis=0)                    # [6, n_filt, 15]
    elif ensemble_aggregation == "all":
        N = conf_stack.shape[0]
        output = conf_stack.reshape(N * NUM_STATISTICS, n_filtrations, NUM_COMBINATIONS)
    else:
        raise ValueError(f"Unknown ensemble_aggregation: {ensemble_aggregation!r}. "
                         "Choose 'mean_std', 'mean_only', or 'all'.")

    output = output.astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    fname = output_feature_name if output_feature_name else protein_id
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"{fname}.npy")
    np.save(out_path, output, allow_pickle=True)

    if print_progress:
        print(f"[topo] Saved {output.shape} → {out_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Extract side-chain-centroid ensemble Persistent Laplacian features.\n"
            "Output: [12, 200, 15] float32 numpy array (mean_std mode)."
        )
    )
    parser.add_argument("--protein_id",    required=True)
    parser.add_argument("--pdb_dir",       required=True,
                        help="Root directory containing ensemble PDB files.")
    parser.add_argument("--output_folder", default="./topo_sidechain_features")
    parser.add_argument("--n_conformers",  type=int, default=10)
    parser.add_argument("--dis_start",     type=float, default=0.0)
    parser.add_argument("--dis_cutoff",    type=float, default=40.0)
    parser.add_argument("--dis_step",      type=float, default=0.2)
    parser.add_argument("--ensemble_aggregation", default="mean_std",
                        choices=["mean_std", "mean_only", "all"])
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

    generate_sidechain_lap_features(
        output_folder=args.output_folder,
        protein_id=args.protein_id,
        pdb_files=pdb_files,
        dis_start=args.dis_start,
        dis_cutoff=args.dis_cutoff,
        dis_step=args.dis_step,
        ensemble_aggregation=args.ensemble_aggregation,
        print_progress=True,
    )
    print(f"[topo] Finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
