"""Motion-guided ensemble topology feature extraction for protein function prediction.

This module extends the protein-only topology approach with two key improvements:

1. **Structural ensemble support**: accepts 10 (or N) PDB conformations per protein.
   Topology features are computed for each conformation, then aggregated across the
   ensemble as [mean, std] → doubling the channel count from 6 → 12.  The std
   channels encode the "geometric molecular signatures" hypothesised to carry
   additional functional information.

2. **Motion-guided cross-pair topology**: residues identified as high-motion by
   GNM/ANM slow-mode square fluctuations and/or PCA per-residue profiles from the
   Ensemble_NMA-PCA pipeline (https://github.com/woldr001/Ensemble_NMA-PCA) play
   the role of the *ligand* in the original TopoFormer protein–ligand scheme.
   Only **cross-distances** between motion atoms and all other (static) protein atoms
   are computed; intra-motion and intra-static distances receive a sentinel value
   that removes them from the persistent Laplacian filtration.

This mirrors exactly how the original binding-affinity TopoFormer captures
protein–ligand geometry: the topology encodes how the "active part" (ligand / motion
residues) is spatially embedded within its context (protein pocket / static residues).

Output shape
------------
``[12, n_filtrations, 121]`` float32 per protein, where:
- **12**  = 6 spectral statistics × 2 ensemble aggregations (mean + std)
- **n_filtrations** = ``int((dis_cutoff - dis_start) / dis_step)`` — default 200
- **121** = 11 motion-side element combos × 11 static-side element combos

Inputs required
---------------
- **Ensemble PDB files**: list/directory of N PDB files (one per conformation)
- **NMA/PCA profile files**: ``anm_gnm_results.npz`` and/or ``pca_results.npz``
  produced by the Ensemble_NMA-PCA pipeline

Usage (CLI)
-----------
    python ensemble_motion_topo_embedding.py \\
        --protein_id    P12345 \\
        --pdb_dir       /data/ensembles/P12345 \\
        --nma_pca_dir   /data/nma_pca/P12345 \\
        --output_folder /data/topo_ensemble \\
        --top_motion_pct 0.20 \\
        --fluctuation_sources gnm pca

Usage (API)
-----------
    from protein_function.topo_extraction.ensemble_motion_topo_embedding import (
        generate_ensemble_motion_lap_features,
    )
    feat = generate_ensemble_motion_lap_features(
        output_folder="./features",
        protein_id="P12345",
        pdb_files=["conf_0.pdb", ..., "conf_9.pdb"],
        nma_pca_dir="./nma_pca/P12345",
        top_motion_pct=0.20,
    )  # shape [12, 200, 121]
"""

import argparse
import glob
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from code_pkg.top_embedding import SimplicialComplex_laplacian


# ---------------------------------------------------------------------------
# Element combination schemes
# ---------------------------------------------------------------------------

# Both "motion" and "static" sides use protein heavy atoms (C, N, O, S).
# This gives 11 × 11 = 121 cross-pair element combinations.
MOTION_SIDE_COMBINATIONS = [
    ("C",),
    ("N",),
    ("O",),
    ("S",),
    ("C", "N"),
    ("C", "O"),
    ("C", "S"),
    ("N", "O"),
    ("N", "S"),
    ("O", "S"),
    ("C", "N", "O", "S"),
]  # 11 combos

STATIC_SIDE_COMBINATIONS = MOTION_SIDE_COMBINATIONS  # same 11 combos

NUM_MOTION_COMBOS  = len(MOTION_SIDE_COMBINATIONS)    # 11
NUM_STATIC_COMBOS  = len(STATIC_SIDE_COMBINATIONS)    # 11
NUM_COMBINATIONS   = NUM_MOTION_COMBOS * NUM_STATIC_COMBOS  # 121
NUM_STATISTICS     = 6   # spectral statistics per filtration step
NUM_ENSEMBLE_STATS = 2   # mean + std across ensemble → 12 total channels


# ---------------------------------------------------------------------------
# PDB parsing (with residue number extraction)
# ---------------------------------------------------------------------------

def get_protein_atoms_with_resnum(
    pdb_file: str,
    consider_ele: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse ATOM records from a PDB file, returning elements, coordinates, and
    residue numbers.

    Args:
        pdb_file: Path to PDB file.
        consider_ele: Elements to keep. Defaults to ['C', 'N', 'O', 'S'].

    Returns:
        (elements [N], coords [N,3], residue_numbers [N])
    """
    if consider_ele is None:
        consider_ele = ["C", "N", "O", "S"]

    elements, coords, resnums = [], [], []
    with open(pdb_file) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue

            # Element: standard PDB column 76-77 first, else atom-name col 12-16
            element_col = line[76:78].strip() if len(line) > 76 else ""
            if element_col and element_col in consider_ele:
                ele = element_col
            else:
                atom_name = line[12:16].strip()
                ele = atom_name[0] if atom_name else ""
                if ele not in consider_ele:
                    continue

            # Residue sequence number: columns 22-26 (0-indexed)
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue

            # Coordinates via '.' delimiter method (matches original TopoFormer parser)
            try:
                parts = line.strip().split(".")
                x = parts[0][-4:].strip() + "." + parts[1][:3].strip()
                y = parts[1][-4:].strip() + "." + parts[2][:3].strip()
                z = parts[2][-4:].strip() + "." + parts[3][:3].strip()
                elements.append(ele)
                coords.append([float(x), float(y), float(z)])
                resnums.append(resnum)
            except (IndexError, ValueError):
                warnings.warn(
                    f"[topo] Could not parse coordinates: {line.strip()!r}",
                    stacklevel=2,
                )
                continue

    return (
        np.array(elements),
        np.array(coords, dtype=np.float64),
        np.array(resnums, dtype=int),
    )


# ---------------------------------------------------------------------------
# NMA / PCA profile loading
# ---------------------------------------------------------------------------

def load_motion_scores(
    nma_pca_dir: str,
    fluctuation_sources: List[str] = None,
    n_slow_modes: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load per-residue motion scores from Ensemble_NMA-PCA output files.

    Reads the ``.npz`` files produced by the Ensemble_NMA-PCA pipeline:
      - ``anm_gnm_results.npz``: GNM slow-mode square fluctuations
      - ``pca_results.npz``:     PCA per-residue displacement magnitudes

    Combines the requested sources into a single per-residue score array.

    Args:
        nma_pca_dir: Directory containing the .npz files (one protein's outputs).
        fluctuation_sources: Which profiles to include. Any combination of
            ``'gnm'``, ``'pca'``, ``'contact_map'``. Defaults to ``['gnm', 'pca']``.
        n_slow_modes: Number of slowest GNM/ANM modes to sum over (default 3).

    Returns:
        (scores, residue_numbers) where scores is shape [N_residues] (sum of
        requested profiles, each normalised to [0,1]) and residue_numbers is
        the integer residue index array the scores correspond to.  Returns
        (None, None) if no recognised profile file is found.
    """
    if fluctuation_sources is None:
        fluctuation_sources = ["gnm", "pca"]

    score_arrays = []
    resids = None

    # --- GNM slow-mode fluctuations ---
    if "gnm" in fluctuation_sources:
        gnm_path = os.path.join(nma_pca_dir, "anm_gnm_results.npz")
        if os.path.exists(gnm_path):
            data = np.load(gnm_path, allow_pickle=True)
            if "gnm_flucts_mean" in data:
                # shape: [N_slow_modes, N_residues] — use first n_slow_modes
                flucts = data["gnm_flucts_mean"][:n_slow_modes]  # [k, N]
                gnm_score = flucts.sum(axis=0)                   # [N]
                # normalise to [0, 1]
                r = gnm_score.max() - gnm_score.min()
                if r > 0:
                    gnm_score = (gnm_score - gnm_score.min()) / r
                score_arrays.append(gnm_score)
                if "resids" in data and resids is None:
                    resids = data["resids"].astype(int)
            else:
                warnings.warn(
                    f"[topo] 'gnm_flucts_mean' not found in {gnm_path}",
                    stacklevel=2,
                )
        else:
            warnings.warn(
                f"[topo] anm_gnm_results.npz not found in {nma_pca_dir}",
                stacklevel=2,
            )

    # --- PCA per-residue displacement profiles ---
    if "pca" in fluctuation_sources:
        pca_path = os.path.join(nma_pca_dir, "pca_results.npz")
        if os.path.exists(pca_path):
            data = np.load(pca_path, allow_pickle=True)
            pc_score = np.zeros(
                len(data["resids"]) if "resids" in data else 0,
                dtype=np.float64,
            )
            for pc_key in ("pc1_profile", "pc2_profile"):
                if pc_key in data:
                    pc = data[pc_key].astype(np.float64)
                    r = pc.max() - pc.min()
                    if r > 0:
                        pc = (pc - pc.min()) / r
                    pc_score += pc
            if pc_score.size > 0:
                r = pc_score.max() - pc_score.min()
                if r > 0:
                    pc_score = (pc_score - pc_score.min()) / r
                score_arrays.append(pc_score)
            if "resids" in data and resids is None:
                resids = data["resids"].astype(int)
        else:
            warnings.warn(
                f"[topo] pca_results.npz not found in {nma_pca_dir}",
                stacklevel=2,
            )

    # --- Contact-map flexibility ---
    if "contact_map" in fluctuation_sources:
        cm_path = os.path.join(nma_pca_dir, "contact_map_results.npz")
        if os.path.exists(cm_path):
            data = np.load(cm_path, allow_pickle=True)
            if "per_residue_flexibility" in data:
                flex = data["per_residue_flexibility"].astype(np.float64)
                r = flex.max() - flex.min()
                if r > 0:
                    flex = (flex - flex.min()) / r
                score_arrays.append(flex)
                if "resids" in data and resids is None:
                    resids = data["resids"].astype(int)
        else:
            warnings.warn(
                f"[topo] contact_map_results.npz not found in {nma_pca_dir}",
                stacklevel=2,
            )

    if not score_arrays:
        return None, None

    # Sum all normalised component scores
    combined = np.sum(score_arrays, axis=0)
    return combined.astype(np.float64), resids


def identify_motion_residues(
    scores: np.ndarray,
    resids: np.ndarray,
    top_pct: float = 0.20,
    min_residues: int = 10,
) -> np.ndarray:
    """Select the top-motion residue numbers from a per-residue score array.

    Args:
        scores: Per-residue motion scores, shape [N_residues].
        resids: Corresponding residue numbers, shape [N_residues].
        top_pct: Fraction of residues to treat as high-motion (default 0.20 = top 20%).
        min_residues: Minimum number of motion residues regardless of top_pct.

    Returns:
        Integer array of residue numbers identified as high-motion.
    """
    n_select = max(min_residues, int(np.ceil(len(scores) * top_pct)))
    n_select = min(n_select, len(scores))
    top_idx = np.argsort(scores)[::-1][:n_select]
    return resids[top_idx]


# ---------------------------------------------------------------------------
# Cross-distance matrix (motion ↔ static, sentinel for intra pairs)
# ---------------------------------------------------------------------------

def generate_motion_cross_distance_matrix(
    motion_xyz: np.ndarray,
    static_xyz: np.ndarray,
    sentinel_cutoff: float = 1000.0,
) -> np.ndarray:
    """Build a distance matrix encoding only motion↔static cross-distances.

    Exactly mirrors the protein–ligand distance matrix in the original
    TopoFormer: intra-motion and intra-static pairs receive a large sentinel
    value (``sentinel_cutoff × 10000``) so they never appear in the persistent
    Laplacian filtration; only cross-pair (motion↔static) distances are real.

    Args:
        motion_xyz: Coordinates of motion-side atoms, shape [M, 3].
        static_xyz: Coordinates of static-side atoms, shape [S, 3].
        sentinel_cutoff: Multiplied by 10000 for the sentinel value.

    Returns:
        Symmetric distance matrix of shape [M+S, M+S].
    """
    M = len(motion_xyz)
    S = len(static_xyz)
    total = M + S
    sentinel = sentinel_cutoff * 10000.0

    dist_mat = np.full((total, total), sentinel, dtype=np.float64)
    np.fill_diagonal(dist_mat, 0.0)

    if M == 0 or S == 0:
        return dist_mat

    # Cross-distances (motion rows 0:M, static rows M:M+S)
    cross = cdist(motion_xyz, static_xyz, metric="euclidean")
    dist_mat[:M, M:] = cross
    dist_mat[M:, :M] = cross.T

    return dist_mat


# ---------------------------------------------------------------------------
# Per-conformation feature extraction
# ---------------------------------------------------------------------------

def _extract_single_conformation_features(
    pdb_file: str,
    motion_resids: np.ndarray,
    filtration: np.ndarray,
    motion_field: Optional[float],
    scl: SimplicialComplex_laplacian.SimplicialComplexLaplacian,
    sentinel_cutoff: float = 1000.0,
) -> np.ndarray:
    """Extract [6, n_filtrations, 121] features for one PDB conformation.

    Args:
        pdb_file: Path to one PDB conformation.
        motion_resids: Integer residue numbers identified as high-motion.
        filtration: Filtration parameter array (e.g. np.arange(0, 20, 0.1)).
        motion_field: If not None, restrict static atoms to within this many Å
            of any motion atom (analogous to consider_field in original code).
        scl: Reusable SimplicialComplexLaplacian instance.
        sentinel_cutoff: Passed to generate_motion_cross_distance_matrix.

    Returns:
        Feature array of shape [6, n_filtrations, 121] as float32.
    """
    n_filtrations  = len(filtration)
    feature_array  = np.zeros([NUM_STATISTICS, n_filtrations, NUM_COMBINATIONS],
                               dtype=np.float32)

    # Parse atoms
    all_ele, all_xyz, all_resnum = get_protein_atoms_with_resnum(pdb_file)
    if len(all_ele) == 0:
        return feature_array

    motion_mask = np.isin(all_resnum, motion_resids)
    static_mask = ~motion_mask

    motion_ele = all_ele[motion_mask]
    motion_xyz = all_xyz[motion_mask]
    static_ele = all_ele[static_mask]
    static_xyz = all_xyz[static_mask]

    # Optional proximity filter on static atoms
    if motion_field is not None and len(motion_xyz) > 0 and len(static_xyz) > 0:
        cross_d = cdist(static_xyz, motion_xyz).min(axis=1)
        nearby  = cross_d <= motion_field
        static_ele = static_ele[nearby]
        static_xyz = static_xyz[nearby]

    if len(motion_ele) == 0 or len(static_ele) == 0:
        return feature_array

    combo_idx = 0
    for m_combo in MOTION_SIDE_COMBINATIONS:
        # Select motion atoms matching this element combination
        m_mask = np.array([e in m_combo for e in motion_ele])
        sel_motion_xyz = motion_xyz[m_mask]

        for s_combo in STATIC_SIDE_COMBINATIONS:
            # Select static atoms matching this element combination
            s_mask = np.array([e in s_combo for e in static_ele])
            sel_static_xyz = static_xyz[s_mask]

            if len(sel_motion_xyz) == 0 or len(sel_static_xyz) == 0:
                combo_idx += 1
                continue

            dist_mat = generate_motion_cross_distance_matrix(
                sel_motion_xyz, sel_static_xyz, sentinel_cutoff
            )

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

            combo_idx += 1

    return feature_array


# ---------------------------------------------------------------------------
# Main ensemble feature extraction
# ---------------------------------------------------------------------------

def generate_ensemble_motion_lap_features(
    output_folder: str,
    protein_id: str,
    pdb_files: List[str],
    nma_pca_dir: str,
    output_feature_name: Optional[str] = None,
    top_motion_pct: float = 0.20,
    min_motion_residues: int = 10,
    fluctuation_sources: List[str] = None,
    n_slow_modes: int = 3,
    motion_field: Optional[float] = None,
    dis_start: float = 0.0,
    dis_cutoff: float = 20.0,
    dis_step: float = 0.1,
    ensemble_aggregation: str = "mean_std",
    print_progress: bool = True,
) -> np.ndarray:
    """Extract motion-guided ensemble topology features for a protein.

    Full pipeline:
      1. Load GNM/ANM/PCA per-residue motion scores from NMA-PCA outputs.
      2. Identify top-motion residue numbers.
      3. For each PDB conformation, compute cross-pair Persistent Laplacian
         features (motion atoms ↔ static atoms) → [6, n_filt, 121].
      4. Aggregate across conformations: mean + std → [12, n_filt, 121].
      5. Save as ``<output_folder>/<output_feature_name>.npy``.

    Args:
        output_folder: Directory to save the ``.npy`` output.
        protein_id: Identifier used for progress messages and default file name.
        pdb_files: List of PDB conformation file paths (length = ensemble size).
        nma_pca_dir: Directory with ``anm_gnm_results.npz`` / ``pca_results.npz``
            from the Ensemble_NMA-PCA pipeline.
        output_feature_name: Base name for the output file. Defaults to protein_id.
        top_motion_pct: Fraction of residues treated as high-motion (default 0.20).
        min_motion_residues: Minimum number of motion residues (default 10).
        fluctuation_sources: Profiles to combine. Any subset of
            ``['gnm', 'pca', 'contact_map']``. Default ``['gnm', 'pca']``.
        n_slow_modes: Number of GNM slow modes to sum (default 3).
        motion_field: Optional proximity cutoff (Å) to restrict static atoms.
            ``None`` (default) = use ALL static atoms.
        dis_start: Filtration start in Å (default 0.0).
        dis_cutoff: Filtration end in Å (default 20.0).
        dis_step: Filtration step in Å (default 0.1).
        ensemble_aggregation: How to combine conformations:
            - ``'mean_std'`` (default): stack mean+std → 12 channels
            - ``'mean_only'``: average only → 6 channels
            - ``'all'``: concatenate all conformations along channel axis
              → 6*N_conformers channels (use only for small ensembles)
        print_progress: Whether to print per-step progress.

    Returns:
        Feature array of shape [C, n_filtrations, 121] as float32, where C
        depends on ensemble_aggregation (12 for mean_std, 6 for mean_only).
        Also saved as ``<output_folder>/<output_feature_name>.npy``.
    """
    if fluctuation_sources is None:
        fluctuation_sources = ["gnm", "pca"]

    if not pdb_files:
        raise ValueError("pdb_files must be a non-empty list.")

    # ------------------------------------------------------------------
    # 1. Load motion scores and identify high-motion residues
    # ------------------------------------------------------------------
    scores, resids = load_motion_scores(nma_pca_dir, fluctuation_sources, n_slow_modes)

    if scores is None or resids is None:
        warnings.warn(
            f"[topo] No motion profiles found in {nma_pca_dir}. "
            "Falling back to ALL residues as 'motion' (reduces to intra-protein topology).",
            stacklevel=2,
        )
        # Fall back: parse first PDB to get all residue numbers
        _, _, all_rn = get_protein_atoms_with_resnum(pdb_files[0])
        motion_resids = np.unique(all_rn)
    else:
        motion_resids = identify_motion_residues(
            scores, resids, top_pct=top_motion_pct,
            min_residues=min_motion_residues
        )

    if print_progress:
        print(f"[topo] {protein_id}: {len(motion_resids)} motion residues "
              f"(top {top_motion_pct*100:.0f}% of {len(resids) if resids is not None else '?'})")

    # ------------------------------------------------------------------
    # 2. Filtration parameters
    # ------------------------------------------------------------------
    filtration = np.round(np.arange(dis_start, dis_cutoff, dis_step), 6)
    n_filtrations = len(filtration)

    # ------------------------------------------------------------------
    # 3. Per-conformation feature extraction
    # ------------------------------------------------------------------
    scl = SimplicialComplex_laplacian.SimplicialComplexLaplacian()
    per_conf_features = []

    for conf_idx, pdb_file in enumerate(pdb_files):
        if print_progress:
            print(f"[topo]   conformation {conf_idx+1}/{len(pdb_files)}: "
                  f"{os.path.basename(pdb_file)}")
        feat = _extract_single_conformation_features(
            pdb_file=pdb_file,
            motion_resids=motion_resids,
            filtration=filtration,
            motion_field=motion_field,
            scl=scl,
        )
        per_conf_features.append(feat)

    # per_conf_features: list of [6, n_filt, 121] arrays
    conf_stack = np.stack(per_conf_features, axis=0)  # [N_conf, 6, n_filt, 121]

    # ------------------------------------------------------------------
    # 4. Ensemble aggregation
    # ------------------------------------------------------------------
    if ensemble_aggregation == "mean_std":
        mean_feat = conf_stack.mean(axis=0)                 # [6, n_filt, 121]
        std_feat  = conf_stack.std(axis=0)                  # [6, n_filt, 121]
        output    = np.concatenate([mean_feat, std_feat], axis=0)  # [12, n_filt, 121]
    elif ensemble_aggregation == "mean_only":
        output = conf_stack.mean(axis=0)                    # [6, n_filt, 121]
    elif ensemble_aggregation == "all":
        # [N_conf, 6, n_filt, 121] → [N_conf*6, n_filt, 121]
        N = conf_stack.shape[0]
        output = conf_stack.reshape(N * NUM_STATISTICS, n_filtrations, NUM_COMBINATIONS)
    else:
        raise ValueError(f"Unknown ensemble_aggregation: {ensemble_aggregation!r}. "
                         "Choose 'mean_std', 'mean_only', or 'all'.")

    output = output.astype(np.float32)

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    fname = output_feature_name if output_feature_name else protein_id
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"{fname}.npy")
    np.save(out_path, output, allow_pickle=True)

    if print_progress:
        print(f"[topo] Saved {output.shape} → {out_path}")

    return output


# ---------------------------------------------------------------------------
# Convenience: discover ensemble PDB files
# ---------------------------------------------------------------------------

def find_ensemble_pdbs(
    pdb_dir: str,
    protein_id: str,
    n_conformers: int = 10,
) -> List[str]:
    """Return a sorted list of PDB conformation files for a protein.

    Searches for files matching common naming conventions:
      - ``<pdb_dir>/<protein_id>/cluster_repr_*.pdb``
      - ``<pdb_dir>/<protein_id>_conf_*.pdb``
      - ``<pdb_dir>/<protein_id>/*.pdb``
      - ``<pdb_dir>/<protein_id>_*.pdb``

    Args:
        pdb_dir: Root directory for PDB ensembles.
        protein_id: Protein identifier.
        n_conformers: Expected number of conformers (used for validation warning).

    Returns:
        Sorted list of PDB file paths. Empty list if nothing found.
    """
    patterns = [
        # aSAM ensemble output layout
        os.path.join(pdb_dir, protein_id, "clustering.nodes_pipeline_v1", "raw.seed_*.pdb"),
        # generic subdirectory layouts
        os.path.join(pdb_dir, protein_id, "cluster_repr_*.pdb"),
        os.path.join(pdb_dir, protein_id, f"{protein_id}_conf_*.pdb"),
        os.path.join(pdb_dir, protein_id, "*.pdb"),
        # flat layouts
        os.path.join(pdb_dir, f"{protein_id}_conf_*.pdb"),
        os.path.join(pdb_dir, f"{protein_id}_*.pdb"),
    ]
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            if len(files) != n_conformers:
                warnings.warn(
                    f"[topo] Found {len(files)} PDB files for {protein_id}, "
                    f"expected {n_conformers}.",
                    stacklevel=2,
                )
            return files
    return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Extract motion-guided ensemble Persistent Laplacian features.\n"
            "Output: [12, n_filtrations, 121] float32 numpy array (mean_std mode)."
        )
    )
    parser.add_argument("--protein_id",   required=True)
    parser.add_argument("--pdb_dir",      required=True,
                        help="Root directory containing ensemble PDB files.")
    parser.add_argument("--nma_pca_dir",  required=True,
                        help="Directory with anm_gnm_results.npz / pca_results.npz.")
    parser.add_argument("--output_folder", default="./topo_ensemble_features")
    parser.add_argument("--n_conformers",  type=int, default=10)
    parser.add_argument("--top_motion_pct", type=float, default=0.20)
    parser.add_argument("--min_motion_residues", type=int, default=10)
    parser.add_argument("--fluctuation_sources", nargs="+",
                        default=["gnm", "pca"],
                        choices=["gnm", "pca", "contact_map"])
    parser.add_argument("--n_slow_modes", type=int, default=3)
    parser.add_argument("--motion_field", type=float, default=None,
                        help="Restrict static atoms to within this Å of motion atoms. "
                             "Default None = use all static atoms.")
    parser.add_argument("--dis_start",   type=float, default=0.0)
    parser.add_argument("--dis_cutoff",  type=float, default=20.0)
    parser.add_argument("--dis_step",    type=float, default=0.1)
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

    generate_ensemble_motion_lap_features(
        output_folder=args.output_folder,
        protein_id=args.protein_id,
        pdb_files=pdb_files,
        nma_pca_dir=args.nma_pca_dir,
        top_motion_pct=args.top_motion_pct,
        min_motion_residues=args.min_motion_residues,
        fluctuation_sources=args.fluctuation_sources,
        n_slow_modes=args.n_slow_modes,
        motion_field=args.motion_field,
        dis_start=args.dis_start,
        dis_cutoff=args.dis_cutoff,
        dis_step=args.dis_step,
        ensemble_aggregation=args.ensemble_aggregation,
        print_progress=True,
    )
    print(f"[topo] Finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
