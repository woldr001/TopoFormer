"""Dataset and data-loading utilities for protein MF GO term prediction.

Each sample consists of three pre-computed feature files:
    - Topology:  <topo_dir>/<protein_id>.npy  → shape [6, 200, 15]  (float32)
    - ESM:       <esm_dir>/<protein_id>.npy   → shape [1152]        (per-residue
                 mean-pooled ESM-C 600M embeddings, already aggregated)
    - ProtTrans: <pt_dir>/<protein_id>.npy    → shape [1024]        (mean-pooled
                 ProtT5 embeddings)

and a label CSV with one row per protein and one column per MF GO term (0/1).

Topology features are scaled with a pre-fitted sklearn Pipeline
(StandardScaler + MinMaxScaler), following the same procedure as
``scaler_for_image_like_data()`` in ``topt_regression_finetuning.py``.

Typical usage
-------------
    from protein_function.data.protein_function_dataset import (
        build_mf_label_matrix,
        fit_topo_scaler,
        ProteinFunctionDataset,
        collate_fn,
    )

    # 1. Build label matrix from a UniProt-style annotation file
    label_df, go_terms = build_mf_label_matrix("mf_annotations.tsv",
                                               min_annotation_count=50)

    # 2. Fit scaler on training topology features
    train_ids = open("train_ids.txt").read().split()
    scaler = fit_topo_scaler(train_ids, topo_dir="./topo_features")

    # 3. Create datasets
    train_ds = ProteinFunctionDataset(
        protein_ids=train_ids,
        topo_dir="./topo_features",
        esm_dir="./esm_features",
        prottrans_dir="./prottrans_features",
        label_df=label_df,
        go_terms=go_terms,
        topo_scaler=scaler,
    )
    loader = DataLoader(train_ds, batch_size=32, collate_fn=collate_fn, shuffle=True)
"""

import os
import pickle
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ---------------------------------------------------------------------------
# Label matrix construction
# ---------------------------------------------------------------------------

def build_mf_label_matrix(
    annotation_file: str,
    min_annotation_count: int = 50,
    sep: str = "\t",
    protein_col: str = "protein_id",
    go_col: str = "go_term",
    namespace_col: Optional[str] = "namespace",
    namespace_value: str = "molecular_function",
) -> Tuple[pd.DataFrame, List[str]]:
    """Build a binary label matrix from a protein→GO annotation file.

    Expected input format (TSV with header):
        protein_id  go_term           namespace
        P12345      GO:0003677        molecular_function
        P12345      GO:0046872        molecular_function
        Q99999      GO:0003824        molecular_function

    Alternatively, a wide CSV with rows = protein_id and columns = GO terms is
    also accepted (detected automatically when ``go_col`` is not present).

    Args:
        annotation_file: Path to annotation TSV/CSV.
        min_annotation_count: Minimum number of proteins annotated with a term
            for the term to be included. Rare terms are dropped.
        sep: Field separator (default '\\t').
        protein_col: Column name for protein IDs.
        go_col: Column name for GO term IDs (long format).
        namespace_col: Column name for GO namespace. Set to None to skip filtering.
        namespace_value: Namespace to retain (default 'molecular_function').

    Returns:
        (label_df, go_terms) where label_df is a DataFrame indexed by protein_id
        with binary columns for each retained MF GO term, and go_terms is the
        ordered list of GO term column names.
    """
    df = pd.read_csv(annotation_file, sep=sep)

    # Detect wide vs. long format
    if go_col not in df.columns:
        # Assume wide format: index column = protein_id, other columns = GO terms
        if protein_col in df.columns:
            df = df.set_index(protein_col)
        # Filter to MF terms if namespace column exists
        go_terms_all = list(df.columns)
    else:
        # Long format: pivot to wide
        if namespace_col and namespace_col in df.columns:
            df = df[df[namespace_col] == namespace_value]
        df[go_col] = df[go_col].str.strip()
        label_df = (
            df.groupby([protein_col, go_col])
            .size()
            .unstack(fill_value=0)
            .clip(upper=1)
        )
        go_terms_all = list(label_df.columns)
        df = label_df

    # Drop terms with fewer than min_annotation_count annotations
    term_counts = df.sum(axis=0)
    valid_terms = term_counts[term_counts >= min_annotation_count].index.tolist()
    df = df[valid_terms]

    print(
        f"[data] Loaded {len(df)} proteins, "
        f"{len(valid_terms)} MF GO terms (≥{min_annotation_count} annotations)."
    )
    return df.astype(np.float32), valid_terms


# ---------------------------------------------------------------------------
# Topology feature scaling
# ---------------------------------------------------------------------------

def fit_topo_scaler(
    protein_ids: List[str],
    topo_dir: str,
    save_path: Optional[str] = None,
) -> Pipeline:
    """Fit a StandardScaler + MinMaxScaler pipeline on training topology features.

    Follows the pattern in ``scaler_for_image_like_data()`` in
    ``topt_regression_finetuning.py``:
        flatten [N, 6, 200, 15] → [N, 18000] → fit → save.

    Args:
        protein_ids: List of protein IDs whose features are used for fitting.
        topo_dir: Directory containing ``<protein_id>.npy`` files.
        save_path: If given, the fitted scaler is pickled to this path.

    Returns:
        Fitted sklearn Pipeline.
    """
    features = []
    missing = []
    for pid in protein_ids:
        path = os.path.join(topo_dir, f"{pid}.npy")
        if not os.path.exists(path):
            missing.append(pid)
            continue
        feat = np.load(path, allow_pickle=True).astype(np.float32)
        features.append(feat.ravel())

    if missing:
        warnings.warn(f"[data] Missing topology files for {len(missing)} proteins.")

    features = np.stack(features, axis=0)  # [N, 18000]
    scaler = Pipeline([("std", StandardScaler()), ("minmax", MinMaxScaler())])
    scaler.fit(features)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"[data] Scaler saved to {save_path}")

    return scaler


def load_topo_scaler(scaler_path: str) -> Pipeline:
    """Load a pickled topology scaler."""
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def apply_topo_scaler(
    feature: np.ndarray,
    scaler: Pipeline,
) -> np.ndarray:
    """Apply scaler to a single topology feature array.

    Args:
        feature: Array of shape [6, n_filtrations, n_combinations].
        scaler: Fitted sklearn Pipeline.

    Returns:
        Scaled array of the same shape.
    """
    original_shape = feature.shape
    scaled = scaler.transform(feature.ravel().reshape(1, -1))
    return scaled.reshape(original_shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProteinFunctionDataset(Dataset):
    """PyTorch Dataset for protein MF GO term prediction.

    Each ``__getitem__`` call loads and returns pre-computed topology, ESM, and
    ProtTrans features for one protein, together with its MF GO term labels.

    Args:
        protein_ids: List of protein identifiers.
        topo_dir: Directory with ``<protein_id>.npy`` topology files [6,200,15].
        esm_dir: Directory with ``<protein_id>.npy`` ESM embedding files [1152].
        prottrans_dir: Directory with ``<protein_id>.npy`` ProtTrans files [1024].
        label_df: DataFrame (index=protein_id, columns=GO terms) of 0/1 labels.
            Missing proteins are treated as all-zero label vectors.
        go_terms: Ordered list of GO term column names to use.
        topo_scaler: Fitted scaler Pipeline, or None to skip scaling.
        topo_feature_shape: Expected shape of raw topology features.
    """

    def __init__(
        self,
        protein_ids: List[str],
        topo_dir: str,
        esm_dir: str,
        prottrans_dir: str,
        label_df: pd.DataFrame,
        go_terms: List[str],
        topo_scaler: Optional[Pipeline] = None,
        topo_feature_shape: Tuple[int, int, int] = (12, 200, 121),
    ):
        self.protein_ids = protein_ids
        self.topo_dir = topo_dir
        self.esm_dir = esm_dir
        self.prottrans_dir = prottrans_dir
        self.label_df = label_df
        self.go_terms = go_terms
        self.topo_scaler = topo_scaler
        self.topo_feature_shape = topo_feature_shape
        self.num_labels = len(go_terms)

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid = self.protein_ids[idx]

        # --- Topology features ---
        topo_path = os.path.join(self.topo_dir, f"{pid}.npy")
        if os.path.exists(topo_path):
            topo = np.load(topo_path, allow_pickle=True).astype(np.float32)
            if self.topo_scaler is not None:
                topo = apply_topo_scaler(topo, self.topo_scaler)
        else:
            warnings.warn(f"[data] Missing topology file for {pid}, using zeros.")
            topo = np.zeros(self.topo_feature_shape, dtype=np.float32)

        # --- ESM embeddings ---
        esm_path = os.path.join(self.esm_dir, f"{pid}.npy")
        if os.path.exists(esm_path):
            esm = np.load(esm_path, allow_pickle=True).astype(np.float32)
            # Support both pre-pooled [D] and per-residue [L, D]
            if esm.ndim == 2:
                esm = esm.mean(axis=0)
        else:
            warnings.warn(f"[data] Missing ESM file for {pid}, using zeros.")
            esm = np.zeros(1152, dtype=np.float32)

        # --- ProtTrans embeddings ---
        pt_path = os.path.join(self.prottrans_dir, f"{pid}.npy")
        if os.path.exists(pt_path):
            prot = np.load(pt_path, allow_pickle=True).astype(np.float32)
            if prot.ndim == 2:
                prot = prot.mean(axis=0)
        else:
            warnings.warn(f"[data] Missing ProtTrans file for {pid}, using zeros.")
            prot = np.zeros(1024, dtype=np.float32)

        # --- Labels ---
        if pid in self.label_df.index:
            label = self.label_df.loc[pid, self.go_terms].values.astype(np.float32)
        else:
            label = np.zeros(self.num_labels, dtype=np.float32)

        return {
            "protein_id": pid,
            "topo_features": torch.from_numpy(topo),
            "esm_features": torch.from_numpy(esm),
            "prottrans_features": torch.from_numpy(prot),
            "labels": torch.from_numpy(label),
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    """Stack a list of samples into a batch dict.

    All feature arrays are fixed size after pre-processing, so simple stacking works.
    The ``protein_id`` field is returned as a list of strings (not stacked).
    """
    protein_ids = [item["protein_id"] for item in batch]
    topo = torch.stack([item["topo_features"] for item in batch])
    esm  = torch.stack([item["esm_features"] for item in batch])
    prot = torch.stack([item["prottrans_features"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "protein_ids": protein_ids,
        "topo_features": topo,
        "esm_features": esm,
        "prottrans_features": prot,
        "labels": labels,
    }
