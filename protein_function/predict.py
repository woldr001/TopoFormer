"""End-to-end protein Molecular Function GO-term prediction.

Given a protein PDB file (and optionally pre-computed sequence embeddings),
this script runs the full pipeline:
  1. Extract protein-only topological features from the PDB file.
  2. Load (or compute) ESM and ProtTrans sequence embeddings.
  3. Run the trained ``TopoFunctionModel`` to predict MF GO terms.
  4. Print/return the top predictions with confidence scores.

Usage (command line)
--------------------
    python protein_function/predict.py \\
        --model_dir   ./mf_model \\
        --protein_pdb /data/proteins/P12345.pdb \\
        --sequence    MKTAYIAK...  \\   # optional, read from PDB SEQRES if absent
        --threshold   0.5

    # If ESM/ProtTrans are already pre-computed:
    python protein_function/predict.py \\
        --model_dir   ./mf_model \\
        --protein_pdb /data/proteins/P12345.pdb \\
        --esm_npy     /data/esm_features/P12345.npy \\
        --prottrans_npy /data/prottrans_features/P12345.npy

Usage (Python API)
------------------
    from protein_function.predict import load_model, predict_function

    model, go_terms, scaler = load_model("./mf_model")
    predictions = predict_function(
        model=model,
        go_terms=go_terms,
        topo_scaler=scaler,
        protein_pdb="P12345.pdb",
        sequence="MKTAYIAK...",
        threshold=0.5,
    )
    # predictions: list of (go_term, probability) sorted by probability descending
"""

import argparse
import os
import sys
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from protein_function.topo_extraction.protein_only_topo_embedding import (
    generate_protein_lap_features,
)
from protein_function.data.protein_function_dataset import (
    apply_topo_scaler,
    load_topo_scaler,
)
from protein_function.models.mini_topt_config import get_mini_topt_config
from protein_function.models.modeling_topo_function import TopoFunctionModel


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_dir: str,
    device: str = None,
    topo_channels: int = 6,
    topo_n_filtrations: int = 200,
    topo_n_combinations: int = 15,
) -> Tuple[TopoFunctionModel, List[str], object]:
    """Load a trained model, GO term list, and topology scaler from a directory.

    The directory should contain:
        - ``pytorch_model.bin`` or ``model.pt``  (model weights)
        - ``go_terms.txt``                       (one GO term per line)
        - ``topo_scaler.pkl``                    (fitted sklearn scaler)
        - ``config.json`` (optional)             (model configuration)

    Args:
        model_dir: Path to the directory saved by the training script.
        device: Torch device string. Defaults to 'cuda' if available.
        topo_channels: Topology feature channels the model was trained with.
            6 for protein_only; 12 for ensemble_motion and sidechain_centroid.
        topo_n_filtrations: Number of filtration steps (topology height).
        topo_n_combinations: Number of element/type combinations (topology width).
            15 for protein_only and sidechain_centroid; 121 for ensemble_motion.

    Returns:
        (model, go_terms, topo_scaler) tuple.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load GO terms
    go_terms_path = os.path.join(model_dir, "go_terms.txt")
    if not os.path.exists(go_terms_path):
        raise FileNotFoundError(f"go_terms.txt not found in {model_dir}")
    with open(go_terms_path) as f:
        go_terms = [l.strip() for l in f if l.strip()]

    # Load scaler
    scaler_path = os.path.join(model_dir, "topo_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"topo_scaler.pkl not found in {model_dir}")
    topo_scaler = load_topo_scaler(scaler_path)

    # Reconstruct model
    num_mf_labels = len(go_terms)
    topt_config = get_mini_topt_config(
        num_channels=topo_channels,
        image_size=(topo_n_filtrations, topo_n_combinations),
        patch_size=(1, topo_n_combinations),
    )
    model = TopoFunctionModel(topt_config=topt_config, num_mf_labels=num_mf_labels)

    # Load weights
    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found in {model_dir}")

    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model, go_terms, topo_scaler


# ---------------------------------------------------------------------------
# On-the-fly sequence embedding
# ---------------------------------------------------------------------------

def _compute_esm_embedding(sequence: str, device: str) -> np.ndarray:
    """Compute ESM-2 embedding for a single sequence. Returns [1280] array."""
    from transformers import AutoTokenizer, EsmModel
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm_model  = EsmModel.from_pretrained(model_name).to(device).eval()
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors="pt", truncation=True,
                           max_length=1024).to(device)
        hidden = esm_model(**inputs).last_hidden_state  # [1, L, D]
        mask   = inputs["attention_mask"][:, 1:-1].unsqueeze(-1).float()
        emb    = (hidden[:, 1:-1] * mask).sum(1) / mask.sum(1)
    del esm_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return emb.squeeze(0).cpu().float().numpy()


def _compute_prottrans_embedding(sequence: str, device: str) -> np.ndarray:
    """Compute ProtTrans T5 embedding for a single sequence. Returns [1024] array."""
    import re
    from transformers import T5Tokenizer, T5EncoderModel
    model_name = "Rostlab/prot_t5_xl_uniref50"
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    pt_model   = T5EncoderModel.from_pretrained(model_name).to(device).eval()
    spaced = " ".join(re.sub(r"[UZOB]", "X", sequence))
    with torch.no_grad():
        inputs = tokenizer(spaced, return_tensors="pt", truncation=True,
                           max_length=1024).to(device)
        hidden = pt_model(**inputs).last_hidden_state   # [1, L, 1024]
        mask   = inputs["attention_mask"].unsqueeze(-1).float()
        emb    = (hidden * mask).sum(1) / mask.sum(1)
    del pt_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return emb.squeeze(0).cpu().float().numpy()


# ---------------------------------------------------------------------------
# Core predict function
# ---------------------------------------------------------------------------

def predict_function(
    model: TopoFunctionModel,
    go_terms: List[str],
    topo_scaler,
    protein_pdb: str,
    sequence: Optional[str] = None,
    esm_npy: Optional[str] = None,
    prottrans_npy: Optional[str] = None,
    threshold: float = 0.5,
    dis_start: float = 0.0,
    dis_cutoff: float = 20.0,
    dis_step: float = 0.1,
    device: str = None,
) -> List[Tuple[str, float]]:
    """Predict MF GO terms for a single protein.

    Args:
        model: Loaded ``TopoFunctionModel`` (in eval mode).
        go_terms: Ordered list of GO term identifiers.
        topo_scaler: Fitted topology feature scaler.
        protein_pdb: Path to the protein PDB file.
        sequence: Amino acid sequence string. Required if ESM/ProtTrans are not
                  pre-computed and will be computed on the fly.
        esm_npy: Path to pre-computed ESM embedding .npy file [1152].
                  If None, ESM is computed on the fly (requires transformers).
        prottrans_npy: Path to pre-computed ProtTrans .npy file [1024].
        threshold: Probability threshold for positive predictions (default 0.5).
        dis_start, dis_cutoff, dis_step: Filtration parameters (must match
            those used during training).
        device: Torch device ('cuda' or 'cpu').

    Returns:
        List of (go_term, probability) tuples for terms above threshold,
        sorted by probability descending.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Topology features
    with tempfile.TemporaryDirectory() as tmpdir:
        topo_feat = generate_protein_lap_features(
            output_folder=tmpdir,
            protein_file=protein_pdb,
            dis_start=dis_start,
            dis_cutoff=dis_cutoff,
            dis_step=dis_step,
            print_progress=False,
        )

    topo_feat = apply_topo_scaler(topo_feat, topo_scaler)
    topo_tensor = torch.from_numpy(topo_feat).unsqueeze(0).to(device)  # [1, 6, 200, 15]

    # 2. ESM embedding
    if esm_npy and os.path.exists(esm_npy):
        esm_arr = np.load(esm_npy, allow_pickle=True).astype(np.float32)
        if esm_arr.ndim == 2:
            esm_arr = esm_arr.mean(axis=0)
    elif sequence:
        print("[predict] Computing ESM embedding (this may take a minute)...")
        esm_arr = _compute_esm_embedding(sequence, device)
    else:
        raise ValueError(
            "Either --esm_npy or --sequence must be provided to compute ESM embedding."
        )
    esm_tensor = torch.from_numpy(esm_arr).unsqueeze(0).to(device)  # [1, D_esm]

    # 3. ProtTrans embedding
    if prottrans_npy and os.path.exists(prottrans_npy):
        pt_arr = np.load(prottrans_npy, allow_pickle=True).astype(np.float32)
        if pt_arr.ndim == 2:
            pt_arr = pt_arr.mean(axis=0)
    elif sequence:
        print("[predict] Computing ProtTrans embedding (this may take a minute)...")
        pt_arr = _compute_prottrans_embedding(sequence, device)
    else:
        raise ValueError(
            "Either --prottrans_npy or --sequence must be provided."
        )
    pt_tensor = torch.from_numpy(pt_arr).unsqueeze(0).to(device)  # [1, 1024]

    # 4. Model inference
    with torch.no_grad():
        probs = model.predict_proba(topo_tensor, esm_tensor, pt_tensor)  # [1, N_mf]
    probs = probs.squeeze(0).cpu().numpy()  # [N_mf]

    # 5. Filter and sort
    results = [
        (go_term, float(prob))
        for go_term, prob in zip(go_terms, probs)
        if prob >= threshold
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict protein Molecular Function GO terms from a PDB file."
    )
    parser.add_argument("--model_dir", required=True,
                        help="Directory with trained model (pytorch_model.bin, go_terms.txt, topo_scaler.pkl).")
    parser.add_argument("--protein_pdb", required=True,
                        help="Path to the protein PDB file.")
    parser.add_argument("--sequence", default=None,
                        help="Amino acid sequence (used to compute ESM/ProtTrans if .npy not provided).")
    parser.add_argument("--esm_npy", default=None,
                        help="Pre-computed ESM embedding .npy file.")
    parser.add_argument("--prottrans_npy", default=None,
                        help="Pre-computed ProtTrans embedding .npy file.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for positive predictions (default 0.5).")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Show only top-k predictions regardless of threshold.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model, go_terms, topo_scaler = load_model(args.model_dir, device=args.device)

    predictions = predict_function(
        model=model,
        go_terms=go_terms,
        topo_scaler=topo_scaler,
        protein_pdb=args.protein_pdb,
        sequence=args.sequence,
        esm_npy=args.esm_npy,
        prottrans_npy=args.prottrans_npy,
        threshold=args.threshold,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    if args.top_k:
        predictions = predictions[: args.top_k]

    print(f"\nPredicted MF GO terms for {os.path.basename(args.protein_pdb)}:")
    print(f"{'GO Term':<20}  {'Probability':>12}")
    print("-" * 35)
    for go_term, prob in predictions:
        print(f"{go_term:<20}  {prob:>12.4f}")
    if not predictions:
        print("  (no terms above threshold)")


if __name__ == "__main__":
    main()
