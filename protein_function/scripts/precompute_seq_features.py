"""Batch pre-computation of ESM and ProtTrans sequence embeddings.

Computes mean-pooled per-residue embeddings for each protein and saves them as
``<protein_id>.npy`` files:
    - ESM:       shape [1152]  (ESM-C 600M, ``facebook/esm2_t33_650M_UR50D``)
    - ProtTrans: shape [1024]  (ProtT5, ``Rostlab/prot_t5_xl_uniref50``)

The embeddings are computed independently (ESM first, ProtTrans second) to avoid
holding both large models in GPU memory simultaneously.

Usage
-----
    # Both ESM and ProtTrans
    python protein_function/scripts/precompute_seq_features.py \\
        --fasta_file /data/sequences.fasta \\
        --esm_output_dir    /data/esm_features \\
        --prottrans_output_dir /data/prottrans_features \\
        --batch_size 4 \\
        --device cuda

    # ESM only
    python protein_function/scripts/precompute_seq_features.py \\
        --fasta_file /data/sequences.fasta \\
        --esm_output_dir /data/esm_features \\
        --skip_prottrans

FASTA format expected:
    >protein_id
    MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL

Dependencies:
    pip install transformers torch biopython
    pip install fair-esm  # for ESM models (optional, HuggingFace ESM-2 also works)
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def parse_fasta(fasta_file: str) -> list:
    """Return list of (protein_id, sequence) tuples from a FASTA file."""
    proteins = []
    current_id, current_seq = None, []
    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    proteins.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        proteins.append((current_id, "".join(current_seq)))
    return proteins


# ---------------------------------------------------------------------------
# ESM embeddings
# ---------------------------------------------------------------------------

def compute_esm_embeddings(
    proteins: list,
    output_dir: str,
    batch_size: int = 4,
    device: str = "cuda",
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    overwrite: bool = False,
) -> None:
    """Compute and save mean-pooled ESM-2 embeddings.

    Uses HuggingFace ``transformers`` ESM-2 model.
    Output per protein: ``<protein_id>.npy`` of shape [1280] for esm2_t33_650M
    or [1152] for ESM-C 600M (adjust ``model_name`` accordingly).

    Args:
        proteins: List of (protein_id, sequence) tuples.
        output_dir: Directory for output .npy files.
        batch_size: Number of sequences per GPU batch.
        device: 'cuda' or 'cpu'.
        model_name: HuggingFace model identifier.
        overwrite: Re-compute even if output file exists.
    """
    from transformers import AutoTokenizer, EsmModel

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading ESM model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    to_process = [
        (pid, seq) for pid, seq in proteins
        if overwrite or not os.path.exists(os.path.join(output_dir, f"{pid}.npy"))
    ]
    logger.info(f"Computing ESM embeddings for {len(to_process)} proteins ...")

    with torch.no_grad():
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i : i + batch_size]
            ids, seqs = zip(*batch)

            inputs = tokenizer(
                list(seqs),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            # last_hidden_state: [B, seq_len+2, hidden_dim]  (+2 for CLS/EOS tokens)
            hidden = outputs.last_hidden_state  # [B, L, D]
            # Mean-pool over residue positions (exclude CLS/EOS)
            attention_mask = inputs["attention_mask"][:, 1:-1].unsqueeze(-1).float()
            residue_hidden  = hidden[:, 1:-1, :]   # strip CLS and EOS
            mean_emb = (residue_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

            for pid, emb in zip(ids, mean_emb):
                out_path = os.path.join(output_dir, f"{pid}.npy")
                np.save(out_path, emb.cpu().float().numpy())

            if (i // batch_size) % 10 == 0:
                logger.info(f"  ESM: {i + len(batch)}/{len(to_process)}")

    # Free GPU memory before loading ProtTrans
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    logger.info("ESM embeddings done.")


# ---------------------------------------------------------------------------
# ProtTrans embeddings
# ---------------------------------------------------------------------------

def compute_prottrans_embeddings(
    proteins: list,
    output_dir: str,
    batch_size: int = 2,
    device: str = "cuda",
    model_name: str = "Rostlab/prot_t5_xl_uniref50",
    overwrite: bool = False,
) -> None:
    """Compute and save mean-pooled ProtTrans T5 embeddings.

    Output per protein: ``<protein_id>.npy`` of shape [1024].

    Args:
        proteins: List of (protein_id, sequence) tuples.
        output_dir: Directory for output .npy files.
        batch_size: Sequences per batch (T5 is large; use 1-4 on 16 GB GPU).
        device: 'cuda' or 'cpu'.
        model_name: HuggingFace model identifier.
        overwrite: Re-compute even if output file exists.
    """
    from transformers import T5Tokenizer, T5EncoderModel
    import re

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading ProtTrans model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    to_process = [
        (pid, seq) for pid, seq in proteins
        if overwrite or not os.path.exists(os.path.join(output_dir, f"{pid}.npy"))
    ]
    logger.info(f"Computing ProtTrans embeddings for {len(to_process)} proteins ...")

    with torch.no_grad():
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i : i + batch_size]
            ids, seqs = zip(*batch)

            # ProtTrans T5 expects space-separated amino acids; replace rare chars
            spaced = [" ".join(re.sub(r"[UZOB]", "X", seq)) for seq in seqs]

            inputs = tokenizer(
                spaced,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            hidden = outputs.last_hidden_state  # [B, L, 1024]
            mask   = inputs["attention_mask"].unsqueeze(-1).float()
            mean_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

            for pid, emb in zip(ids, mean_emb):
                out_path = os.path.join(output_dir, f"{pid}.npy")
                np.save(out_path, emb.cpu().float().numpy())

            if (i // batch_size) % 10 == 0:
                logger.info(f"  ProtTrans: {i + len(batch)}/{len(to_process)}")

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    logger.info("ProtTrans embeddings done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute ESM and ProtTrans sequence embeddings."
    )
    parser.add_argument("--fasta_file", required=True,
                        help="FASTA file with protein sequences.")
    parser.add_argument("--esm_output_dir", default=None,
                        help="Output directory for ESM embeddings.")
    parser.add_argument("--prottrans_output_dir", default=None,
                        help="Output directory for ProtTrans embeddings.")
    parser.add_argument("--esm_model",
                        default="facebook/esm2_t33_650M_UR50D",
                        help="HuggingFace ESM model ID.")
    parser.add_argument("--prottrans_model",
                        default="Rostlab/prot_t5_xl_uniref50",
                        help="HuggingFace ProtTrans model ID.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_esm", action="store_true")
    parser.add_argument("--skip_prottrans", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    proteins = parse_fasta(args.fasta_file)
    logger.info(f"Loaded {len(proteins)} sequences from {args.fasta_file}")

    if not args.skip_esm:
        if args.esm_output_dir is None:
            parser.error("--esm_output_dir is required unless --skip_esm is set.")
        compute_esm_embeddings(
            proteins=proteins,
            output_dir=args.esm_output_dir,
            batch_size=args.batch_size,
            device=args.device,
            model_name=args.esm_model,
            overwrite=args.overwrite,
        )

    if not args.skip_prottrans:
        if args.prottrans_output_dir is None:
            parser.error("--prottrans_output_dir is required unless --skip_prottrans is set.")
        pt_batch = max(1, args.batch_size // 2)  # T5 needs more memory
        compute_prottrans_embeddings(
            proteins=proteins,
            output_dir=args.prottrans_output_dir,
            batch_size=pt_batch,
            device=args.device,
            model_name=args.prottrans_model,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
