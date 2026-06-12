"""Configuration factories for the mini TopoT encoder used in protein function prediction.

Two feature modes are supported:

**protein_only** (``get_mini_topt_config``):
    Input [B, 6, 200, 15] — single-structure protein-only topology.
    - 6 spectral statistics (channels)
    - 200 filtration steps (0–20 Å)
    - 15 element combinations (protein C/N/O/S singles/pairs/triples/all)

**ensemble_motion** (``get_ensemble_motion_topt_config``):
    Input [B, 12, 200, 121] — motion-guided ensemble topology.
    - 12 channels (6 stats × mean + 6 stats × std across 10 conformations)
    - 200 filtration steps (0–20 Å)
    - 121 element combinations (11 motion-side × 11 static-side cross-pairs)

**sidechain_centroid** (``get_sidechain_topt_config``):
    Input [B, 12, 200, 15] — side-chain-centroid ensemble topology.
    - 12 channels (6 stats × mean + 6 stats × std across 10 conformations)
    - 200 filtration steps (0–40 Å at 0.2 Å)
    - 15 side-chain-type combinations (flat list over 4 chemical classes)

All encoders use hidden_size=256, 4 layers (appropriate for GO annotation
dataset scale of 30k–100k proteins; a full 86M-parameter ViT would overfit).
"""

import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from code_pkg.top_transformer.configuration_topt import TopTConfig


def get_mini_topt_config(
    num_channels: int = 6,
    image_size: tuple = (200, 15),
    patch_size: tuple = (1, 15),
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: int = 1024,
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    mask_ratio: float = 0.0,
    pooler_type: str = "cls_token",
    **overrides,
) -> TopTConfig:
    """Return a TopTConfig sized for protein-only [6, 200, 15] features.

    Args:
        num_channels: Number of spectral statistic channels (default 6).
        image_size: (height, width) = (n_filtrations, n_combinations).
        patch_size: (patch_h, patch_w); (1, 15) gives one token per filtration step.
        hidden_size: Transformer hidden dimension (default 256, ~3M params).
        num_hidden_layers: Number of transformer encoder layers (default 4).
        num_attention_heads: Number of attention heads (default 4).
        intermediate_size: Feed-forward intermediate dimension (default 1024).
        hidden_dropout_prob: Dropout on hidden layers.
        attention_probs_dropout_prob: Dropout on attention weights.
        mask_ratio: Masking ratio (0.0 = disabled during inference/fine-tuning).
        pooler_type: How to pool encoder output ('cls_token' or 'avg_token').
        **overrides: Any additional TopTConfig keyword arguments.

    Returns:
        A ``TopTConfig`` instance ready for ``TopTModel(config)``.
    """
    return TopTConfig(
        num_channels=num_channels,
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        mask_ratio=mask_ratio,
        pooler_type=pooler_type,
        **overrides,
    )


def get_ensemble_motion_topt_config(
    num_channels: int = 12,
    image_size: tuple = (200, 121),
    patch_size: tuple = (1, 121),
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: int = 1024,
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    mask_ratio: float = 0.0,
    pooler_type: str = "cls_token",
    **overrides,
) -> TopTConfig:
    """Return a TopTConfig sized for motion-guided ensemble [12, 200, 121] features.

    The 12-channel input encodes ensemble information:
      - Channels 0-5:  mean topology feature across 10 conformations
      - Channels 6-11: std  topology feature across 10 conformations

    The 121-combination width comes from 11 motion-side × 11 static-side element
    cross-pairs (C/N/O/S singles, pairs, and all-heavy), mirroring the protein–ligand
    cross-pair scheme of the original TopoFormer.

    Args:
        num_channels: Spectral statistic channels × ensemble stats (default 12).
        image_size: (height, width) = (n_filtrations, n_combinations).
        patch_size: (1, 121) gives one token per filtration step → 200 tokens.
        hidden_size: Transformer hidden dimension (default 256).
        num_hidden_layers: Number of transformer encoder layers (default 4).
        num_attention_heads: Number of attention heads (default 4).
        intermediate_size: Feed-forward intermediate dimension (default 1024).
        hidden_dropout_prob: Dropout on hidden layers.
        attention_probs_dropout_prob: Dropout on attention weights.
        mask_ratio: 0.0 disables masking during fine-tuning.
        pooler_type: 'cls_token' or 'avg_token'.
        **overrides: Any additional TopTConfig keyword arguments.

    Returns:
        A ``TopTConfig`` instance ready for ``TopTModel(config)``.
    """
    return TopTConfig(
        num_channels=num_channels,
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        mask_ratio=mask_ratio,
        pooler_type=pooler_type,
        **overrides,
    )


def get_sidechain_topt_config(
    num_channels: int = 12,
    image_size: tuple = (200, 15),
    patch_size: tuple = (1, 15),
    hidden_size: int = 256,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    intermediate_size: int = 1024,
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    mask_ratio: float = 0.0,
    pooler_type: str = "cls_token",
    **overrides,
) -> TopTConfig:
    """Return a TopTConfig sized for side-chain-centroid ensemble [12, 200, 15] features.

    The 12-channel input encodes ensemble information:
      - Channels 0-5:  mean topology feature across 10 conformations
      - Channels 6-11: std  topology feature across 10 conformations

    The 15-combination width comes from the flat list of side-chain-type subsets
    (4 singles + 6 pairs + 4 triples + 1 quadruple) over the four chemical classes
    (nonpolar, aromatic, polar-uncharged, polar-charged).  Each residue is reduced
    to the geometric centroid of its side-chain heavy atoms (glycine → Cα).

    Args:
        num_channels: Spectral statistic channels × ensemble stats (default 12).
        image_size: (height, width) = (n_filtrations, n_combinations) = (200, 15).
        patch_size: (1, 15) gives one token per filtration step → 200 tokens.
        hidden_size: Transformer hidden dimension (default 256).
        num_hidden_layers: Number of transformer encoder layers (default 4).
        num_attention_heads: Number of attention heads (default 4).
        intermediate_size: Feed-forward intermediate dimension (default 1024).
        hidden_dropout_prob: Dropout on hidden layers.
        attention_probs_dropout_prob: Dropout on attention weights.
        mask_ratio: 0.0 disables masking during fine-tuning.
        pooler_type: 'cls_token' or 'avg_token'.
        **overrides: Any additional TopTConfig keyword arguments.

    Returns:
        A ``TopTConfig`` instance ready for ``TopTModel(config)``.
    """
    return TopTConfig(
        num_channels=num_channels,
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        mask_ratio=mask_ratio,
        pooler_type=pooler_type,
        **overrides,
    )


def load_pretrained_topt_weights(
    mini_topo_encoder,
    pretrained_model_path: str,
    strict: bool = False,
) -> list:
    """Partially transfer pretrained TopoFormer encoder weights to the mini encoder.

    The patch-embedding Conv2d and positional embeddings are incompatible between
    the original (hidden_size=768, patch=(1,429)) and mini (hidden_size=256,
    patch=(1,15)) configs, so those layers are re-initialized.  Compatible
    transformer attention/dense/LayerNorm weights are sliced to fit the smaller
    hidden dimension.

    Args:
        mini_topo_encoder: A ``TopTModel`` instance with the mini config.
        pretrained_model_path: Path to directory containing ``pytorch_model.bin``
            (or a ``config.json`` + ``pytorch_model.bin`` from ``save_pretrained``).
        strict: Whether to raise on any missing/unexpected key. Defaults to False.

    Returns:
        List of weight keys that were skipped (incompatible shapes).
    """
    import torch

    pretrained_bin = os.path.join(pretrained_model_path, "pytorch_model.bin")
    if not os.path.exists(pretrained_bin):
        raise FileNotFoundError(
            f"No pytorch_model.bin found at {pretrained_model_path}. "
            "Download a pretrained TopoFormer checkpoint first."
        )

    pretrained_state = torch.load(pretrained_bin, map_location="cpu")
    mini_state = mini_topo_encoder.state_dict()
    mini_hidden = mini_topo_encoder.config.hidden_size

    skipped = []
    new_state = {}

    for key, mini_val in mini_state.items():
        if key not in pretrained_state:
            skipped.append(key)
            continue

        pre_val = pretrained_state[key]

        if pre_val.shape == mini_val.shape:
            new_state[key] = pre_val
        elif pre_val.ndim == mini_val.ndim:
            # Slice each dimension to the mini size
            try:
                slices = tuple(slice(0, s) for s in mini_val.shape)
                new_state[key] = pre_val[slices].contiguous()
            except Exception:
                skipped.append(key)
        else:
            skipped.append(key)

    missing, unexpected = mini_topo_encoder.load_state_dict(new_state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Strict weight loading failed.\nMissing: {missing}\nUnexpected: {unexpected}"
        )

    print(f"[weight-transfer] Loaded {len(new_state)} keys, skipped {len(skipped)}: {skipped}")
    return skipped
