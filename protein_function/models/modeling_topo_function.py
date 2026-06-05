"""Combined topology + sequence model for protein Molecular Function (MF) GO term prediction.

Architecture
------------
- Topology branch: mini TopTModel encoder
    Input:  [B, 6, 200, 15]  (6 spectral stats × 200 filtration steps × 15 element combos)
    Output: CLS token → [B, 256]
- Sequence branch: two separate linear projectors for ESM and ProtTrans embeddings
    ESM input:       [B, 1152] (mean-pooled ESM-C 600M per-residue embeddings)
    ProtTrans input: [B, 1024] (mean-pooled ProtT5 per-residue embeddings)
    Output:          concatenated → [B, 1024]  (two × 512-dim projections)
- Fusion MLP:  [B, 1280] → FC(1280, 512) → GELU → Dropout → FC(512, N_mf)
- Loss: BCEWithLogitsLoss (multi-label classification)

All three input tensors are required at training time; at inference ``labels``
may be None and the returned loss will also be None.
"""

import sys
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from code_pkg.top_transformer.modeling_topt import TopTModel
from code_pkg.top_transformer.configuration_topt import TopTConfig
from .mini_topt_config import get_mini_topt_config


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TopoFunctionOutput:
    """Output from :class:`TopoFunctionModel`.

    Attributes:
        loss: BCEWithLogitsLoss scalar if ``labels`` was provided, else None.
        logits: Raw (pre-sigmoid) predictions of shape [B, num_mf_labels].
        topo_emb: CLS-pooled topology embedding [B, topo_hidden_size].
        seq_emb: Concatenated projected sequence embedding [B, seq_proj_dim*2].
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    topo_emb: Optional[torch.FloatTensor] = None
    seq_emb: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# Sequence projector
# ---------------------------------------------------------------------------

class SequenceFeatureProjector(nn.Module):
    """Projects mean-pooled ESM and ProtTrans embeddings to a shared dimension.

    Args:
        esm_dim: Dimensionality of ESM embeddings (default 1152 for ESM-C 600M).
        prottrans_dim: Dimensionality of ProtTrans T5 embeddings (default 1024).
        proj_dim: Output projection dimension per embedding (default 512).
        dropout: Dropout probability after projection (default 0.1).
    """

    def __init__(
        self,
        esm_dim: int = 1280,
        prottrans_dim: int = 1024,
        proj_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(prottrans_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        esm_features: torch.Tensor,        # [B, esm_dim]
        prottrans_features: torch.Tensor,  # [B, prottrans_dim]
    ) -> torch.Tensor:                     # [B, proj_dim * 2]
        esm_emb  = self.esm_proj(esm_features)
        prot_emb = self.prot_proj(prottrans_features)
        return torch.cat([esm_emb, prot_emb], dim=-1)


# ---------------------------------------------------------------------------
# Fusion MLP
# ---------------------------------------------------------------------------

class FusionMLP(nn.Module):
    """Two-layer MLP that maps the fused embedding to GO term logits.

    Args:
        input_dim: Dimension of the concatenated topo + sequence embedding.
        hidden_dim: Hidden layer dimension (default 512).
        num_labels: Number of MF GO terms to predict.
        dropout: Dropout probability (default 0.3).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_labels: int = 490,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TopoFunctionModel(nn.Module):
    """Protein Molecular Function GO-term predictor combining topology and sequences.

    Args:
        topt_config: :class:`TopTConfig` for the mini topology encoder. Use
            :func:`~protein_function.models.mini_topt_config.get_mini_topt_config`
            for the default protein-only configuration.
        num_mf_labels: Number of MF GO terms to predict (determined from data).
        esm_dim: ESM embedding dimension (default 1280 for ESM-2 650M).
        prottrans_dim: ProtTrans T5 embedding dimension (default 1024).
        seq_proj_dim: Projection dimension per sequence model (default 512).
        fusion_hidden_dim: Hidden dimension of the fusion MLP (default 512).
        topo_dropout: Dropout in the topology encoder (applied via config).
        fusion_dropout: Dropout in the fusion MLP (default 0.3).

    Inputs to ``forward()``:
        topo_features:      [B, 12, 200, 121] protein topology features (scaled)
        esm_features:       [B, esm_dim]     mean-pooled ESM embeddings
        prottrans_features: [B, prottrans_dim] mean-pooled ProtTrans embeddings
        labels:             [B, num_mf_labels] float multi-hot labels (optional)

    Returns:
        :class:`TopoFunctionOutput` with ``loss``, ``logits``, ``topo_emb``, ``seq_emb``.
    """

    def __init__(
        self,
        topt_config: TopTConfig = None,
        num_mf_labels: int = 490,
        esm_dim: int = 1280,
        prottrans_dim: int = 1024,
        seq_proj_dim: int = 512,
        fusion_hidden_dim: int = 512,
        topo_dropout: float = 0.1,
        fusion_dropout: float = 0.3,
    ):
        super().__init__()

        if topt_config is None:
            topt_config = get_mini_topt_config(
                hidden_dropout_prob=topo_dropout,
                attention_probs_dropout_prob=topo_dropout,
            )
        self.topt_config = topt_config

        # Topology branch
        self.topo_encoder = TopTModel(topt_config)

        # Sequence branch
        self.seq_projector = SequenceFeatureProjector(
            esm_dim=esm_dim,
            prottrans_dim=prottrans_dim,
            proj_dim=seq_proj_dim,
            dropout=topo_dropout,
        )

        # Fusion
        topo_dim = topt_config.hidden_size               # 256
        seq_dim  = seq_proj_dim * 2                      # 1024
        self.fusion = FusionMLP(
            input_dim=topo_dim + seq_dim,                # 1280
            hidden_dim=fusion_hidden_dim,
            num_labels=num_mf_labels,
            dropout=fusion_dropout,
        )

        self.num_mf_labels = num_mf_labels

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        topo_features: torch.Tensor,
        esm_features: torch.Tensor,
        prottrans_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> TopoFunctionOutput:
        """Forward pass.

        Args:
            topo_features:      Float tensor [B, 6, 200, 15].
            esm_features:       Float tensor [B, esm_dim].
            prottrans_features: Float tensor [B, prottrans_dim].
            labels:             Float tensor [B, num_mf_labels] with 0/1 values,
                                or None during inference.

        Returns:
            :class:`TopoFunctionOutput`.
        """
        # --- Topology branch ---
        # TopTModel forward: (topological_features, noise=None)
        # During fine-tuning mask_ratio is 0, so no masking occurs.
        topo_out = self.topo_encoder(topo_features)
        # last_hidden_state: [B, n_tokens+1, hidden_size]
        # CLS token is at position 0
        topo_emb = topo_out.last_hidden_state[:, 0, :]   # [B, 256]

        # --- Sequence branch ---
        seq_emb = self.seq_projector(esm_features, prottrans_features)  # [B, 1024]

        # --- Fusion ---
        fused  = torch.cat([topo_emb, seq_emb], dim=-1)  # [B, 1280]
        logits = self.fusion(fused)                        # [B, num_mf_labels]

        # --- Loss ---
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        return TopoFunctionOutput(
            loss=loss,
            logits=logits,
            topo_emb=topo_emb,
            seq_emb=seq_emb,
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        topo_features: torch.Tensor,
        esm_features: torch.Tensor,
        prottrans_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return sigmoid-transformed probabilities [B, num_mf_labels]."""
        with torch.no_grad():
            out = self.forward(topo_features, esm_features, prottrans_features)
        return torch.sigmoid(out.logits)

    def freeze_topo_encoder(self):
        """Freeze topology encoder weights (useful for initial sequence-only warm-up)."""
        for p in self.topo_encoder.parameters():
            p.requires_grad = False

    def unfreeze_topo_encoder(self):
        """Unfreeze topology encoder weights."""
        for p in self.topo_encoder.parameters():
            p.requires_grad = True

    def freeze_seq_projector(self):
        """Freeze sequence projector weights."""
        for p in self.seq_projector.parameters():
            p.requires_grad = False

    def unfreeze_seq_projector(self):
        """Unfreeze sequence projector weights."""
        for p in self.seq_projector.parameters():
            p.requires_grad = True
