"""DETR-style Transformer decoder for modulation connection prediction.

Predicts a set of modulation connections using learned query embeddings
and cross-attention to the AST encoder's patch sequence. Each query
outputs: exists (is this slot active?), source, destination, amount,
bipolar, power, stereo.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModulationDecoder(nn.Module):
    """DETR-style set prediction decoder for modulation connections.

    Uses learned query embeddings that attend to the encoder's patch
    sequence via cross-attention, then predict per-query outputs.

    Args:
        n_queries: Number of learned query slots (max predicted connections).
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        n_layers: Number of transformer decoder layers.
        encoder_dim: Dimension of the encoder's patch features.
        n_sources: Number of modulation source classes.
        n_destinations: Number of modulation destination classes.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_queries: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        d_ff: int = 512,
        n_layers: int = 3,
        encoder_dim: int = 768,
        n_sources: int = 32,
        n_destinations: int = 406,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.d_model = d_model
        self.n_sources = n_sources
        self.n_destinations = n_destinations

        # Learned query embeddings
        self.query_embed = nn.Embedding(n_queries, d_model)

        # Project encoder features to decoder dimension
        self.encoder_proj = nn.Linear(encoder_dim, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Per-query output heads
        self.exists_head = nn.Linear(d_model, 1)
        self.source_head = nn.Linear(d_model, n_sources)
        self.dest_head = nn.Linear(d_model, n_destinations)
        self.amount_head = nn.Linear(d_model, 1)
        self.bipolar_head = nn.Linear(d_model, 1)
        self.power_head = nn.Linear(d_model, 1)
        self.stereo_head = nn.Linear(d_model, 1)

    def forward(
        self,
        patch_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict modulation connections from encoder patch features.

        Args:
            patch_features: (B, N, encoder_dim) patch sequence from AST.

        Returns:
            Dict with keys:
                "exists":  (B, Q) existence logits
                "source":  (B, Q, n_sources) source logits
                "dest":    (B, Q, n_destinations) destination logits
                "amount":  (B, Q) amount predictions (tanh -> [-1, 1])
                "bipolar": (B, Q) bipolar logits
                "power":   (B, Q) power predictions (unconstrained)
                "stereo":  (B, Q) stereo logits
        """
        B = patch_features.shape[0]

        # Project encoder features to decoder dimension
        memory = self.encoder_proj(patch_features)  # (B, N, d_model)

        # Expand queries for the batch
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, d_model)

        # Decode: queries attend to encoder memory
        decoded = self.decoder(queries, memory)  # (B, Q, d_model)

        # Per-query predictions
        exists = self.exists_head(decoded).squeeze(-1)  # (B, Q)
        source = self.source_head(decoded)  # (B, Q, n_sources)
        dest = self.dest_head(decoded)  # (B, Q, n_destinations)
        amount = torch.tanh(self.amount_head(decoded).squeeze(-1))  # (B, Q)
        bipolar = self.bipolar_head(decoded).squeeze(-1)  # (B, Q)
        power = self.power_head(decoded).squeeze(-1)  # (B, Q)
        stereo = self.stereo_head(decoded).squeeze(-1)  # (B, Q)

        return {
            "exists": exists,
            "source": source,
            "dest": dest,
            "amount": amount,
            "bipolar": bipolar,
            "power": power,
            "stereo": stereo,
        }
