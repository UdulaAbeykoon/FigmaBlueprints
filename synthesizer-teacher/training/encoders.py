"""AST (Audio Spectrogram Transformer) encoder wrapper for VitalInverseModelV2.

Loads a pretrained AST checkpoint (MIT/ast-finetuned-audioset-10-10-0.4593),
adapts the patch embedding for variable-length 128-bin log-mel spectrograms,
and exposes both the CLS token (B, 768) and patch sequence (B, N, 768)
for downstream scalar heads and cross-attention decoders.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class ASTEncoder(nn.Module):
    """Audio Spectrogram Transformer encoder.

    Wraps the ViT backbone from a pretrained AST checkpoint with a custom
    patch embedding that accepts variable-length 128-bin log-mel spectrograms.

    The original AST uses a fixed 128x1024 input (10.24s at 100 frames/s).
    We replace the patch embedding to accept (1, 128, T) for any T, and
    interpolate positional embeddings to match.

    Args:
        pretrained: HuggingFace model ID or local path for AST checkpoint.
        freeze_strategy: One of "all", "last2", "none".
            "all"   -- freeze entire encoder (Phase 1 warmup)
            "last2" -- freeze all except last 2 transformer blocks (Phase 2)
            "none"  -- all layers trainable (Phase 3)
        n_mels: Number of mel frequency bins (must match training data).
        patch_size: Patch size for the custom embedding (freq, time).
        patch_stride: Patch stride for the custom embedding (freq, time).
    """

    FEATURE_DIM = 768  # AST hidden dimension

    def __init__(
        self,
        pretrained: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        freeze_strategy: str = "last2",
        n_mels: int = 128,
        patch_size: tuple[int, int] = (16, 16),
        patch_stride: tuple[int, int] = (10, 10),
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Load pretrained AST
        from transformers import ASTModel

        self._ast = ASTModel.from_pretrained(pretrained)
        config = self._ast.config

        # Replace patch embedding with custom one that handles variable T
        self.patch_embed = nn.Conv2d(
            1,
            config.hidden_size,
            kernel_size=patch_size,
            stride=patch_stride,
        )

        # Copy pretrained patch embedding weights if shapes match
        pretrained_proj = self._ast.embeddings.patch_embeddings.projection
        if pretrained_proj.weight.shape == self.patch_embed.weight.shape:
            self.patch_embed.weight.data.copy_(pretrained_proj.weight.data)
            self.patch_embed.bias.data.copy_(pretrained_proj.bias.data)
            log.info("Copied pretrained patch embedding weights")
        else:
            log.info(
                "Patch embedding shape mismatch (pretrained %s vs custom %s); "
                "using random init",
                pretrained_proj.weight.shape,
                self.patch_embed.weight.shape,
            )

        # CLS and distillation tokens from pretrained model
        self.cls_token = self._ast.embeddings.cls_token  # (1, 1, 768)
        self.dist_token = self._ast.embeddings.distillation_token  # (1, 1, 768)

        # Store pretrained positional embeddings for interpolation
        # Shape: (1, n_pretrained_patches + 2, 768) where +2 = cls + dist tokens
        self._pretrained_pos_embed = nn.Parameter(
            self._ast.embeddings.position_embeddings.data.clone(),
            requires_grad=False,
        )

        # Learnable positional embedding (will be resized per forward pass)
        self.pos_embed_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Layer norm (from pretrained)
        self.layernorm = self._ast.layernorm

        # Transformer encoder blocks
        self.encoder = self._ast.encoder

        # Remove the original embeddings module to save memory
        del self._ast.embeddings
        # Keep _ast reference for config access only
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_hidden_layers

        # Apply freeze strategy
        self.set_freeze_strategy(freeze_strategy)

    def set_freeze_strategy(self, strategy: str) -> None:
        """Update which layers are frozen.

        Args:
            strategy: "all", "last2", or "none".
        """
        self._freeze_strategy = strategy

        if strategy == "all":
            for param in self.parameters():
                param.requires_grad = False
            # Keep patch_embed and pos_embed_proj trainable even when "all"
            # since they are new/adapted
            for param in self.patch_embed.parameters():
                param.requires_grad = True
            for param in self.pos_embed_proj.parameters():
                param.requires_grad = True
            log.info("AST encoder: frozen all (except patch embed + pos proj)")

        elif strategy == "last2":
            # Freeze everything first
            for param in self.parameters():
                param.requires_grad = False
            # Unfreeze patch embed, pos proj
            for param in self.patch_embed.parameters():
                param.requires_grad = True
            for param in self.pos_embed_proj.parameters():
                param.requires_grad = True
            # Unfreeze last 2 transformer blocks
            n_layers = len(self.encoder.layer)
            for layer in self.encoder.layer[n_layers - 2 :]:
                for param in layer.parameters():
                    param.requires_grad = True
            # Unfreeze layernorm
            for param in self.layernorm.parameters():
                param.requires_grad = True
            log.info(
                "AST encoder: frozen except last 2/%d blocks + patch embed",
                n_layers,
            )

        elif strategy == "none":
            for param in self.parameters():
                param.requires_grad = True
            log.info("AST encoder: all layers unfrozen")

        else:
            raise ValueError(f"Unknown freeze_strategy: {strategy!r}")

    def _interpolate_pos_embed(self, n_patches: int) -> torch.Tensor:
        """Interpolate pretrained positional embeddings to match n_patches.

        The pretrained AST has pos embeddings for a fixed number of patches.
        We interpolate to handle variable-length input.

        Args:
            n_patches: Number of patches in current input (excluding cls/dist).

        Returns:
            (1, n_patches + 2, hidden_size) positional embeddings.
        """
        # Split cls+dist tokens from patch positions
        cls_dist_pos = self._pretrained_pos_embed[:, :2, :]  # (1, 2, D)
        patch_pos = self._pretrained_pos_embed[:, 2:, :]  # (1, N_pretrained, D)

        n_pretrained = patch_pos.shape[1]

        if n_patches == n_pretrained:
            return self._pretrained_pos_embed

        # 1D interpolation: reshape to (1, D, N) -> interpolate -> (1, D, n_patches) -> (1, n_patches, D)
        patch_pos = patch_pos.transpose(1, 2)  # (1, D, N)
        patch_pos = F.interpolate(
            patch_pos,
            size=n_patches,
            mode="linear",
            align_corners=False,
        )
        patch_pos = patch_pos.transpose(1, 2)  # (1, n_patches, D)

        return torch.cat([cls_dist_pos, patch_pos], dim=1)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a log-mel spectrogram.

        Args:
            x: (B, 1, n_mels, T) log-mel spectrogram.

        Returns:
            cls_token: (B, 768) CLS token features for scalar heads.
            patch_sequence: (B, N, 768) patch sequence for cross-attention.
        """
        B = x.shape[0]

        # Patch embedding: (B, 1, n_mels, T) -> (B, D, H', W') -> (B, N, D)
        patches = self.patch_embed(x)  # (B, D, H', W')
        H_out, W_out = patches.shape[2], patches.shape[3]
        n_patches = H_out * W_out
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, D)

        # Prepend CLS and distillation tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, dist_tokens, patches], dim=1)  # (B, N+2, D)

        # Add interpolated positional embeddings
        pos_embed = self._interpolate_pos_embed(n_patches)
        pos_embed = pos_embed.to(tokens.device, dtype=tokens.dtype)
        tokens = tokens + pos_embed

        # Pass through transformer encoder
        encoder_output = self.encoder(tokens, return_dict=True)
        hidden_states = encoder_output.last_hidden_state  # (B, N+2, D)

        # Apply layer norm
        hidden_states = self.layernorm(hidden_states)

        # Split CLS token and patch sequence
        cls_out = hidden_states[:, 0]  # (B, D) - CLS token
        patch_out = hidden_states[:, 2:]  # (B, N, D) - skip dist token

        return cls_out, patch_out
