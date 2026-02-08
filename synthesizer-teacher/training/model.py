"""Model architectures for Vital inverse synthesis.

V1: ResNet-18 backbone with continuous + categorical + modulation prediction heads.
V2: AST encoder with DETR modulation decoder, dedicated wavetable/LFO heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ModulationHead(nn.Module):
    """Factored bilinear head for sparse modulation matrix prediction.

    Produces a (B, 4, n_sources, n_destinations) tensor representing the
    modulation routing matrix. The 4 channels are: amount, bipolar, power, stereo.

    Uses source/destination embeddings + bilinear scoring to keep parameter
    count manageable (~7M params vs 512*4*32*406 = 26M for a naive linear).
    """

    def __init__(
        self,
        feature_dim: int = 512,
        n_sources: int = 32,
        n_destinations: int = 406,
        embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_src = n_sources
        self.n_dst = n_destinations
        self.embed_dim = embed_dim

        self.source_proj = nn.Linear(feature_dim, n_sources * embed_dim)
        self.dest_proj = nn.Linear(feature_dim, n_destinations * embed_dim)
        # 4 bilinear matrices: amount, bipolar, power, stereo
        self.channel_weights = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
            for _ in range(4)
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: (B, feature_dim) backbone features.

        Returns:
            (B, 4, n_sources, n_destinations) modulation matrix.
        """
        B = features.shape[0]
        src = self.source_proj(features).view(B, self.n_src, self.embed_dim)
        dst = self.dest_proj(features).view(B, self.n_dst, self.embed_dim)
        channels = []
        for W in self.channel_weights:
            transformed = torch.matmul(src, W)  # (B, n_src, embed_dim)
            ch = torch.bmm(transformed, dst.transpose(1, 2))  # (B, n_src, n_dst)
            channels.append(ch)
        return torch.stack(channels, dim=1)  # (B, 4, n_src, n_dst)


class VitalInverseModel(nn.Module):
    """Predicts Vital synth parameters from a log-mel spectrogram.

    Architecture:
        ResNet-18 backbone (conv1 adapted for 1-channel input, pretrained
        ImageNet weights averaged across RGB channels). Early layers
        (conv1 through layer2) optionally frozen.

        Continuous head: shared MLP -> sigmoid output in [0,1].
        Categorical heads: one small MLP per categorical param, outputting
        logits with n_options classes each.
        Modulation head (optional): factored bilinear -> (B, 4, 32, 406).

    Args:
        n_continuous: Number of continuous parameters to predict.
        categorical_n_options: List of class counts for each categorical param.
        mlp_hidden: Hidden dimension for MLP heads.
        dropout: Dropout rate in heads.
        freeze_early: If True, freeze conv1 through layer2.
        simple_categorical_heads: If True, use single linear layer instead of MLP.
        n_mod_sources: Number of modulation sources (0 = no modulation head).
        n_mod_destinations: Number of modulation destinations (0 = no modulation head).
        mod_embed_dim: Embedding dimension for modulation head.
    """

    def __init__(
        self,
        n_continuous: int,
        categorical_n_options: list[int],
        mlp_hidden: int = 512,
        dropout: float = 0.1,
        freeze_early: bool = True,
        simple_categorical_heads: bool = True,
        n_mod_sources: int = 0,
        n_mod_destinations: int = 0,
        mod_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.n_continuous = n_continuous
        self.categorical_n_options = categorical_n_options
        self.n_mod_sources = n_mod_sources
        self.n_mod_destinations = n_mod_destinations

        # --- Backbone ---
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Adapt conv1 from 3ch -> 1ch by averaging pretrained weights
        old_weight = backbone.conv1.weight.data  # (64, 3, 7, 7)
        new_weight = old_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False,
        )
        backbone.conv1.weight.data = new_weight

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        if freeze_early:
            for module in [self.conv1, self.bn1, self.layer1, self.layer2]:
                for param in module.parameters():
                    param.requires_grad = False

        # --- Continuous head ---
        self.continuous_head = nn.Sequential(
            nn.Linear(512, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_continuous),
            nn.Sigmoid(),
        )

        # --- Categorical heads (one per param) ---
        self.categorical_heads = nn.ModuleList()
        for n_opts in categorical_n_options:
            if simple_categorical_heads:
                head = nn.Linear(512, n_opts)
            else:
                head = nn.Sequential(
                    nn.Linear(512, mlp_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_hidden, n_opts),
                )
            self.categorical_heads.append(head)

        # --- Modulation head (optional, tier-3 only) ---
        self.modulation_head: ModulationHead | None = None
        if n_mod_sources > 0 and n_mod_destinations > 0:
            self.modulation_head = ModulationHead(
                feature_dim=512,
                n_sources=n_mod_sources,
                n_destinations=n_mod_destinations,
                embed_dim=mod_embed_dim,
            )

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
        """Forward pass.

        Args:
            x: (B, 1, n_mels, time) log-mel spectrogram.

        Returns:
            continuous_pred: (B, n_continuous) in [0, 1].
            categorical_logits: list of (B, n_options_i) tensors.
            modulation_pred: (B, 4, n_sources, n_destinations) or None.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)  # (B, 512)

        continuous_pred = self.continuous_head(features)
        categorical_logits = [head(features) for head in self.categorical_heads]

        modulation_pred = None
        if self.modulation_head is not None:
            modulation_pred = self.modulation_head(features)

        return continuous_pred, categorical_logits, modulation_pred


class VitalInverseModelV2(nn.Module):
    """V2 architecture: AST encoder + DETR modulation decoder.

    Improvements over V1:
        - AST encoder (768-dim, AudioSet pretrained) replaces ResNet-18 (512-dim)
        - Dedicated wavetable MLP heads per oscillator (separate from generic categoricals)
        - LFO shape prediction heads
        - DETR-style set prediction for modulation connections
        - Wider MLP heads (1024 hidden) to match richer 768-dim features

    Args:
        n_continuous: Number of continuous parameters to predict.
        categorical_n_options: Class counts for each categorical param (excl. wavetables).
        n_wavetables: Number of wavetable options per oscillator.
        n_lfo_shapes: Number of LFO shape classes.
        n_mod_queries: Number of modulation query slots (max connections).
        n_mod_sources: Number of modulation source classes.
        n_mod_destinations: Number of modulation destination classes.
        mlp_hidden: Hidden dimension for continuous MLP head.
        dropout: Dropout rate in heads.
        encoder_pretrained: HuggingFace model ID for AST.
        freeze_strategy: Initial encoder freeze strategy ("all", "last2", "none").
    """

    def __init__(
        self,
        n_continuous: int,
        categorical_n_options: list[int],
        n_wavetables: int = 70,
        n_lfo_shapes: int = 8,
        n_mod_queries: int = 20,
        n_mod_sources: int = 32,
        n_mod_destinations: int = 406,
        mlp_hidden: int = 1024,
        dropout: float = 0.1,
        encoder_pretrained: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        freeze_strategy: str = "last2",
    ) -> None:
        super().__init__()
        self.n_continuous = n_continuous
        self.categorical_n_options = categorical_n_options
        self.n_wavetables = n_wavetables
        self.n_lfo_shapes = n_lfo_shapes
        self.n_mod_queries = n_mod_queries
        self.encoder_type = "ast"

        from training.encoders import ASTEncoder

        # --- AST Encoder ---
        self.encoder = ASTEncoder(
            pretrained=encoder_pretrained,
            freeze_strategy=freeze_strategy,
        )
        enc_dim = ASTEncoder.FEATURE_DIM  # 768

        # --- Continuous head (wider: 1024 hidden) ---
        self.continuous_head = nn.Sequential(
            nn.Linear(enc_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_continuous),
            nn.Sigmoid(),
        )

        # --- Generic categorical heads (one per param, excl. wavetables) ---
        self.categorical_heads = nn.ModuleList()
        for n_opts in categorical_n_options:
            self.categorical_heads.append(nn.Linear(enc_dim, n_opts))

        # --- Dedicated wavetable heads (3 oscillators) ---
        self.wavetable_heads = nn.ModuleList()
        for _ in range(3):
            self.wavetable_heads.append(nn.Sequential(
                nn.Linear(enc_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, n_wavetables),
            ))

        # --- LFO shape heads (8 LFOs) ---
        self.lfo_shape_heads = nn.ModuleList()
        for _ in range(8):
            self.lfo_shape_heads.append(nn.Linear(enc_dim, n_lfo_shapes))

        # --- DETR Modulation Decoder ---
        self.modulation_decoder = None
        if n_mod_sources > 0 and n_mod_destinations > 0 and n_mod_queries > 0:
            from training.modulation_decoder import ModulationDecoder

            self.modulation_decoder = ModulationDecoder(
                n_queries=n_mod_queries,
                d_model=256,
                nhead=8,
                d_ff=512,
                n_layers=3,
                encoder_dim=enc_dim,
                n_sources=n_mod_sources,
                n_destinations=n_mod_destinations,
                dropout=dropout,
            )

    def forward(
        self, x: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor] | None]:
        """Forward pass.

        Args:
            x: (B, 1, n_mels, T) log-mel spectrogram.

        Returns:
            Dict with keys:
                "continuous": (B, n_continuous) in [0, 1]
                "categorical": list of (B, n_options_i) logits
                "wavetable": list of 3x (B, n_wavetables) logits
                "lfo_shapes": list of 8x (B, n_lfo_shapes) logits
                "modulation": dict from ModulationDecoder or None
        """
        # Encode
        cls_token, patch_sequence = self.encoder(x)

        # Scalar predictions from CLS token
        continuous_pred = self.continuous_head(cls_token)
        categorical_logits = [head(cls_token) for head in self.categorical_heads]
        wavetable_logits = [head(cls_token) for head in self.wavetable_heads]
        lfo_shape_logits = [head(cls_token) for head in self.lfo_shape_heads]

        # Modulation from patch sequence via DETR decoder
        modulation = None
        if self.modulation_decoder is not None:
            modulation = self.modulation_decoder(patch_sequence)

        return {
            "continuous": continuous_pred,
            "categorical": categorical_logits,
            "wavetable": wavetable_logits,
            "lfo_shapes": lfo_shape_logits,
            "modulation": modulation,
        }
