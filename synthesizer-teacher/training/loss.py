"""Loss functions for V1 and V2 architectures.

V1: importance-weighted MSE + per-param cross-entropy with label smoothing.
V2: adds ModulationSetLoss (Hungarian matching), LFOShapeLoss, separate wavetable loss.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# Prefixes for continuous params that are always unlearnable at Tier 1-2
# (no modulation routing means LFOs, random generators, and envelopes 3-6
# have zero effect on audio output)
_STATIC_MASK_PREFIXES = (
    "lfo_1_", "lfo_2_", "lfo_3_", "lfo_4_",
    "lfo_5_", "lfo_6_", "lfo_7_", "lfo_8_",
    "random_1_", "random_2_", "random_3_", "random_4_",
    "env_3_", "env_4_", "env_5_", "env_6_",
)

# Mapping from module-on categorical param prefix to the continuous param
# prefix it gates.  E.g. "chorus_on" gates all "chorus_*" continuous params.
_MODULE_ON_SUFFIXES = (
    "chorus", "compressor", "delay", "distortion",
    "eq", "flanger", "phaser", "reverb",
    "filter_1", "filter_2", "filter_fx",
)


class ModulationLoss(nn.Module):
    """Sparsity-aware loss for modulation matrix prediction.

    The modulation matrix is (B, 4, n_sources, n_destinations) with channels:
        0: amount [-1, 1]
        1: bipolar {0, 1}
        2: power [-10, 10]
        3: stereo {0, 1}

    Loss components:
        - BCE with pos_weight for connection presence (|amount| > threshold)
        - MSE on amount/power for active connections only
        - BCE on bipolar/stereo for active connections only
    """

    def __init__(self, pos_weight: float = 20.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute modulation loss.

        Args:
            pred: (B, 4, n_src, n_dst) predicted modulation matrix.
            target: (B, 4, n_src, n_dst) target modulation matrix.

        Returns:
            Scalar loss tensor.
        """
        target_amount = target[:, 0]  # (B, n_src, n_dst)
        active = (target_amount.abs() > 1e-6).float()

        # Connection presence loss (BCE on predicted amount magnitude)
        pred_presence = pred[:, 0].abs()
        pw = torch.tensor(self.pos_weight, device=pred.device)
        presence_loss = F.binary_cross_entropy_with_logits(
            pred_presence, active, pos_weight=pw,
        )

        # Active-connection param losses (only where target has connections)
        n_active = active.sum().clamp(min=1)

        # Amount MSE (channel 0)
        amount_loss = ((pred[:, 0] - target[:, 0]) ** 2 * active).sum() / n_active

        # Power MSE (channel 2)
        power_loss = ((pred[:, 2] - target[:, 2]) ** 2 * active).sum() / n_active

        # Bipolar BCE (channel 1)
        bipolar_loss = F.binary_cross_entropy_with_logits(
            pred[:, 1], target[:, 1], reduction="none",
        )
        bipolar_loss = (bipolar_loss * active).sum() / n_active

        # Stereo BCE (channel 3)
        stereo_loss = F.binary_cross_entropy_with_logits(
            pred[:, 3], target[:, 3], reduction="none",
        )
        stereo_loss = (stereo_loss * active).sum() / n_active

        return presence_loss + amount_loss + power_loss + bipolar_loss + stereo_loss


class VitalLoss(nn.Module):
    """Combined loss for continuous and categorical parameter prediction.

    L_total = continuous_weight * L_cont + categorical_weight * L_cat

    L_cont = mean(importance_weights * (pred - target)^2)
    L_cat  = mean([CE(logits_i, target_i, label_smoothing) for i in categoricals])

    Args:
        importance_weights: (n_continuous,) float tensor. If None, uniform.
        n_continuous: Number of continuous params (used if weights is None).
        continuous_weight: Multiplier for continuous loss.
        categorical_weight: Multiplier for categorical loss.
        label_smoothing: Label smoothing factor for cross-entropy (helps with class imbalance).
        continuous_names: List of continuous param names (for conditional masking).
        categorical_names: List of categorical param names (for conditional masking).
        conditional_mask: If True, mask loss for unlearnable params.
    """

    def __init__(
        self,
        importance_weights: torch.Tensor | None = None,
        n_continuous: int = 0,
        continuous_weight: float = 1.0,
        categorical_weight: float = 0.5,
        label_smoothing: float = 0.0,
        continuous_names: list[str] | None = None,
        categorical_names: list[str] | None = None,
        conditional_mask: bool = False,
        tier: int = 1,
        modulation_loss_weight: float = 0.0,
        modulation_pos_weight: float = 20.0,
        modulation_warmup_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.continuous_weight = continuous_weight
        self.categorical_weight = categorical_weight
        self.label_smoothing = label_smoothing
        self.conditional_mask = conditional_mask
        self.tier = tier
        self._modulation_weight = modulation_loss_weight
        self._modulation_warmup_epochs = modulation_warmup_epochs
        self._modulation_loss: ModulationLoss | None = None
        if modulation_loss_weight > 0:
            self._modulation_loss = ModulationLoss(pos_weight=modulation_pos_weight)

        if importance_weights is not None:
            self.register_buffer("importance_weights", importance_weights.float())
        else:
            self.register_buffer(
                "importance_weights", torch.ones(n_continuous, dtype=torch.float32)
            )

        # Build conditional mask mappings
        self._static_mask_indices: list[int] = []
        self._module_on_map: dict[int, list[int]] = {}

        if conditional_mask and continuous_names is not None:
            # Static mask: indices of continuous params that are always unlearnable
            # For tier >= 3, modulation routing makes LFOs/randoms/env audible,
            # so skip static masking entirely.
            if tier < 3:
                for i, name in enumerate(continuous_names):
                    if any(name.startswith(prefix) for prefix in _STATIC_MASK_PREFIXES):
                        self._static_mask_indices.append(i)

                if self._static_mask_indices:
                    log.info(
                        "Conditional loss mask: %d static-masked continuous params "
                        "(LFOs, random gens, env 3-6)",
                        len(self._static_mask_indices),
                    )
            else:
                log.info(
                    "Tier %d: skipping static loss mask (LFOs/randoms/envs "
                    "are audible via modulation routing)",
                    tier,
                )

            # Module-on mapping: categorical index -> list of continuous indices
            # Active for all tiers — disabled-module params are inaudible regardless
            if categorical_names is not None:
                for module in _MODULE_ON_SUFFIXES:
                    on_name = f"{module}_on"
                    if on_name in categorical_names:
                        cat_idx = categorical_names.index(on_name)
                        cont_indices = [
                            i for i, name in enumerate(continuous_names)
                            if name.startswith(f"{module}_")
                        ]
                        if cont_indices:
                            self._module_on_map[cat_idx] = cont_indices
                            log.info(
                                "Conditional loss mask: %s gates %d continuous params",
                                on_name, len(cont_indices),
                            )

    def forward(
        self,
        continuous_pred: torch.Tensor,
        continuous_target: torch.Tensor,
        categorical_logits: list[torch.Tensor],
        categorical_target: torch.Tensor,
        modulation_pred: torch.Tensor | None = None,
        modulation_target: torch.Tensor | None = None,
        current_epoch: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss.

        Args:
            continuous_pred: (B, n_continuous) predicted values in [0,1].
            continuous_target: (B, n_continuous) target values in [0,1].
            categorical_logits: List of (B, n_options_i) logit tensors.
            categorical_target: (B, n_categorical) integer class indices.
            modulation_pred: (B, 4, n_src, n_dst) or None.
            modulation_target: (B, 4, n_src, n_dst) or None.
            current_epoch: Current training epoch (for modulation loss warmup).

        Returns:
            (total_loss, continuous_loss, categorical_loss, modulation_loss)
        """
        # Continuous: importance-weighted MSE
        sq_err = (continuous_pred - continuous_target) ** 2  # (B, n_cont)

        # Apply conditional mask if enabled
        if self.conditional_mask and (self._static_mask_indices or self._module_on_map):
            mask = torch.ones_like(continuous_target)  # (B, n_cont)

            # Zero out static mask indices (always unlearnable)
            if self._static_mask_indices:
                mask[:, self._static_mask_indices] = 0.0

            # Zero out continuous params for disabled modules
            for cat_idx, cont_indices in self._module_on_map.items():
                # Where the module is off (categorical target == 0), mask its params
                module_off = (categorical_target[:, cat_idx] == 0).unsqueeze(1)  # (B, 1)
                # Broadcast: zero out cont_indices columns for rows where module is off
                mask[:, cont_indices] = mask[:, cont_indices] * (~module_off).float()

            sq_err = sq_err * mask
            weighted_sq_err = sq_err * self.importance_weights.unsqueeze(0)
            cont_loss = weighted_sq_err.sum() / mask.sum().clamp(min=1)
        else:
            weighted_sq_err = sq_err * self.importance_weights.unsqueeze(0)
            cont_loss = weighted_sq_err.mean()

        # Categorical: mean cross-entropy over all categorical params with label smoothing
        if len(categorical_logits) > 0:
            cat_losses = []
            for i, logits in enumerate(categorical_logits):
                target_i = categorical_target[:, i]
                n_classes = logits.shape[1]
                # Clamp targets to valid range to prevent CUDA assert crashes
                # (can happen if dataset has values exceeding schema n_options)
                target_i = target_i.clamp(0, n_classes - 1)
                cat_losses.append(
                    F.cross_entropy(logits, target_i, label_smoothing=self.label_smoothing)
                )
            cat_loss = torch.stack(cat_losses).mean()
        else:
            cat_loss = torch.tensor(0.0, device=continuous_pred.device)

        # Modulation loss (optional, with linear warmup)
        mod_loss = torch.tensor(0.0, device=continuous_pred.device)
        if (
            modulation_pred is not None
            and modulation_target is not None
            and modulation_target.numel() > 0
            and self._modulation_loss is not None
        ):
            mod_loss = self._modulation_loss(modulation_pred, modulation_target)

        # Linearly ramp modulation weight from 0 to configured value over warmup
        if self._modulation_warmup_epochs > 0:
            warmup_scale = min(current_epoch / self._modulation_warmup_epochs, 1.0)
        else:
            warmup_scale = 1.0
        effective_mod_weight = self._modulation_weight * warmup_scale

        total = (
            self.continuous_weight * cont_loss
            + self.categorical_weight * cat_loss
            + effective_mod_weight * mod_loss
        )
        return total, cont_loss, cat_loss, mod_loss
