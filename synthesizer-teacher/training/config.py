"""Training configuration dataclass with validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    """All hyperparameters for the Tier-1 training pipeline."""

    # Dataset
    dataset_path: Path = field(default_factory=lambda: Path("data/tier1_20k.h5"))
    val_fraction: float = 0.15

    # Mel spectrogram
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512

    # Model
    freeze_early: bool = True
    mlp_hidden: int = 512
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_epochs: int = 5
    num_workers: int = 4  # Now configurable instead of hardcoded

    # Early stopping
    early_stopping_patience: int = 0  # 0 = disabled
    early_stopping_min_delta: float = 1e-4

    # Gradient accumulation
    gradient_accumulation_steps: int = 1

    # Loss
    continuous_loss_weight: float = 1.0
    categorical_loss_weight: float = 0.5
    categorical_label_smoothing: float = 0.0  # Label smoothing for class imbalance
    conditional_loss_mask: bool = True  # Mask loss for unlearnable params
    simple_categorical_heads: bool = True  # Use linear heads instead of MLP
    tier: int = 0  # 0 = auto-detect from dataset schema; 1/2/3 explicit

    # Wavetable catalog (for tier-3 preset rendering/export)
    wavetable_catalog: str = ""  # Path to wavetable_catalog.json, or "" to auto-detect

    # Modulation (tier-3, V1 dense head)
    modulation_loss_weight: float = 0.3
    modulation_pos_weight: float = 20.0  # BCE pos_weight for sparse connections

    # V2 architecture
    encoder_type: str = "resnet18"  # "resnet18" (V1) or "ast" (V2)

    # V2 phase-based training
    phase1_epochs: int = 30  # Warmup: frozen encoder, scalar heads only
    phase2_epochs: int = 120  # Fine-tune: unfreeze last 2 blocks, enable all heads
    # Phase 3 = remaining epochs (total - phase1 - phase2): full unfreeze

    # V2 learning rates (discriminative)
    encoder_lr: float = 1e-5  # Lower LR for pretrained encoder in phase 2
    head_lr: float = 1e-4  # Higher LR for prediction heads

    # V2 head config
    n_mod_queries: int = 20  # DETR modulation decoder query slots
    n_lfo_shapes: int = 8  # Number of LFO shape classes
    n_wavetables: int = 70  # Number of wavetable options per oscillator

    # V2 loss weights
    wavetable_loss_weight: float = 1.0
    lfo_shape_loss_weight: float = 0.3
    modulation_set_loss_weight: float = 0.5
    modulation_warmup_epochs: int = 10  # Ramp mod loss weight over first N epochs of phase 2

    # SpecAugment
    spec_aug_freq_mask: int = 12  # max frequency bands to mask
    spec_aug_time_mask: int = 12  # max time steps to mask
    spec_aug_n_masks: int = 2  # number of each mask type

    # W&B
    wandb_project: str = "vital-inverse-synthesis"

    # Evaluation
    log_audio_every: int = 5
    n_render_eval: int = 8
    compute_spectral_metrics: bool = True  # Compute spectral distance on validation

    # Device
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        errors: list[str] = []

        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
        if self.lr <= 0:
            errors.append(f"lr must be positive, got {self.lr}")
        if self.epochs <= 0:
            errors.append(f"epochs must be positive, got {self.epochs}")
        if not 0 < self.val_fraction < 1:
            errors.append(f"val_fraction must be in (0, 1), got {self.val_fraction}")
        if self.dropout < 0 or self.dropout >= 1:
            errors.append(f"dropout must be in [0, 1), got {self.dropout}")
        if self.mlp_hidden <= 0:
            errors.append(f"mlp_hidden must be positive, got {self.mlp_hidden}")
        if self.warmup_epochs < 0:
            errors.append(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
        if self.num_workers < 0:
            errors.append(f"num_workers must be >= 0, got {self.num_workers}")
        if self.gradient_accumulation_steps < 1:
            errors.append(f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}")
        if self.categorical_label_smoothing < 0 or self.categorical_label_smoothing >= 1:
            errors.append(f"categorical_label_smoothing must be in [0, 1), got {self.categorical_label_smoothing}")
        if self.early_stopping_patience < 0:
            errors.append(f"early_stopping_patience must be >= 0, got {self.early_stopping_patience}")

        if errors:
            raise ValueError("TrainConfig validation failed:\n  " + "\n  ".join(errors))
