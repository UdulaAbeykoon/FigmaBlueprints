"""Audio validation: rejection sampling for degenerate renders."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from datagen.config import PipelineConfig

log = logging.getLogger(__name__)


class RejectReason(Enum):
    SILENT = auto()
    CLIPPING = auto()
    NAN = auto()
    INF = auto()
    WRONG_SHAPE = auto()


@dataclass
class ValidationResult:
    valid: bool
    reason: RejectReason | None = None
    rms: float = 0.0
    peak: float = 0.0


class AudioValidator:
    """Validates rendered audio, rejecting silent, clipping, or corrupt samples."""

    def __init__(self, config: PipelineConfig) -> None:
        self.rms_floor = config.rms_floor
        self.peak_ceiling = config.peak_ceiling
        self.expected_channels = 2
        self.expected_samples = config.n_audio_samples

    def validate(self, audio: np.ndarray | None) -> ValidationResult:
        """Check a rendered audio array for common problems.

        Args:
            audio: Expected shape (2, n_samples) float32.

        Returns:
            ValidationResult with valid flag and diagnostics.
        """
        if audio is None:
            return ValidationResult(valid=False, reason=RejectReason.SILENT)

        # Shape check
        if audio.ndim != 2 or audio.shape[0] != self.expected_channels:
            return ValidationResult(valid=False, reason=RejectReason.WRONG_SHAPE)

        # NaN check
        if np.any(np.isnan(audio)):
            return ValidationResult(valid=False, reason=RejectReason.NAN)

        # Inf check
        if np.any(np.isinf(audio)):
            return ValidationResult(valid=False, reason=RejectReason.INF)

        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # Silent check
        if rms < self.rms_floor:
            return ValidationResult(valid=False, reason=RejectReason.SILENT, rms=rms, peak=peak)

        # Clipping check
        if peak > self.peak_ceiling:
            return ValidationResult(valid=False, reason=RejectReason.CLIPPING, rms=rms, peak=peak)

        return ValidationResult(valid=True, rms=rms, peak=peak)
