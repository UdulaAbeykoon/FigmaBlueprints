"""Pipeline configuration: tier definitions, paths, and constants."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vital install paths (macOS defaults; Linux/Windows added for completeness)
# ---------------------------------------------------------------------------
VITAL_PRESET_DIRS: list[Path] = [
    Path.home() / "Library" / "Application Support" / "Vital" / "Factory" / "Presets",
    Path.home() / "Library" / "Application Support" / "Vital" / "User" / "Presets",
    Path("/Library/Audio/Presets/Vital"),
    # Linux
    Path.home() / ".local" / "share" / "vital" / "Factory" / "Presets",
    Path.home() / ".local" / "share" / "vital" / "User" / "Presets",
    # Windows (via WSL or native)
    Path.home() / "Documents" / "Vital" / "Factory" / "Presets",
    Path.home() / "Documents" / "Vital" / "User" / "Presets",
]

VITAL_WAVETABLE_DIRS: list[Path] = [
    Path.home() / "Library" / "Application Support" / "Vital" / "Factory" / "Wavetables",
    Path.home() / "Library" / "Application Support" / "Vital" / "User" / "Wavetables",
    Path.home() / ".local" / "share" / "vital" / "Factory" / "Wavetables",
    Path.home() / ".local" / "share" / "vital" / "User" / "Wavetables",
]

# ---------------------------------------------------------------------------
# Tier 1 module prefixes: params whose name starts with these are tier 1.
# Everything else discovered from Vita is tier 2+.
# ---------------------------------------------------------------------------
TIER1_MODULES: set[str] = {"osc_1", "osc_2", "filter_1", "env_1"}

# Params that are tier 1 but don't match a module prefix pattern.
TIER1_EXTRA_PARAMS: set[str] = {"volume"}

# ---------------------------------------------------------------------------
# Modulation
# ---------------------------------------------------------------------------

# Per-connection modulation properties stored as matrix channels.
N_MOD_CHANNELS = 4
MOD_CHANNEL_AMOUNT = 0
MOD_CHANNEL_BIPOLAR = 1
MOD_CHANNEL_POWER = 2
MOD_CHANNEL_STEREO = 3

# Params that crash Vita when used as modulation destinations.
# Controls whose ``details.options`` property segfaults Vita's C++ engine.
# For categorical ones, n_options is inferred from ``int(max - min) + 1``.
OPTIONS_CRASH_CONTROLS: frozenset[str] = frozenset({
    "filter_1_style", "filter_2_style", "filter_fx_style",
    "osc_1_view_2d", "osc_2_view_2d", "osc_3_view_2d", "view_spectrogram",
})

# Params that crash Vita when used as modulation destinations.
MOD_DEST_BLOCKLIST: set[str] = {
    "chorus_voices",
    "compressor_band_lower_ratio", "compressor_band_lower_threshold",
    "compressor_band_upper_ratio", "compressor_band_upper_threshold",
    "compressor_enabled_bands",
    "compressor_high_lower_ratio", "compressor_high_lower_threshold",
    "compressor_high_upper_ratio", "compressor_high_upper_threshold",
    "compressor_low_lower_ratio", "compressor_low_lower_threshold",
    "compressor_low_upper_ratio", "compressor_low_upper_threshold",
    "delay_style", "distortion_type",
    "flanger_offset", "phaser_offset", "reverb_damping",
    "pitch_wheel", "mod_wheel", "stereo_mode",
}

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for the dataset generation pipeline."""

    tier: Literal[1, 2, 3] = 1
    n_samples: int = 1000
    output_path: Path = field(default_factory=lambda: Path("data/dataset.h5"))
    seed: int = 42
    workers: int = 1

    # Rendering
    sample_rate: int = 44100
    render_duration: float = 2.0  # seconds
    midi_notes: list[int] = field(default_factory=lambda: [48, 60, 72])
    midi_velocity: int = 100
    note_duration: float = 1.5  # how long the note is held (rest is release)

    # Validation
    rms_floor: float = 0.01  # minimum RMS to not be considered silent (~-40 dB)
    peak_ceiling: float = 0.99  # maximum peak before clipping
    render_timeout: float = 5.0  # seconds per render

    # HDF5
    chunk_size: int = 256
    compression: str = "gzip"
    compression_level: int = 4

    # Multiprocessing
    max_tasks_per_child: int = 500

    # Wavetable catalog
    wavetable_catalog_path: Path = field(
        default_factory=lambda: Path("data/wavetable_catalog.json")
    )

    # Community presets
    community_preset_dir: Path = field(
        default_factory=lambda: Path("presets/community")
    )

    @property
    def n_audio_samples(self) -> int:
        return int(self.sample_rate * self.render_duration)
