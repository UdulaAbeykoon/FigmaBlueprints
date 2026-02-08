"""HDF5 dataset schema definition for audio-parameter storage."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from datagen.config import N_MOD_CHANNELS, PipelineConfig
from datagen.params.registry import ParamRegistry


@dataclass
class HDF5Schema:
    """Defines the HDF5 dataset layout for a given pipeline configuration.

    Datasets:
        audio/waveforms:      (N, 2, n_samples)                   float32
        params/continuous:    (N, n_continuous)                     float32
        params/categorical:   (N, n_categorical)                   int32
        params/modulation_t3: (N, 4, n_src, n_dst)                 float32  (tier 3 only)
        metadata/midi_note:   (N,)                                 int32
        metadata/source:      (N,)                                 S10
        metadata/tier:        (N,)                                 int32
        metadata/preset_hash: (N,)                                 S64
        metadata/preset_name: (N,)                                 S80
    """

    n_audio_samples: int
    n_continuous: int
    n_categorical: int
    n_wavetable: int
    tier: int
    chunk_size: int
    compression: str
    compression_level: int
    sample_rate: int
    render_duration: float
    note_duration: float
    continuous_names: list[str]
    categorical_names: list[str]
    wavetable_names: list[str]
    mod_source_names: list[str]
    categorical_n_options: list[int] = field(default_factory=list)
    # Tier 3
    n_mod_sources: int = 0
    n_mod_destinations: int = 0

    @classmethod
    def from_config(cls, config: PipelineConfig, registry: ParamRegistry) -> HDF5Schema:
        continuous_names = registry.continuous_names(config.tier)
        categorical_names = registry.categorical_names(config.tier)
        wavetable_names = registry.wavetable_names(config.tier)

        all_cat_names = categorical_names + wavetable_names
        categorical_n_options = [registry[name].n_options for name in all_cat_names]

        return cls(
            n_audio_samples=config.n_audio_samples,
            n_continuous=len(continuous_names),
            n_categorical=len(categorical_names) + len(wavetable_names),
            n_wavetable=len(wavetable_names),
            tier=config.tier,
            chunk_size=config.chunk_size,
            compression=config.compression,
            compression_level=config.compression_level,
            sample_rate=config.sample_rate,
            render_duration=config.render_duration,
            note_duration=config.note_duration,
            continuous_names=continuous_names,
            categorical_names=categorical_names,
            wavetable_names=wavetable_names,
            mod_source_names=list(registry.mod_sources),
            categorical_n_options=categorical_n_options,
            n_mod_sources=registry.n_mod_sources,
            n_mod_destinations=len(registry.mod_destinations(config.tier)),
        )

    def dataset_specs(self) -> dict[str, dict]:
        """Return a dict of dataset name -> creation kwargs for h5py."""
        cs = self.chunk_size
        comp = self.compression
        comp_lvl = self.compression_level

        specs = {
            "audio/waveforms": {
                "dtype": np.float32,
                "shape": (0, 2, self.n_audio_samples),
                "maxshape": (None, 2, self.n_audio_samples),
                "chunks": (min(cs, 64), 2, self.n_audio_samples),
                "compression": comp,
                "compression_opts": comp_lvl,
            },
            "params/continuous": {
                "dtype": np.float32,
                "shape": (0, self.n_continuous),
                "maxshape": (None, self.n_continuous),
                "chunks": (cs, self.n_continuous),
                "compression": comp,
                "compression_opts": comp_lvl,
            },
            "params/categorical": {
                "dtype": np.int32,
                "shape": (0, self.n_categorical),
                "maxshape": (None, self.n_categorical),
                "chunks": (cs, self.n_categorical),
                "compression": comp,
                "compression_opts": comp_lvl,
            },
            "metadata/midi_note": {
                "dtype": np.int32,
                "shape": (0,),
                "maxshape": (None,),
                "chunks": (cs,),
            },
            "metadata/source": {
                "dtype": "S10",
                "shape": (0,),
                "maxshape": (None,),
                "chunks": (cs,),
            },
            "metadata/tier": {
                "dtype": np.int32,
                "shape": (0,),
                "maxshape": (None,),
                "chunks": (cs,),
            },
            "metadata/preset_hash": {
                "dtype": "S64",
                "shape": (0,),
                "maxshape": (None,),
                "chunks": (cs,),
            },
            "metadata/preset_name": {
                "dtype": "S80",
                "shape": (0,),
                "maxshape": (None,),
                "chunks": (cs,),
            },
        }

        if self.tier >= 3 and self.n_mod_sources > 0 and self.n_mod_destinations > 0:
            specs["params/modulation_t3"] = {
                "dtype": np.float32,
                "shape": (0, N_MOD_CHANNELS, self.n_mod_sources, self.n_mod_destinations),
                "maxshape": (None, N_MOD_CHANNELS, self.n_mod_sources, self.n_mod_destinations),
                "chunks": (min(cs, 32), N_MOD_CHANNELS, self.n_mod_sources, self.n_mod_destinations),
                "compression": comp,
                "compression_opts": comp_lvl,
            }

        return specs

    def schema_attributes(self) -> dict[str, str | int | float | list[str]]:
        """Return schema metadata to store as HDF5 attributes."""
        attrs: dict[str, str | int | float | list[str]] = {
            "version": "2.1.0",
            "sample_rate": self.sample_rate,
            "render_duration": self.render_duration,
            "note_duration": self.note_duration,
            "tier": self.tier,
            "n_continuous": self.n_continuous,
            "n_categorical": self.n_categorical,
            "continuous_names": self.continuous_names,
            "categorical_names": self.categorical_names,
            "wavetable_names": self.wavetable_names,
            "categorical_n_options": self.categorical_n_options,
        }

        if self.tier >= 3:
            attrs["mod_source_names"] = self.mod_source_names

        return attrs
