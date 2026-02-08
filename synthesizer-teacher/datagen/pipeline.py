"""Pipeline orchestrator: sampling -> rendering -> validation -> storage."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm

from datagen.config import (
    MOD_CHANNEL_AMOUNT,
    MOD_CHANNEL_BIPOLAR,
    MOD_CHANNEL_POWER,
    MOD_CHANNEL_STEREO,
    N_MOD_CHANNELS,
    PipelineConfig,
)
from datagen.params.registry import ParamRegistry
from datagen.presets.synthetic import SyntheticPresetGenerator
from datagen.render.engine import RenderEngine
from datagen.render.pool import RenderPool
from datagen.render.validator import AudioValidator
from datagen.storage.schema import HDF5Schema
from datagen.storage.writer import HDF5Writer
from datagen.wavetables.catalog import WavetableCatalog

log = logging.getLogger(__name__)

# Type alias for the render function accepted by the generation loop.
# Maps a batch of presets to per-preset lists of (midi_note, audio) pairs.
RenderBatchFn = Callable[
    [list[dict[str, Any]]],
    list[list[tuple[int, np.ndarray | None]]],
]


@dataclass
class PipelineStats:
    """Statistics from a pipeline run."""

    total_attempted: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def rejection_rate(self) -> float:
        if self.total_attempted == 0:
            return 0.0
        return self.total_rejected / self.total_attempted

    @property
    def renders_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return self.total_attempted / self.elapsed_seconds


class Pipeline:
    """Orchestrates the full dataset generation pipeline.

    Ties together preset generation, rendering, validation, and HDF5 storage.
    Presets are generated in adaptive batches — the batch size adjusts based
    on the observed rejection rate so the pipeline always converges to the
    target sample count without pre-allocating a fixed multiplier.

    Usage::

        config = PipelineConfig(tier=1, n_samples=1000)
        registry = ParamRegistry.from_synth(vita.Synth())
        pipeline = Pipeline(config, registry=registry)
        stats = pipeline.run()
    """

    def __init__(
        self,
        config: PipelineConfig,
        registry: ParamRegistry,
        catalog: WavetableCatalog | None = None,
    ) -> None:
        self.config = config
        self.catalog = catalog
        self.registry = registry
        self.schema = HDF5Schema.from_config(config, registry)
        self.validator = AudioValidator(config)

        # Pre-compute name lists (stable for the lifetime of the pipeline).
        self._continuous_names = registry.continuous_names(config.tier)
        self._categorical_names = registry.categorical_names(config.tier)
        self._wavetable_names = registry.wavetable_names(config.tier)
        self._mod_destinations = registry.mod_destinations(config.tier)
        self._n_mod_sources = registry.n_mod_sources

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        extra_presets: list[dict[str, Any]] | None = None,
    ) -> PipelineStats:
        """Run the full pipeline: generate, render, validate, store.

        Args:
            extra_presets: Additional presets (community/factory) to include
                          alongside synthetic generation.

        Returns:
            PipelineStats with counts and timing.
        """
        stats = PipelineStats()
        start_time = time.monotonic()

        n_extra = len(extra_presets) if extra_presets else 0
        log.info(
            "Pipeline: tier=%d, target=%d samples, %d extra presets",
            self.config.tier, self.config.n_samples, n_extra,
        )

        generator = SyntheticPresetGenerator(
            config=self.config,
            registry=self.registry,
            catalog=self.catalog,
            seed=self.config.seed,
            seed_presets=extra_presets,
        )

        with HDF5Writer(self.config.output_path, self.schema) as writer:
            if self.config.workers > 1:
                with RenderPool(self.config) as pool:
                    self._generation_loop(
                        generator, extra_presets, pool.render_batch, writer, stats,
                    )
            else:
                engine = RenderEngine(self.config)
                self._generation_loop(
                    generator,
                    extra_presets,
                    lambda batch: [engine.render_preset_multi_note(p) for p in batch],
                    writer,
                    stats,
                )

        stats.elapsed_seconds = time.monotonic() - start_time

        log.info(
            "Pipeline complete: %d accepted / %d attempted "
            "(%.1f%% rejection) in %.1fs (%.1f renders/s)",
            stats.total_accepted,
            stats.total_attempted,
            stats.rejection_rate * 100,
            stats.elapsed_seconds,
            stats.renders_per_second,
        )
        return stats

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _generation_loop(
        self,
        generator: SyntheticPresetGenerator,
        extra_presets: list[dict[str, Any]] | None,
        render_batch: RenderBatchFn,
        writer: HDF5Writer,
        stats: PipelineStats,
    ) -> None:
        """Generate, render, validate, and store in adaptive batches.

        Processes ``extra_presets`` first, then streams synthetic presets
        until the target sample count is reached.
        """
        batch_size = self.config.chunk_size
        pending: list[dict[str, Any]] = list(extra_presets or [])

        pbar = tqdm(desc="Generating", unit="sample", total=self.config.n_samples)

        while stats.total_accepted < self.config.n_samples:
            # Ensure we have presets to process
            if not pending:
                pending = generator.generate_batch(
                    self._next_batch_size(stats)
                )

            batch = pending[:batch_size]
            del pending[:batch_size]

            results = render_batch(batch)
            for preset, preset_results in zip(batch, results):
                if stats.total_accepted >= self.config.n_samples:
                    break
                for midi_note, audio in preset_results:
                    if stats.total_accepted >= self.config.n_samples:
                        break
                    self._record_sample(preset, midi_note, audio, writer, stats)

            pbar.n = stats.total_accepted
            pbar.set_postfix(rej=f"{stats.rejection_rate:.0%}")
            pbar.refresh()

        pbar.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_batch_size(self, stats: PipelineStats) -> int:
        """Estimate how many presets to generate for the next batch."""
        remaining = self.config.n_samples - stats.total_accepted
        if stats.total_attempted > 0:
            accept_rate = max(0.1, 1.0 - stats.rejection_rate)
        else:
            accept_rate = 0.5  # conservative initial estimate

        notes_per_preset = len(self.config.midi_notes)
        presets_needed = int(remaining / (accept_rate * notes_per_preset)) + 16
        return max(presets_needed, 64)

    def _record_sample(
        self,
        preset: dict[str, Any],
        midi_note: int,
        audio: np.ndarray | None,
        writer: HDF5Writer,
        stats: PipelineStats,
    ) -> None:
        """Validate a rendered sample and write to HDF5 if accepted."""
        stats.total_attempted += 1
        result = self.validator.validate(audio)

        if not result.valid:
            stats.total_rejected += 1
            reason = result.reason.name if result.reason else "UNKNOWN"
            stats.rejection_reasons[reason] = (
                stats.rejection_reasons.get(reason, 0) + 1
            )
            return

        writer.append(
            audio=audio,
            continuous=self._extract_continuous(preset),
            categorical=self._extract_categorical(preset),
            midi_note=midi_note,
            source=preset.get("_source", "synthetic"),
            tier=self.config.tier,
            preset_hash=preset.get("_preset_hash", ""),
            preset_name=preset.get("_preset_name", ""),
            modulation_t3=self._extract_modulation_t3(preset),
        )
        stats.total_accepted += 1

    # ------------------------------------------------------------------
    # Parameter extraction
    # ------------------------------------------------------------------

    def _extract_continuous(self, preset: dict[str, Any]) -> np.ndarray:
        """Extract continuous params into a float32 vector."""
        vec = np.zeros(len(self._continuous_names), dtype=np.float32)
        for i, name in enumerate(self._continuous_names):
            if name in preset:
                vec[i] = float(preset[name])
        return vec

    def _extract_categorical(self, preset: dict[str, Any]) -> np.ndarray:
        """Extract categorical + wavetable params into an int32 vector."""
        all_names = self._categorical_names + self._wavetable_names
        vec = np.zeros(len(all_names), dtype=np.int32)
        for i, name in enumerate(all_names):
            if name in preset:
                vec[i] = int(preset[name])
        return vec

    def _extract_modulation_t3(self, preset: dict[str, Any]) -> np.ndarray | None:
        """Extract tier 3 modulation as dense (4, n_src, n_dst) matrix.

        Channels: [amount, bipolar, power, stereo].
        """
        if self.config.tier < 3:
            return None

        n_dst = len(self._mod_destinations)
        matrix = np.zeros(
            (N_MOD_CHANNELS, self._n_mod_sources, n_dst), dtype=np.float32,
        )

        mod_data = preset.get("_modulation_t3")
        if mod_data is None:
            return matrix

        for conn in mod_data.get("connections", []):
            src_idx = conn["source_idx"]
            dst_idx = conn["dest_idx"]
            if 0 <= src_idx < self._n_mod_sources and 0 <= dst_idx < n_dst:
                matrix[MOD_CHANNEL_AMOUNT, src_idx, dst_idx] = conn["amount"]
                matrix[MOD_CHANNEL_BIPOLAR, src_idx, dst_idx] = conn.get("bipolar", 0.0)
                matrix[MOD_CHANNEL_POWER, src_idx, dst_idx] = conn.get("power", 0.0)
                matrix[MOD_CHANNEL_STEREO, src_idx, dst_idx] = conn.get("stereo", 0.0)

        return matrix
