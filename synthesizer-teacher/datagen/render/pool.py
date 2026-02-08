"""Multiprocessing pool of RenderEngine workers."""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any

import numpy as np

from datagen.config import PipelineConfig

log = logging.getLogger(__name__)

# Per-worker global synth engine (initialized in worker_init)
_worker_engine: Any = None


def _worker_init(config_dict: dict[str, Any]) -> None:
    """Initialize a RenderEngine in each worker process."""
    global _worker_engine
    from datagen.config import PipelineConfig
    from datagen.render.engine import RenderEngine

    config = PipelineConfig(**config_dict)
    _worker_engine = RenderEngine(config)
    log.debug("Worker %d initialized RenderEngine", mp.current_process().pid)


def _worker_render(
    args: tuple[dict[str, Any], int, int],
) -> tuple[int, np.ndarray | None]:
    """Render a single preset at a MIDI note. Called in worker process.

    Args:
        args: (preset_dict, midi_note, midi_velocity)

    Returns:
        (midi_note, audio_or_None)
    """
    preset, midi_note, midi_velocity = args
    global _worker_engine
    try:
        audio = _worker_engine.render_preset(
            preset, midi_note=midi_note, midi_velocity=midi_velocity
        )
        return (midi_note, audio)
    except Exception as e:
        log.debug("Worker render failed: %s", e)
        return (midi_note, None)


class RenderPool:
    """Multiprocessing pool of RenderEngine workers.

    Each worker process creates its own vita.Synth instance on init.
    Uses ``maxtasksperchild`` to reclaim leaked memory from Vita.

    Usage::

        pool = RenderPool(config)
        results = pool.render_batch(presets, midi_notes=[60])
        pool.close()
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._pool: mp.Pool | None = None

        if config.workers > 1:
            # Serialize config as dict for pickling
            config_dict = self._config_to_dict(config)
            self._pool = mp.Pool(
                processes=config.workers,
                initializer=_worker_init,
                initargs=(config_dict,),
                maxtasksperchild=config.max_tasks_per_child,
            )
            log.info("RenderPool started with %d workers", config.workers)
        else:
            # Single-process mode: just create engine directly
            from datagen.render.engine import RenderEngine
            self._engine = RenderEngine(config)
            log.info("RenderPool running in single-process mode")

    def _config_to_dict(self, config: PipelineConfig) -> dict[str, Any]:
        """Convert PipelineConfig to a picklable dict."""
        from dataclasses import asdict
        d = asdict(config)
        # Convert Path objects to strings for pickling
        for key, val in d.items():
            if hasattr(val, "__fspath__"):
                d[key] = str(val)
        return d

    def render_batch(
        self,
        presets: list[dict[str, Any]],
        midi_notes: list[int] | None = None,
    ) -> list[list[tuple[int, np.ndarray | None]]]:
        """Render a batch of presets, each at multiple MIDI notes.

        Args:
            presets: List of preset parameter dicts.
            midi_notes: MIDI notes to render. Defaults to config.midi_notes.

        Returns:
            List of lists of (midi_note, audio_or_None) per preset.
        """
        notes = midi_notes or self.config.midi_notes
        velocity = self.config.midi_velocity

        if self._pool is not None:
            # Build work items
            work_items = [
                (preset, note, velocity)
                for preset in presets
                for note in notes
            ]

            # Map with timeout
            raw_results = self._pool.map(
                _worker_render,
                work_items,
                chunksize=max(1, len(work_items) // (self.config.workers * 4)),
            )

            # Group results by preset
            n_notes = len(notes)
            grouped = []
            for i in range(len(presets)):
                preset_results = raw_results[i * n_notes : (i + 1) * n_notes]
                grouped.append(preset_results)
            return grouped
        else:
            # Single-process mode
            results = []
            for preset in presets:
                preset_results = self._engine.render_preset_multi_note(preset, notes)
                results.append(preset_results)
            return results

    def close(self) -> None:
        """Shut down the pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> RenderPool:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
