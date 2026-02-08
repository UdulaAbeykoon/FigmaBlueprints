"""Chunked HDF5 writer with gzip compression and append support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from datagen.storage.schema import HDF5Schema

log = logging.getLogger(__name__)


class HDF5Writer:
    """Writes audio-parameter pairs to HDF5 with chunked, compressed storage.

    Supports appending to existing files and auto-flushing at chunk boundaries
    for crash resilience.

    Usage::

        schema = HDF5Schema.from_config(config)
        with HDF5Writer(Path("data.h5"), schema) as writer:
            writer.append(audio, continuous, categorical, metadata)
    """

    def __init__(self, path: Path, schema: HDF5Schema) -> None:
        self.path = path
        self.schema = schema
        self._file: h5py.File | None = None
        self._count = 0
        self._buffer: dict[str, list[np.ndarray]] = {}
        self._flush_every = schema.chunk_size

    def open(self) -> None:
        """Open or create the HDF5 file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.exists():
            self._file = h5py.File(self.path, "a")
            # Get current count from existing data
            if "audio/waveforms" in self._file:
                self._count = self._file["audio/waveforms"].shape[0]
                log.info("Opened existing HDF5 with %d samples", self._count)
            else:
                self._create_datasets()
        else:
            self._file = h5py.File(self.path, "w")
            self._create_datasets()

    def _create_datasets(self) -> None:
        """Create all datasets and write schema attributes."""
        assert self._file is not None

        for ds_name, spec in self.schema.dataset_specs().items():
            if ds_name not in self._file:
                self._file.create_dataset(ds_name, **spec)

        # Write schema attributes
        schema_grp = self._file.require_group("schema")
        for key, val in self.schema.schema_attributes().items():
            if isinstance(val, list):
                schema_grp.attrs[key] = [
                    s.encode("utf-8") if isinstance(s, str) else s for s in val
                ]
            else:
                schema_grp.attrs[key] = val

        self._file.flush()

    def append(
        self,
        audio: np.ndarray,
        continuous: np.ndarray,
        categorical: np.ndarray,
        midi_note: int,
        source: str,
        tier: int,
        preset_hash: str,
        preset_name: str = "",
        modulation_t3: np.ndarray | None = None,
    ) -> None:
        """Buffer a single sample. Flushes to disk at chunk boundaries.

        Args:
            audio: Shape (2, n_samples) float32.
            continuous: Shape (n_continuous,) float32, normalized [0,1].
            categorical: Shape (n_categorical,) int32.
            midi_note: MIDI note number.
            source: One of "synthetic", "community", "factory".
            tier: Tier number (1, 2, 3).
            preset_hash: SHA256 hash string.
            modulation_t3: Shape (4, n_src, n_dst) float32, tier 3 modulation.
        """
        expected_audio = (2, self.schema.n_audio_samples)
        assert audio.shape == expected_audio, (
            f"audio shape {audio.shape} != expected {expected_audio}"
        )
        assert continuous.shape == (self.schema.n_continuous,), (
            f"continuous shape {continuous.shape} != expected ({self.schema.n_continuous},)"
        )
        assert categorical.shape == (self.schema.n_categorical,), (
            f"categorical shape {categorical.shape} != expected ({self.schema.n_categorical},)"
        )

        self._buffer_add("audio/waveforms", audio)
        self._buffer_add("params/continuous", continuous)
        self._buffer_add("params/categorical", categorical)
        self._buffer_add("metadata/midi_note", np.array(midi_note, dtype=np.int32))
        self._buffer_add("metadata/source", np.bytes_(source.encode("utf-8")))
        self._buffer_add("metadata/tier", np.array(tier, dtype=np.int32))
        self._buffer_add("metadata/preset_hash", np.bytes_(preset_hash.encode("utf-8")))
        self._buffer_add("metadata/preset_name", np.bytes_(preset_name.encode("utf-8")))

        if modulation_t3 is not None and self.schema.tier >= 3:
            self._buffer_add("params/modulation_t3", modulation_t3)

        if len(self._buffer.get("audio/waveforms", [])) >= self._flush_every:
            self.flush()

    def _buffer_add(self, key: str, data: np.ndarray) -> None:
        if key not in self._buffer:
            self._buffer[key] = []
        self._buffer[key].append(data)

    def flush(self) -> None:
        """Write buffered data to disk."""
        if not self._buffer:
            return
        assert self._file is not None

        n_new = len(self._buffer.get("audio/waveforms", []))
        if n_new == 0:
            return

        for ds_name, items in self._buffer.items():
            if ds_name not in self._file:
                continue

            ds = self._file[ds_name]
            batch = np.stack(items)
            old_len = ds.shape[0]
            new_len = old_len + len(items)
            ds.resize(new_len, axis=0)
            ds[old_len:new_len] = batch

        self._count += n_new
        self._buffer.clear()
        self._file.flush()
        log.debug("Flushed %d samples (total: %d)", n_new, self._count)

    @property
    def count(self) -> int:
        return self._count + len(self._buffer.get("audio/waveforms", []))

    def write_importance_weights(self, weights: np.ndarray) -> None:
        """Write importance weights as a separate dataset."""
        assert self._file is not None
        grp = self._file.require_group("importance_weights")
        if "weights" in grp:
            del grp["weights"]
        grp.create_dataset("weights", data=weights.astype(np.float32))
        self._file.flush()

    def close(self) -> None:
        """Flush remaining data and close the file."""
        self.flush()
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> HDF5Writer:
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
