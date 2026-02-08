"""HDF5 reader for verification and downstream use."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

log = logging.getLogger(__name__)


class HDF5Reader:
    """Read and inspect HDF5 dataset files.

    Usage::

        with HDF5Reader(Path("data.h5")) as reader:
            info = reader.info()
            audio, params = reader.get_sample(42)
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: h5py.File | None = None

    def open(self) -> None:
        self._file = h5py.File(self.path, "r")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> HDF5Reader:
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def n_samples(self) -> int:
        assert self._file is not None
        return self._file["audio/waveforms"].shape[0]

    def info(self) -> dict[str, Any]:
        """Return summary information about the dataset."""
        assert self._file is not None
        f = self._file

        info: dict[str, Any] = {
            "n_samples": f["audio/waveforms"].shape[0],
            "file_size_mb": self.path.stat().st_size / (1024 * 1024),
        }

        # Dataset shapes
        info["shapes"] = {}
        for name in _walk_datasets(f):
            info["shapes"][name] = f[name].shape

        # Schema attributes
        if "schema" in f:
            info["schema"] = {}
            for key, val in f["schema"].attrs.items():
                if isinstance(val, np.ndarray):
                    val = [
                        v.decode("utf-8") if isinstance(v, bytes) else v
                        for v in val
                    ]
                elif isinstance(val, bytes):
                    val = val.decode("utf-8")
                info["schema"][key] = val

        # Source distribution
        if "metadata/source" in f:
            sources = f["metadata/source"][:]
            unique, counts = np.unique(sources, return_counts=True)
            info["source_distribution"] = {
                s.decode("utf-8") if isinstance(s, bytes) else s: int(c)
                for s, c in zip(unique, counts)
            }

        # Tier distribution
        if "metadata/tier" in f:
            tiers = f["metadata/tier"][:]
            unique, counts = np.unique(tiers, return_counts=True)
            info["tier_distribution"] = {
                int(t): int(c) for t, c in zip(unique, counts)
            }

        # MIDI note distribution
        if "metadata/midi_note" in f:
            notes = f["metadata/midi_note"][:]
            unique, counts = np.unique(notes, return_counts=True)
            info["midi_note_distribution"] = {
                int(n): int(c) for n, c in zip(unique, counts)
            }

        return info

    def get_sample(self, idx: int) -> dict[str, Any]:
        """Load a single sample by index."""
        assert self._file is not None
        f = self._file

        sample: dict[str, Any] = {
            "audio": f["audio/waveforms"][idx],
            "continuous": f["params/continuous"][idx],
            "categorical": f["params/categorical"][idx],
            "midi_note": int(f["metadata/midi_note"][idx]),
            "source": f["metadata/source"][idx].decode("utf-8"),
            "tier": int(f["metadata/tier"][idx]),
            "preset_hash": f["metadata/preset_hash"][idx].decode("utf-8"),
        }

        if "metadata/preset_name" in f:
            sample["preset_name"] = f["metadata/preset_name"][idx].decode("utf-8")

        if "params/modulation_t3" in f:
            sample["modulation_t3"] = f["params/modulation_t3"][idx]

        return sample

    def get_batch(self, start: int, end: int) -> dict[str, np.ndarray]:
        """Load a contiguous batch of samples."""
        assert self._file is not None
        f = self._file

        batch: dict[str, np.ndarray] = {
            "audio": f["audio/waveforms"][start:end],
            "continuous": f["params/continuous"][start:end],
            "categorical": f["params/categorical"][start:end],
            "midi_note": f["metadata/midi_note"][start:end],
        }

        if "params/modulation_t3" in f:
            batch["modulation_t3"] = f["params/modulation_t3"][start:end]

        return batch

    def get_continuous_names(self) -> list[str]:
        """Return continuous parameter names from schema."""
        assert self._file is not None
        if "schema" not in self._file:
            return []
        raw = self._file["schema"].attrs.get("continuous_names", [])
        return [v.decode("utf-8") if isinstance(v, bytes) else v for v in raw]

    def get_categorical_names(self) -> list[str]:
        """Return categorical parameter names from schema."""
        assert self._file is not None
        if "schema" not in self._file:
            return []
        raw = self._file["schema"].attrs.get("categorical_names", [])
        return [v.decode("utf-8") if isinstance(v, bytes) else v for v in raw]

    def get_importance_weights(self) -> np.ndarray | None:
        """Return importance weights if they exist."""
        assert self._file is not None
        if "importance_weights/weights" in self._file:
            return self._file["importance_weights/weights"][:]
        return None


def _walk_datasets(group: h5py.Group, prefix: str = "") -> list[str]:
    """Recursively list all dataset paths in an HDF5 group."""
    names: list[str] = []
    for key in group:
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(group[key], h5py.Dataset):
            names.append(path)
        elif isinstance(group[key], h5py.Group):
            names.extend(_walk_datasets(group[key], path))
    return names
