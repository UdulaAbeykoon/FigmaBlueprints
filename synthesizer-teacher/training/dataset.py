"""PyTorch Dataset reading precomputed mel spectrograms from HDF5."""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)

# How many samples to read/write per sequential HDF5 chunk during precompute.
_CHUNK = 512

FEATURES_KEY = "features/mel_spectrogram"
MOD_FEATURES_KEY = "features/modulation_t3"
MOD_PARAMS_KEY = "params/modulation_t3"


def make_train_val_split(
    path: Path,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices by preset hash so no preset leaks across sets.

    Returns:
        (train_indices, val_indices) as sorted int64 numpy arrays.
    """
    with h5py.File(path, "r") as f:
        hashes = f["metadata/preset_hash"][:]

    groups: dict[bytes, list[int]] = {}
    for idx, h in enumerate(hashes):
        groups.setdefault(bytes(h), []).append(idx)

    group_keys = list(groups.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(group_keys)

    n_val_groups = max(1, int(len(group_keys) * val_fraction))
    val_keys = set(group_keys[:n_val_groups])

    train_indices: list[int] = []
    val_indices: list[int] = []
    for key in group_keys:
        (val_indices if key in val_keys else train_indices).extend(groups[key])

    train_arr = np.sort(np.array(train_indices, dtype=np.int64))
    val_arr = np.sort(np.array(val_indices, dtype=np.int64))

    log.info(
        "Split: %d presets (%d train, %d val) -> %d train / %d val samples",
        len(group_keys),
        len(group_keys) - n_val_groups,
        n_val_groups,
        len(train_arr),
        len(val_arr),
    )
    return train_arr, val_arr


# ------------------------------------------------------------------
# Precompute mels to disk (run once)
# ------------------------------------------------------------------


def precompute_mels(
    path: Path,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    device: torch.device | None = None,
) -> None:
    """Read audio sequentially, compute log-mel spectrograms on device, write uncompressed.

    Adds ``features/mel_spectrogram`` to the existing HDF5 file.
    Uncompressed storage enables fast random access during training.
    """
    dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(path, "a") as f:
        sr = int(f["schema"].attrs["sample_rate"])
        n_total = f["audio/waveforms"].shape[0]

        mel_transform = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, power=2.0,
        ).to(dev)

        # Determine output shape from one sample
        sample = torch.from_numpy(f["audio/waveforms"][0].mean(axis=0)).unsqueeze(0).to(dev)
        mel_shape = torch.log1p(mel_transform(sample.unsqueeze(0))).shape[1:]  # (1, n_mels, T)

        # Delete existing and recreate
        if FEATURES_KEY in f:
            del f[FEATURES_KEY]

        ds = f.create_dataset(
            FEATURES_KEY,
            shape=(n_total, *mel_shape),
            dtype=np.float32,
            chunks=(1, *mel_shape),  # one sample per chunk → optimal random access
        )
        ds.attrs["n_mels"] = n_mels
        ds.attrs["n_fft"] = n_fft
        ds.attrs["hop_length"] = hop_length
        ds.attrs["sample_rate"] = sr

        for start in tqdm(range(0, n_total, _CHUNK), desc="Computing mels"):
            end = min(start + _CHUNK, n_total)
            audio = f["audio/waveforms"][start:end]  # (chunk, 2, samples)

            mono = torch.from_numpy(audio.mean(axis=1)).unsqueeze(1).to(dev)
            mels = torch.log1p(mel_transform(mono))  # (chunk, 1, n_mels, T)
            ds[start:end] = mels.cpu().numpy()

    size_gb = n_total * np.prod(mel_shape) * 4 / 1e9
    log.info("Wrote %s: %d samples, %.2f GB", FEATURES_KEY, n_total, size_gb)


# ------------------------------------------------------------------
# Precompute modulation to uncompressed HDF5 (run once)
# ------------------------------------------------------------------


def precompute_modulation(path: Path) -> None:
    """Copy modulation_t3 from compressed to uncompressed for fast random access.

    The original ``params/modulation_t3`` is gzip-compressed, making random
    access catastrophically slow in a DataLoader. This copies it to
    ``features/modulation_t3`` as uncompressed, chunked per-sample.
    """
    with h5py.File(path, "a") as f:
        if MOD_PARAMS_KEY not in f:
            log.info("No %s in dataset; skipping modulation precompute", MOD_PARAMS_KEY)
            return

        src = f[MOD_PARAMS_KEY]
        n_total = src.shape[0]
        mod_shape = src.shape[1:]  # (4, 32, N_dest)

        if MOD_FEATURES_KEY in f:
            del f[MOD_FEATURES_KEY]

        dst = f.create_dataset(
            MOD_FEATURES_KEY,
            shape=(n_total, *mod_shape),
            dtype=np.float32,
            chunks=(1, *mod_shape),
        )

        for start in tqdm(range(0, n_total, _CHUNK), desc="Precomputing modulation"):
            end = min(start + _CHUNK, n_total)
            dst[start:end] = src[start:end]

    size_gb = n_total * np.prod(mod_shape) * 4 / 1e9
    log.info("Wrote %s: %d samples, %.2f GB", MOD_FEATURES_KEY, n_total, size_gb)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------


class VitalDataset(Dataset):
    """Reads precomputed mels + params from HDF5.

    Each DataLoader worker lazily opens its own file handle.
    Mels are stored uncompressed so random access is fast.

    Returns 5-tuples: (mel, continuous, categorical, midi_note, modulation).
    When no modulation data exists, modulation is an empty tensor.
    """

    def __init__(
        self,
        path: Path,
        indices: np.ndarray,
        training: bool = False,
        spec_aug_freq_mask: int = 12,
        spec_aug_time_mask: int = 12,
        spec_aug_n_masks: int = 2,
    ) -> None:
        self.path = path
        self.indices = indices
        self._file: h5py.File | None = None
        self._has_modulation: bool | None = None  # Lazy-detected

        # SpecAugment (training only)
        self.augment: torch.nn.Sequential | None = None
        if training and spec_aug_n_masks > 0:
            layers: list[torch.nn.Module] = []
            for _ in range(spec_aug_n_masks):
                layers.append(T.FrequencyMasking(freq_mask_param=spec_aug_freq_mask))
                layers.append(T.TimeMasking(time_mask_param=spec_aug_time_mask))
            self.augment = torch.nn.Sequential(*layers)

    def _ensure_open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
            # Detect modulation data on first open
            if self._has_modulation is None:
                self._has_modulation = (
                    MOD_FEATURES_KEY in self._file
                    or MOD_PARAMS_KEY in self._file
                )
        return self._file

    @property
    def has_modulation(self) -> bool:
        """Whether the dataset contains modulation data."""
        if self._has_modulation is None:
            self._ensure_open()
        return self._has_modulation  # type: ignore[return-value]

    def __del__(self) -> None:
        """Close h5py file handle to prevent resource leaks."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass  # File may already be closed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Return (mel, continuous_params, categorical_params, midi_note, modulation).

        Note: midi_note is for rendering comparison audio, NOT for model input.
        The model should be pitch-agnostic.

        modulation is (4, n_sources, n_destinations) or empty tensor if not available.
        """
        f = self._ensure_open()
        i = int(self.indices[idx])
        mel = torch.from_numpy(f[FEATURES_KEY][i])
        if self.augment is not None:
            mel = self.augment(mel)
        cont = torch.from_numpy(f["params/continuous"][i].astype(np.float32))
        cat = torch.from_numpy(f["params/categorical"][i].astype(np.int64))
        midi_note = int(f["metadata/midi_note"][i])

        # Load modulation matrix if available
        if self._has_modulation:
            # Prefer uncompressed features/ copy, fall back to compressed params/
            mod_key = MOD_FEATURES_KEY if MOD_FEATURES_KEY in f else MOD_PARAMS_KEY
            mod = torch.from_numpy(f[mod_key][i].astype(np.float32))
        else:
            mod = torch.empty(0)

        return mel, cont, cat, midi_note, mod
