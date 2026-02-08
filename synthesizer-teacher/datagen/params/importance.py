"""Perturbation-based parameter importance weight computation.

For each parameter, perturb by +10%, re-render, and measure
multi-resolution STFT distance from the original. Parameters whose
perturbation causes large spectral changes get higher importance weights.

Supports multiprocessing: each worker gets its own Vita instance.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from datagen.config import PipelineConfig
    from datagen.params.registry import ParamRegistry
    from datagen.render.engine import RenderEngine

log = logging.getLogger(__name__)

# Perturbation magnitude (fraction of [0,1] range)
PERTURBATION_DELTA = 0.10


def _spectral_distance(audio_a: np.ndarray, audio_b: np.ndarray) -> float:
    """Multi-resolution STFT distance between two audio signals.

    Uses vectorized strided views instead of a per-frame Python loop.
    """
    # Mono downmix once
    mono_a = audio_a.mean(axis=0) if audio_a.ndim == 2 else audio_a
    mono_b = audio_b.mean(axis=0) if audio_b.ndim == 2 else audio_b

    # Pad to same length
    max_len = max(len(mono_a), len(mono_b))
    if len(mono_a) < max_len:
        mono_a = np.pad(mono_a, (0, max_len - len(mono_a)))
    if len(mono_b) < max_len:
        mono_b = np.pad(mono_b, (0, max_len - len(mono_b)))

    distances = []
    for n_fft in [512, 2048]:  # 2 resolutions instead of 4 — sufficient for ranking
        hop = n_fft // 4
        n_frames = 1 + (max_len - n_fft) // hop
        if n_frames <= 0:
            continue

        window = np.hanning(n_fft).astype(np.float32)

        # Strided view: (n_frames, n_fft) without copying
        stride = mono_a.strides[0]
        shape = (n_frames, n_fft)
        strides = (stride * hop, stride)
        frames_a = np.lib.stride_tricks.as_strided(mono_a, shape=shape, strides=strides)
        frames_b = np.lib.stride_tricks.as_strided(mono_b, shape=shape, strides=strides)

        # Windowed FFT on all frames at once
        spec_a = np.abs(np.fft.rfft(frames_a * window, axis=1)).mean(axis=0)
        spec_b = np.abs(np.fft.rfft(frames_b * window, axis=1)).mean(axis=0)

        log_a = np.log1p(spec_a)
        log_b = np.log1p(spec_b)
        distances.append(float(np.mean(np.abs(log_a - log_b))))

    return float(np.mean(distances)) if distances else 0.0


# ---------------------------------------------------------------------------
# Per-worker state for multiprocessing
# ---------------------------------------------------------------------------
_worker_engine: Any = None
_worker_continuous_names: list[str] = []


def _worker_init(config_dict: dict[str, Any], continuous_names: list[str]) -> None:
    """Initialize a RenderEngine in each worker process."""
    global _worker_engine, _worker_continuous_names
    from datagen.config import PipelineConfig
    from datagen.render.engine import RenderEngine

    config = PipelineConfig(**config_dict)
    _worker_engine = RenderEngine(config)
    _worker_continuous_names = continuous_names


def _worker_process_preset(preset: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Process one preset: render original, then perturb each param.

    Returns:
        (distances, valid_mask) — both shape (n_params,).
    """
    global _worker_engine, _worker_continuous_names
    n_params = len(_worker_continuous_names)
    distances = np.zeros(n_params, dtype=np.float64)
    valid = np.zeros(n_params, dtype=np.int64)

    try:
        original_audio = _worker_engine.render_preset(preset, midi_note=60)
    except Exception:
        return distances, valid

    if original_audio is None:
        return distances, valid

    for i, param_name in enumerate(_worker_continuous_names):
        if param_name not in preset:
            continue

        original_val = preset[param_name]
        perturbed_val = min(original_val + PERTURBATION_DELTA, 1.0)
        if perturbed_val == original_val:
            perturbed_val = max(original_val - PERTURBATION_DELTA, 0.0)
        if perturbed_val == original_val:
            continue

        perturbed_preset = dict(preset)
        perturbed_preset[param_name] = perturbed_val

        try:
            perturbed_audio = _worker_engine.render_preset(perturbed_preset, midi_note=60)
        except Exception:
            continue

        if perturbed_audio is None:
            continue

        distances[i] = _spectral_distance(original_audio, perturbed_audio)
        valid[i] = 1

    return distances, valid


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_importance_weights(
    engine: RenderEngine,
    config: PipelineConfig,
    registry: ParamRegistry,
    n_base_presets: int = 500,
    seed: int = 42,
    workers: int = 0,
    continuous_names_override: list[str] | None = None,
) -> np.ndarray:
    """Compute importance weights for continuous parameters.

    For each of ``n_base_presets`` random presets and each continuous param:
    1. Render the original preset
    2. Perturb the param by +delta (single direction)
    3. Render the perturbed preset
    4. Measure spectral distance

    Args:
        workers: Number of parallel workers. 0 = auto (cpu_count), 1 = serial.
        continuous_names_override: If provided, compute weights for exactly
            these params instead of deriving from config.tier.

    Returns:
        Array of shape (n_continuous,) with mean spectral distances,
        normalized to sum to n_continuous (so mean weight = 1.0).
    """
    from datagen.params.sampler import ParamSampler

    sampler = ParamSampler(config, registry, seed=seed)
    continuous_names = continuous_names_override or registry.continuous_names(config.tier)
    n_params = len(continuous_names)

    total_distances = np.zeros(n_params, dtype=np.float64)
    valid_counts = np.zeros(n_params, dtype=np.int64)

    presets = sampler.sample_batch(n_base_presets)

    if workers == 0:
        workers = max(1, mp.cpu_count() or 1)

    if workers > 1:
        log.info("Computing importance weights with %d workers", workers)
        from dataclasses import asdict

        config_dict = asdict(config)
        for key, val in config_dict.items():
            if hasattr(val, "__fspath__"):
                config_dict[key] = str(val)

        # Strip unpicklable internal keys from presets
        clean_presets = []
        for p in presets:
            clean = {k: v for k, v in p.items() if not k.startswith("_")}
            clean_presets.append(clean)

        with mp.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(config_dict, continuous_names),
            maxtasksperchild=200,
        ) as pool:
            for dist, valid in tqdm(
                pool.imap_unordered(_worker_process_preset, clean_presets),
                total=len(clean_presets),
                desc="Computing importance weights",
            ):
                total_distances += dist
                valid_counts += valid
    else:
        # Single-process fallback
        for preset in tqdm(presets, desc="Computing importance weights"):
            try:
                original_audio = engine.render_preset(preset, midi_note=60)
            except Exception:
                continue

            if original_audio is None:
                continue

            for i, param_name in enumerate(continuous_names):
                if param_name not in preset:
                    continue

                original_val = preset[param_name]
                perturbed_val = min(original_val + PERTURBATION_DELTA, 1.0)
                if perturbed_val == original_val:
                    perturbed_val = max(original_val - PERTURBATION_DELTA, 0.0)
                if perturbed_val == original_val:
                    continue

                perturbed_preset = dict(preset)
                perturbed_preset[param_name] = perturbed_val

                try:
                    perturbed_audio = engine.render_preset(perturbed_preset, midi_note=60)
                except Exception:
                    continue

                if perturbed_audio is None:
                    continue

                dist = _spectral_distance(original_audio, perturbed_audio)
                total_distances[i] += dist
                valid_counts[i] += 1

    # Average distances
    mask = valid_counts > 0
    weights = np.zeros(n_params, dtype=np.float32)
    weights[mask] = (total_distances[mask] / valid_counts[mask]).astype(np.float32)

    # Normalize so mean weight = 1.0
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights * (n_params / weights.sum())

    log.info(
        "Importance weights computed: min=%.4f, max=%.4f, mean=%.4f",
        weights.min(), weights.max(), weights.mean(),
    )
    return weights
