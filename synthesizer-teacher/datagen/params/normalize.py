"""Normalize and denormalize parameter values using Vita control ranges."""

from __future__ import annotations

import numpy as np

from datagen.params.registry import ParamInfo, ParamRegistry


def normalize(value: float, info: ParamInfo) -> float:
    """Normalize a raw parameter value to [0, 1]."""
    if info.range == 0:
        return 0.0
    return (value - info.min_val) / info.range


def denormalize(value: float, info: ParamInfo) -> float:
    """Denormalize a [0, 1] value back to the raw parameter range."""
    return value * info.range + info.min_val


def normalize_vector(
    raw_values: dict[str, float],
    registry: ParamRegistry,
    param_names: list[str],
) -> np.ndarray:
    """Normalize a dict of raw param values to a float32 vector in [0, 1].

    Parameters not present in ``raw_values`` default to their ParamInfo default,
    normalized.
    """
    vec = np.zeros(len(param_names), dtype=np.float32)
    for i, name in enumerate(param_names):
        info = registry.get(name)
        if info is None:
            continue
        raw = raw_values.get(name, info.default_val)
        vec[i] = normalize(raw, info)
    return vec


def denormalize_vector(
    norm_values: np.ndarray,
    registry: ParamRegistry,
    param_names: list[str],
) -> dict[str, float]:
    """Denormalize a [0, 1] vector back to a dict of raw parameter values."""
    result: dict[str, float] = {}
    for i, name in enumerate(param_names):
        info = registry.get(name)
        if info is None:
            continue
        result[name] = denormalize(float(norm_values[i]), info)
    return result
