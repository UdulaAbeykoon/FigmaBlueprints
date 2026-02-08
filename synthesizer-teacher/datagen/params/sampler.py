"""Parameter sampling: Latin Hypercube for continuous, uniform for categoricals."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats.qmc import LatinHypercube

from datagen.config import PipelineConfig
from datagen.params.registry import ParamRegistry

log = logging.getLogger(__name__)


class ParamSampler:
    """Samples random parameter vectors for Vital preset generation.

    Uses Latin Hypercube Sampling (LHS) for continuous params to ensure
    good coverage of the parameter space, and uniform random for categoricals.
    """

    def __init__(
        self,
        config: PipelineConfig,
        registry: ParamRegistry,
        n_wavetables: int = 25,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.n_wavetables = n_wavetables
        self.rng = np.random.default_rng(seed)

        self._continuous_names = registry.continuous_names(config.tier)
        self._categorical_names = registry.categorical_names(config.tier)
        self._wavetable_names = registry.wavetable_names(config.tier)

    def sample_batch(self, n: int) -> list[dict[str, Any]]:
        """Sample a batch of n parameter dictionaries.

        Uses LHS for continuous params to ensure better space coverage
        than pure random sampling.
        """
        n_cont = len(self._continuous_names)

        # LHS gives values in [0, 1] for each dimension
        if n_cont > 0:
            lhs = LatinHypercube(d=n_cont, seed=self.rng)
            lhs_samples = lhs.random(n=n)  # shape (n, n_cont)
        else:
            lhs_samples = np.empty((n, 0))

        presets: list[dict[str, Any]] = []
        for i in range(n):
            preset = self._build_preset(lhs_samples[i])
            presets.append(preset)

        return presets

    def _build_preset(self, lhs_row: np.ndarray) -> dict[str, Any]:
        """Build a single preset dict from an LHS sample row."""
        preset: dict[str, Any] = {}

        # Continuous params: full [0, 1] range
        for j, name in enumerate(self._continuous_names):
            preset[name] = float(lhs_row[j])

        # Categorical params: uniform random over options
        for name in self._categorical_names:
            info = self.registry.get(name)
            n_opts = info.n_options if info else 8
            preset[name] = int(self.rng.integers(0, n_opts))

        # Wavetable params: uniform random over catalog
        for name in self._wavetable_names:
            preset[name] = int(self.rng.integers(0, self.n_wavetables))

        # Heuristic constraints to reduce silent/degenerate presets.
        # osc_1 must be on and audible; volume in a safe range.
        if "osc_1_on" in preset:
            preset["osc_1_on"] = 1
        if "osc_1_level" in preset:
            preset["osc_1_level"] = max(0.3, preset["osc_1_level"])
        if "volume" in preset:
            preset["volume"] = 0.55 + preset["volume"] * 0.30  # map to [0.55, 0.85]

        # Tier 3 modulation
        if self.config.tier >= 3:
            preset["_modulation_t3"] = self._sample_tier3_modulation()

        return preset

    def _sample_tier3_modulation(self) -> dict[str, Any]:
        """Sample Tier 3 sparse modulation matrix.

        Number of connections drawn from Geometric(p=0.15), clamped to [0, 20].
        Each connection is a random (source, destination, amount, bipolar,
        power, stereo) tuple.
        """
        mod_sources = self.registry.mod_sources
        destinations = self.registry.mod_destinations(max_tier=3)
        n_src = len(mod_sources)
        n_dst = len(destinations)

        # Number of active connections
        n_connections = min(int(self.rng.geometric(0.15)), 20)

        connections: list[dict[str, Any]] = []
        for _ in range(n_connections):
            src_idx = int(self.rng.integers(0, n_src))
            dst_idx = int(self.rng.integers(0, n_dst))
            connections.append({
                "source": mod_sources[src_idx],
                "destination": destinations[dst_idx],
                "source_idx": src_idx,
                "dest_idx": dst_idx,
                "amount": float(self.rng.uniform(-1.0, 1.0)),
                "bipolar": float(self.rng.integers(0, 2)),
                "power": float(self.rng.uniform(0.0, 1.0)),
                "stereo": float(self.rng.integers(0, 2)),
            })

        return {
            "connections": connections,
            "n_sources": n_src,
            "n_destinations": n_dst,
        }
