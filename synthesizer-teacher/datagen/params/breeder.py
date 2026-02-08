"""Derive new presets from a pool of seed presets via crossover, interpolation, and mutation.

Inspired by https://github.com/SlavaCat118/Vinetics (per-parameter genetic
crossover between Vital presets), but adapted for ML dataset generation:
we need high diversity while staying near the manifold of "real" sounds.

Key insight: Vital parameters are organized into **modules** (osc_1, filter_2,
chorus, etc.) that form coherent units of sound design. Crossing parameters
*within* a module across parents produces degenerate sounds. All breeding
strategies therefore operate at the module level — every parameter in a
module comes from the same parent.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from datagen.config import PipelineConfig
from datagen.params.registry import ParamRegistry

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module grouping: maps Vital param names to their parent module.
# ---------------------------------------------------------------------------

# Modules with a numeric suffix: osc_1, filter_2, env_3, lfo_5, random_2…
_NUMBERED_PREFIXES = frozenset({"osc", "filter", "env", "lfo", "random"})

# Stand-alone effect / voice modules (no numeric suffix)
_STANDALONE_MODULES = frozenset({
    "chorus", "compressor", "delay", "distortion", "eq",
    "flanger", "phaser", "reverb", "sample",
})


def _module_of(name: str) -> str:
    """Extract the module prefix from a Vital parameter name.

    Examples::

        osc_1_level       -> osc_1
        filter_fx_cutoff  -> filter_fx
        chorus_dry_wet    -> chorus
        env_3_attack      -> env_3
        volume            -> global
    """
    parts = name.split("_")
    if parts[0] in _NUMBERED_PREFIXES and len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}"
    if parts[0] in _STANDALONE_MODULES:
        return parts[0]
    return "global"


class PresetBreeder:
    """Derives new presets from a pool of seed presets.

    Three breeding strategies (selected randomly per offspring):

    - **Crossover** (40%): Pick 2 parents, assign each *module* to one
      parent so all params within a module stay coherent. Non-scalar data
      (wavetable blobs, LFO shapes) is merged per-module from the correct
      donor.
    - **Interpolation** (30%): Linearly interpolate continuous params
      between 2 parents; categoricals and non-scalar data come from the
      dominant parent.
    - **Mutation** (30%): Clone a single parent and add Gaussian noise
      to continuous params, with per-preset random sigma.
    """

    STRATEGY_WEIGHTS: tuple[float, ...] = (0.4, 0.3, 0.3)

    # Non-scalar keys inherited from a single donor (interpolation / mutation).
    _SINGLE_DONOR_KEYS: tuple[str, ...] = (
        "_lfos", "_wavetables", "_sample", "_modulation_t3",
    )

    def __init__(
        self,
        config: PipelineConfig,
        registry: ParamRegistry,
        seed_presets: list[dict[str, Any]],
        mutation_sigma: float = 0.08,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.seed_presets = seed_presets
        self.mutation_sigma = mutation_sigma
        self.rng = np.random.default_rng(seed)

        self._continuous_names = registry.continuous_names(config.tier)
        self._categorical_names = registry.categorical_names(config.tier)
        self._wavetable_names = registry.wavetable_names(config.tier)
        self._all_param_names = (
            self._continuous_names
            + self._categorical_names
            + self._wavetable_names
        )
        self._continuous_set = frozenset(self._continuous_names)

        # Pre-compute module -> param name grouping (stable).
        self._module_groups: dict[str, list[str]] = {}
        for name in self._all_param_names:
            self._module_groups.setdefault(_module_of(name), []).append(name)

    def breed_batch(self, n: int) -> list[dict[str, Any]]:
        """Breed n new presets from the seed pool."""
        strategies = self.rng.choice(3, size=n, p=self.STRATEGY_WEIGHTS)
        return [self._breed_one(int(s)) for s in strategies]

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _breed_one(self, strategy: int) -> dict[str, Any]:
        if strategy == 0:
            return self._crossover()
        elif strategy == 1:
            return self._interpolate()
        return self._mutate()

    def _crossover(self) -> dict[str, Any]:
        """Module-level crossover between 2 parents.

        For each module (osc_1, filter_2, chorus, …), one parent is chosen
        at random to donate *all* parameters for that module. This preserves
        internal coherence — e.g. an oscillator's wavetable position, level,
        detune, and phase all come from the same parent.

        Non-scalar data (wavetable blobs, LFO shapes) is merged per-module
        so the waveform data matches the scalar params.
        """
        p1, p2 = self._pick_parents(2)

        # Assign each module to a parent
        module_donor: dict[str, dict[str, Any]] = {}
        for module in self._module_groups:
            module_donor[module] = p1 if self.rng.random() < 0.5 else p2

        # Build scalar params from module donors
        preset: dict[str, Any] = {}
        for module, names in self._module_groups.items():
            donor = module_donor[module]
            for name in names:
                if name in donor:
                    preset[name] = donor[name]

        # Merge non-scalar data per-module
        self._merge_nonscalar(preset, module_donor, p1)
        return preset

    def _interpolate(self) -> dict[str, Any]:
        """Linear interpolation of continuous params between 2 parents."""
        p1, p2 = self._pick_parents(2)
        alpha = float(self.rng.uniform(0.15, 0.85))

        preset: dict[str, Any] = {}

        # Continuous: lerp
        for name in self._continuous_names:
            v1 = p1.get(name, 0.5)
            v2 = p2.get(name, 0.5)
            preset[name] = float(np.clip(
                v1 * (1 - alpha) + v2 * alpha, 0.0, 1.0,
            ))

        # Categorical + wavetable: take from the dominant parent
        dominant, secondary = (p1, p2) if alpha < 0.5 else (p2, p1)
        for name in self._categorical_names + self._wavetable_names:
            if name in dominant:
                preset[name] = dominant[name]
            elif name in secondary:
                preset[name] = secondary[name]

        self._inherit_single_donor(preset, dominant)
        return preset

    def _mutate(self) -> dict[str, Any]:
        """Clone a parent and perturb continuous params."""
        (parent,) = self._pick_parents(1)

        # Per-preset sigma drawn from a half-normal around the base sigma
        sigma = abs(self.rng.normal(self.mutation_sigma, self.mutation_sigma * 0.5))

        preset: dict[str, Any] = {}
        for name in self._all_param_names:
            if name not in parent:
                continue
            if name in self._continuous_set:
                noise = self.rng.normal(0, sigma)
                preset[name] = float(np.clip(parent[name] + noise, 0.0, 1.0))
            else:
                preset[name] = parent[name]

        self._inherit_single_donor(preset, parent)

        # Also perturb modulation amounts if present
        mod = preset.get("_modulation_t3")
        if mod is not None:
            connections = [dict(c) for c in mod.get("connections", [])]
            for conn in connections:
                conn["amount"] = float(
                    np.clip(conn["amount"] + self.rng.normal(0, sigma), -1.0, 1.0)
                )
            preset["_modulation_t3"] = {**mod, "connections": connections}

        return preset

    # ------------------------------------------------------------------
    # Non-scalar data merging
    # ------------------------------------------------------------------

    def _inherit_single_donor(
        self, preset: dict[str, Any], donor: dict[str, Any],
    ) -> None:
        """Copy all non-scalar data from a single donor (for interp/mutation)."""
        for key in self._SINGLE_DONOR_KEYS:
            if key in donor:
                preset[key] = donor[key]

    def _merge_nonscalar(
        self,
        preset: dict[str, Any],
        module_donor: dict[str, dict[str, Any]],
        fallback: dict[str, Any],
    ) -> None:
        """Merge non-scalar data from per-module donors (for crossover).

        Each oscillator's wavetable blob comes from the parent that donated
        that oscillator's scalar params. Same for LFO shapes. Modulation
        routing comes from the fallback (base) parent.
        """
        # Wavetables: per-oscillator from osc_N donor
        self._merge_indexed_key(
            preset, "_wavetables",
            [module_donor.get(f"osc_{i + 1}", fallback) for i in range(3)],
        )

        # LFO shapes: per-LFO from lfo_N donor
        self._merge_indexed_key(
            preset, "_lfos",
            [module_donor.get(f"lfo_{i + 1}", fallback) for i in range(8)],
        )

        # Sample data: from the sample module donor
        sample_donor = module_donor.get("sample", fallback)
        if "_sample" in sample_donor:
            preset["_sample"] = sample_donor["_sample"]

        # Modulation routing: from fallback (too complex to merge per-connection)
        if "_modulation_t3" in fallback:
            preset["_modulation_t3"] = fallback["_modulation_t3"]

    @staticmethod
    def _merge_indexed_key(
        preset: dict[str, Any],
        key: str,
        donors: list[dict[str, Any]],
    ) -> None:
        """Merge a list-typed non-scalar key by picking each slot from its donor.

        Only sets the key if at least one donor has data for its slot.
        """
        merged: list[Any] = []
        has_data = False
        for i, donor in enumerate(donors):
            donor_list = donor.get(key, [])
            if i < len(donor_list) and donor_list[i]:
                merged.append(donor_list[i])
                has_data = True
            else:
                merged.append(None)

        if has_data:
            # Fill holes with empty dicts so the list index is correct.
            # The engine will keep the synth's existing data for these slots.
            preset[key] = [entry if entry is not None else {} for entry in merged]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _pick_parents(self, n: int) -> list[dict[str, Any]]:
        """Pick n distinct parents from the seed pool."""
        replace = n > len(self.seed_presets)
        indices = self.rng.choice(len(self.seed_presets), size=n, replace=replace)
        return [self.seed_presets[int(i)] for i in indices]
