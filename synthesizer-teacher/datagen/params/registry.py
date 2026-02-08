"""Parameter registry: discovers Vital params from a live Vita instance."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from datagen.config import (
    MOD_DEST_BLOCKLIST,
    OPTIONS_CRASH_CONTROLS,
    TIER1_EXTRA_PARAMS,
    TIER1_MODULES,
)

log = logging.getLogger(__name__)


@dataclass
class ParamInfo:
    """Metadata for a single Vital parameter."""

    name: str
    param_type: Literal["continuous", "categorical", "wavetable"]
    min_val: float = 0.0
    max_val: float = 1.0
    default_val: float = 0.0
    n_options: int = 0
    tier: int = 1

    @property
    def range(self) -> float:
        return self.max_val - self.min_val


# Controls excluded from discovery — purely visual or non-sound-affecting.
# Everything else is discovered automatically from Vita's control API.
_EXCLUDE_NAMES: set[str] = {"view_spectrogram"}
_EXCLUDE_SUFFIXES: tuple[str, ...] = ("_view_2d",)


def _should_exclude(name: str) -> bool:
    """Return True if this control should be excluded from discovery.

    Only excludes:
    - ``modulation_*``: per-slot params handled via _modulation_t3
    - ``view_spectrogram``, ``*_view_2d``: purely visual
    """
    if name in _EXCLUDE_NAMES:
        return True
    if name.startswith("modulation_"):
        return True
    for suffix in _EXCLUDE_SUFFIXES:
        if name.endswith(suffix):
            return True
    return False


def _assign_tier(name: str) -> int:
    """Determine which tier a parameter first appears in.

    Tier 1: params belonging to core modules (TIER1_MODULES) or TIER1_EXTRA_PARAMS.
    Tier 2: everything else discovered from Vita.
    (Tier 3 is tier 2 params + modulation matrix — no extra static params.)
    """
    if name in TIER1_EXTRA_PARAMS:
        return 1
    for module in TIER1_MODULES:
        if name.startswith(module + "_"):
            return 1
    return 2


class ParamRegistry:
    """Registry of Vital parameters, built from a live Vita synth.

    Construction via ``from_synth(synth)``: queries a live ``vita.Synth``
    for all control metadata. Parameters are auto-classified as continuous
    or categorical based on their ``options`` list from the Vita API.

    Module on/off switches (``*_on``) are included as regular params
    so they appear in the dataset label and get set during rendering.

    Modulation slot controls (``modulation_N_*``) are excluded — modulation
    routing is handled separately via ``_modulation_t3``.
    """

    def __init__(self) -> None:
        self._params: dict[str, ParamInfo] = {}
        self._mod_sources: list[str] = []
        self._mod_source_index: dict[str, int] = {}
        self._mod_destinations: list[str] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_synth(cls, synth) -> ParamRegistry:
        """Discover parameters from a live vita.Synth instance."""
        import vita

        registry = cls()
        controls = synth.get_controls()

        # Query modulation sources/destinations from Vita itself
        registry._mod_sources = sorted(vita.get_modulation_sources())
        registry._mod_source_index = {s: i for i, s in enumerate(registry._mod_sources)}
        all_destinations = vita.get_modulation_destinations()
        registry._mod_destinations = sorted(
            d for d in all_destinations if d not in MOD_DEST_BLOCKLIST
        )
        log.info(
            "Vita mod sources: %d, destinations: %d (after blocklist)",
            len(registry._mod_sources), len(registry._mod_destinations),
        )

        skipped = 0
        for name in controls:
            if _should_exclude(name):
                skipped += 1
                continue

            try:
                details = synth.get_control_details(name)
            except Exception:
                log.warning("Could not get details for control %s, skipping", name)
                continue

            min_val = float(details.min)
            max_val = float(details.max)
            default_val = float(details.default_value)

            # details.options segfaults for some controls (e.g. filter_*_style)
            if name in OPTIONS_CRASH_CONTROLS:
                n_options = int(max_val - min_val) + 1
            else:
                n_options = len(details.options)

            if n_options > 0:
                param_type = "categorical"
            else:
                param_type = "continuous"

            info = ParamInfo(
                name=name,
                param_type=param_type,
                min_val=min_val,
                max_val=max_val,
                default_val=default_val,
                n_options=n_options,
                tier=_assign_tier(name),
            )
            registry._params[name] = info

        log.info(
            "Registry: %d params (%d continuous, %d categorical, %d excluded)",
            len(registry._params),
            sum(1 for p in registry._params.values() if p.param_type == "continuous"),
            sum(1 for p in registry._params.values() if p.param_type == "categorical"),
            skipped,
        )
        return registry

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, name: str) -> ParamInfo | None:
        return self._params.get(name)

    def __getitem__(self, name: str) -> ParamInfo:
        return self._params[name]

    def __len__(self) -> int:
        return len(self._params)

    def __contains__(self, name: str) -> bool:
        return name in self._params

    def continuous(self, max_tier: int = 3) -> list[ParamInfo]:
        return sorted(
            [p for p in self._params.values() if p.param_type == "continuous" and p.tier <= max_tier],
            key=lambda p: p.name,
        )

    def categorical(self, max_tier: int = 3) -> list[ParamInfo]:
        return sorted(
            [p for p in self._params.values() if p.param_type == "categorical" and p.tier <= max_tier],
            key=lambda p: p.name,
        )

    def wavetable(self, max_tier: int = 3) -> list[ParamInfo]:
        return sorted(
            [p for p in self._params.values() if p.param_type == "wavetable" and p.tier <= max_tier],
            key=lambda p: p.name,
        )

    def for_tier(self, tier: int) -> list[ParamInfo]:
        return [p for p in self._params.values() if p.tier <= tier]

    @property
    def all_params(self) -> dict[str, ParamInfo]:
        return dict(self._params)

    def continuous_names(self, max_tier: int = 3) -> list[str]:
        return [p.name for p in self.continuous(max_tier)]

    def categorical_names(self, max_tier: int = 3) -> list[str]:
        return [p.name for p in self.categorical(max_tier)]

    def wavetable_names(self, max_tier: int = 3) -> list[str]:
        if max_tier < 2:
            return []
        return [p.name for p in self.wavetable(max_tier)]

    # ------------------------------------------------------------------
    # Modulation (data-driven from Vita API)
    # ------------------------------------------------------------------

    @property
    def mod_sources(self) -> list[str]:
        return self._mod_sources

    @property
    def mod_source_index(self) -> dict[str, int]:
        return self._mod_source_index

    @property
    def n_mod_sources(self) -> int:
        return len(self._mod_sources)

    def mod_destinations(self, max_tier: int = 3) -> list[str]:
        return list(self._mod_destinations)

    # ------------------------------------------------------------------
    # Wavetable params (virtual, not in Vita controls)
    # ------------------------------------------------------------------

    def add_wavetable_params(self, n_wavetables: int) -> None:
        """Add virtual wavetable categorical params for each oscillator.

        These don't exist as Vita controls — they're tracked by the pipeline
        to select which factory wavetable to inject via JSON.
        """
        for osc in ["osc_1", "osc_2", "osc_3"]:
            name = f"{osc}_wavetable"
            if name not in self._params:
                self._params[name] = ParamInfo(
                    name=name,
                    param_type="wavetable",
                    min_val=0.0,
                    max_val=float(n_wavetables - 1),
                    default_val=0.0,
                    n_options=n_wavetables,
                    tier=2,
                )
