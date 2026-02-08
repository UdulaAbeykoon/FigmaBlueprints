"""Standard LFO shapes for Vital synthesizer.

Defines 8 standard LFO shapes as JSON-serializable point arrays compatible
with Vita's LFO format. Each shape is represented as a list of
(x, y, power) tuples defining control points for Vital's LFO curve editor.

Vital's LFO format uses:
    - ``num_points``: number of control points
    - ``points``: flat list of (x, y, power) triples
    - ``powers``: flat list of curve powers between points
    - ``smooth``: boolean for cubic interpolation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Shape indices (stable ordering)
SHAPE_SINE = 0
SHAPE_TRIANGLE = 1
SHAPE_SAW_UP = 2
SHAPE_SAW_DOWN = 3
SHAPE_SQUARE = 4
SHAPE_RANDOM_SH = 5
SHAPE_SMOOTH_RANDOM = 6
SHAPE_FLAT = 7

N_LFO_SHAPES = 8

SHAPE_NAMES = [
    "sine",
    "triangle",
    "saw_up",
    "saw_down",
    "square",
    "random_sh",
    "smooth_random",
    "flat",
]


@dataclass
class LFOShapeEntry:
    """A single LFO shape definition."""

    index: int
    name: str
    points: list[tuple[float, float, float]]  # (x, y, power)
    smooth: bool = False


def _make_sine() -> list[tuple[float, float, float]]:
    """Sine wave approximated with control points and power curves."""
    return [
        (0.0, 0.5, 0.0),
        (0.25, 1.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.75, 0.0, 0.0),
        (1.0, 0.5, 0.0),
    ]


def _make_triangle() -> list[tuple[float, float, float]]:
    """Triangle wave: linear ramps up and down."""
    return [
        (0.0, 0.0, 0.0),
        (0.5, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    ]


def _make_saw_up() -> list[tuple[float, float, float]]:
    """Saw up: linear ramp from 0 to 1."""
    return [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
    ]


def _make_saw_down() -> list[tuple[float, float, float]]:
    """Saw down: linear ramp from 1 to 0."""
    return [
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    ]


def _make_square() -> list[tuple[float, float, float]]:
    """Square wave: instant transitions at 50% duty cycle."""
    return [
        (0.0, 1.0, 0.0),
        (0.5, 1.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ]


def _make_random_sh() -> list[tuple[float, float, float]]:
    """Random sample-and-hold: step function (4 steps as template)."""
    return [
        (0.0, 0.2, 0.0),
        (0.25, 0.2, 0.0),
        (0.25, 0.8, 0.0),
        (0.5, 0.8, 0.0),
        (0.5, 0.4, 0.0),
        (0.75, 0.4, 0.0),
        (0.75, 0.9, 0.0),
        (1.0, 0.9, 0.0),
    ]


def _make_smooth_random() -> list[tuple[float, float, float]]:
    """Smooth random: gentle curves between random levels."""
    return [
        (0.0, 0.5, 0.0),
        (0.2, 0.8, 0.0),
        (0.4, 0.3, 0.0),
        (0.6, 0.7, 0.0),
        (0.8, 0.2, 0.0),
        (1.0, 0.5, 0.0),
    ]


def _make_flat() -> list[tuple[float, float, float]]:
    """Flat: constant value (no modulation effect)."""
    return [
        (0.0, 0.5, 0.0),
        (1.0, 0.5, 0.0),
    ]


_SHAPE_BUILDERS = [
    _make_sine,
    _make_triangle,
    _make_saw_up,
    _make_saw_down,
    _make_square,
    _make_random_sh,
    _make_smooth_random,
    _make_flat,
]


class LFOShapeCatalog:
    """Catalog of standard LFO shapes for dataset generation and inference.

    Usage::

        catalog = LFOShapeCatalog()
        shape_json = catalog.to_vital_json(SHAPE_SINE)
        # -> dict suitable for injection into Vital's LFO JSON slot
    """

    def __init__(self) -> None:
        self.shapes: list[LFOShapeEntry] = []
        for i, (name, builder) in enumerate(zip(SHAPE_NAMES, _SHAPE_BUILDERS)):
            smooth = name in ("sine", "smooth_random")
            self.shapes.append(
                LFOShapeEntry(
                    index=i,
                    name=name,
                    points=builder(),
                    smooth=smooth,
                )
            )

    def __len__(self) -> int:
        return len(self.shapes)

    def __getitem__(self, index: int) -> LFOShapeEntry:
        return self.shapes[index]

    def get_by_name(self, name: str) -> LFOShapeEntry:
        """Get shape by name."""
        for shape in self.shapes:
            if shape.name == name:
                return shape
        raise KeyError(f"Unknown LFO shape: {name!r}")

    def to_vital_json(self, shape_index: int) -> dict:
        """Convert a shape to Vital's JSON LFO format.

        Args:
            shape_index: Index into the catalog [0..7].

        Returns:
            Dict with ``num_points``, ``points``, ``powers``, ``smooth``
            keys matching Vital's internal LFO representation.
        """
        entry = self.shapes[shape_index]
        n = len(entry.points)

        # Vital stores points as flat array: [x0, y0, power0, x1, y1, power1, ...]
        flat_points = []
        for x, y, power in entry.points:
            flat_points.extend([x, y, power])

        # Powers between points (n-1 values)
        powers = [0.0] * max(n - 1, 0)

        # Sine uses smooth interpolation
        return {
            "num_points": n,
            "points": flat_points,
            "powers": powers,
            "smooth": entry.smooth,
        }

    def name_for_index(self, index: int) -> str:
        """Get shape name for an index."""
        if 0 <= index < len(self.shapes):
            return self.shapes[index].name
        return f"unknown_{index}"

    def index_for_name(self, name: str) -> int:
        """Get index for a shape name."""
        for shape in self.shapes:
            if shape.name == name:
                return shape.index
        raise KeyError(f"Unknown LFO shape: {name!r}")
