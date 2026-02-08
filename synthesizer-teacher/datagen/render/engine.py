"""Render engine: wraps vita.Synth for preset rendering."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from datagen.config import MOD_DEST_BLOCKLIST, OPTIONS_CRASH_CONTROLS, PipelineConfig

log = logging.getLogger(__name__)


class RenderEngine:
    """Wraps a vita.Synth instance for rendering presets to audio.

    Every preset — synthetic or community — is rendered through the same
    param-by-param control API so the dataset label exactly matches what
    was sent to the synth:

        1. Reset to init state
        2. Set every scalar param via ``ctrl.set(raw_value)``
        3. Inject non-scalar data (wavetables, LFO shapes, samples) via JSON
        4. Apply modulation connections via ``connect_modulation()``
        5. Render the MIDI note

    Continuous params are stored as linear-normalized [0,1] in the preset
    dict and denormalized here using each control's min/max range. Categorical
    params are stored as integer indices and set directly.
    """

    def __init__(self, config: PipelineConfig) -> None:
        import vita

        self.config = config
        self.synth = vita.Synth()
        self._default_json = self.synth.to_json()

        # Cache control metadata (built once from Vita API)
        self._control_ranges: dict[str, tuple[float, float]] = {}
        self._categorical_names: set[str] = set()
        for name in self.synth.get_controls():
            try:
                details = self.synth.get_control_details(name)
                lo, hi = float(details.min), float(details.max)
                self._control_ranges[name] = (lo, hi)
                # details.options segfaults for some controls — infer
                # categoricality from integer range instead.
                if name in OPTIONS_CRASH_CONTROLS:
                    if hi > lo and hi == float(int(hi)):
                        self._categorical_names.add(name)
                elif len(details.options) > 0:
                    self._categorical_names.add(name)
            except (RuntimeError, ValueError):
                pass

        # Cache controls dict for reuse in _apply_params and _apply_modulation
        self._controls = self.synth.get_controls()

    def reset(self) -> None:
        """Reset synth to default init state."""
        self.synth.load_json(self._default_json)

    def render_preset(
        self,
        preset: dict[str, Any],
        midi_note: int = 60,
        midi_velocity: int | None = None,
    ) -> np.ndarray | None:
        """Render a single preset at a given MIDI note.

        All presets follow the same param-by-param rendering flow to ensure
        the dataset label accurately reflects what was sent to the synth.

        Args:
            preset: Parameter dict. Continuous values are normalized [0,1],
                    categoricals are int indices, internal keys start with ``_``.
            midi_note: MIDI note number to render.
            midi_velocity: MIDI velocity (0-127). Defaults to config.midi_velocity.

        Returns:
            Stereo audio array of shape (2, n_samples) or None if rendering fails.
        """
        velocity = midi_velocity if midi_velocity is not None else self.config.midi_velocity

        try:
            self.reset()
            self._apply_params(preset)
            self._inject_wavetables(preset)
            self._inject_nonscalar_data(preset)
            self._apply_modulation(preset)

            audio = self._render_note(midi_note, velocity)
            return audio
        except (RuntimeError, ValueError, OSError) as e:
            log.warning("Render failed for note %d: %s", midi_note, e)
            return None

    def render_preset_multi_note(
        self,
        preset: dict[str, Any],
        midi_notes: list[int] | None = None,
    ) -> list[tuple[int, np.ndarray | None]]:
        """Render a preset at multiple MIDI notes."""
        notes = midi_notes or self.config.midi_notes
        results = []
        for note in notes:
            audio = self.render_preset(preset, midi_note=note)
            results.append((note, audio))
        return results

    def _apply_params(self, preset: dict[str, Any]) -> None:
        """Set every param via ctrl.set(raw_value).

        Param type is determined by ``_categorical_names`` (built from Vita's
        control metadata at init), not by the Python type of the value.

        Continuous params are denormalized from [0,1] using cached control
        ranges: ``raw = min + norm * (max - min)``.
        Categorical params are set as raw indices directly.
        """
        controls = self._controls

        for name, value in preset.items():
            if name.startswith("_"):
                continue
            if "wavetable" in name:
                continue
            if name not in controls:
                continue

            ctrl = controls[name]
            if name in self._categorical_names:
                lo, _ = self._control_ranges.get(name, (0.0, 1.0))
                ctrl.set(float(value) + lo)
            else:
                lo, hi = self._control_ranges.get(name, (0.0, 1.0))
                raw = lo + float(value) * (hi - lo)
                ctrl.set(raw)

    def _inject_wavetables(self, preset: dict[str, Any]) -> None:
        """Inject wavetable data from catalog via JSON manipulation."""
        wt_catalog = preset.get("_wavetable_catalog")
        if wt_catalog is None:
            return

        has_wt = any(f"osc_{i}_wavetable" in preset for i in (1, 2, 3))
        if not has_wt:
            return

        synth_json = self.synth.to_json()
        try:
            synth_data = json.loads(synth_json)
        except json.JSONDecodeError:
            log.warning("Could not parse synth JSON for wavetable injection")
            return

        settings = synth_data.get("settings", synth_data)
        wavetables = settings.get("wavetables", [])

        for param_name in ["osc_1_wavetable", "osc_2_wavetable", "osc_3_wavetable"]:
            if param_name not in preset:
                continue
            wt_idx = preset[param_name]
            osc_idx = int(param_name.split("_")[1]) - 1
            if wt_idx < len(wt_catalog):
                wt_entry = wt_catalog.get_by_index(wt_idx)
                wt_data = json.loads(wt_entry.base64_data)
                while len(wavetables) <= osc_idx:
                    wavetables.append({})
                wavetables[osc_idx] = wt_data

        settings["wavetables"] = wavetables
        synth_data["settings"] = settings
        self.synth.load_json(json.dumps(synth_data))

    def _inject_nonscalar_data(self, preset: dict[str, Any]) -> None:
        """Inject non-scalar preset data (LFO shapes, sample, wavetables) via JSON.

        Community presets store LFO waveform shapes in ``_lfos``, sample audio
        data in ``_sample``, and wavetable definitions in ``_wavetables``.
        These can't be set via the control API — they must be injected into
        the synth's JSON state.

        For list-typed keys (``_lfos``, ``_wavetables``), individual slots
        are merged into the existing synth state rather than replacing the
        entire list, so catalog-injected or default data is preserved for
        slots that have no override.
        """
        has_data = any(preset.get(k) for k in ("_lfos", "_sample", "_wavetables"))
        if not has_data:
            return

        synth_json = self.synth.to_json()
        try:
            synth_data = json.loads(synth_json)
        except json.JSONDecodeError:
            log.warning("Could not parse synth JSON for non-scalar injection")
            return

        settings = synth_data.get("settings", synth_data)

        # Scalar key: replace outright
        sample = preset.get("_sample")
        if sample:
            settings["sample"] = sample

        # List-typed keys: merge per-slot so empty dicts don't clobber
        # existing data (e.g. catalog-injected wavetables).
        for preset_key, settings_key in (("_lfos", "lfos"), ("_wavetables", "wavetables")):
            src = preset.get(preset_key)
            if not src:
                continue
            existing = settings.get(settings_key, [])
            for i, entry in enumerate(src):
                if entry:  # skip empty / None slots
                    while len(existing) <= i:
                        existing.append({})
                    existing[i] = entry
            settings[settings_key] = existing

        synth_data["settings"] = settings
        self.synth.load_json(json.dumps(synth_data))

    def _apply_modulation(self, preset: dict[str, Any]) -> None:
        """Apply modulation connections for tier 3 presets."""
        self.synth.clear_modulations()

        if self.config.tier < 3:
            return

        mod_data = preset.get("_modulation_t3")
        if mod_data is None:
            return

        connections = mod_data.get("connections", [])
        slot = 0
        conn_params: list[tuple[str, dict]] = []
        for conn in connections:
            source = conn["source"]
            dest = conn["destination"]

            if dest in MOD_DEST_BLOCKLIST:
                continue

            try:
                self.synth.connect_modulation(source, dest)
                slot += 1

                prefix = f"modulation_{slot}"
                conn_params.append((prefix, conn))
            except Exception as e:
                log.debug("Failed to apply mod connection %s->%s: %s", source, dest, e)

        # Fetch controls once after all connections are made
        if conn_params:
            controls = self.synth.get_controls()
            for prefix, conn in conn_params:
                if f"{prefix}_amount" in controls:
                    controls[f"{prefix}_amount"].set(conn["amount"])
                if f"{prefix}_bipolar" in controls:
                    controls[f"{prefix}_bipolar"].set(conn.get("bipolar", 0.0))
                if f"{prefix}_power" in controls:
                    controls[f"{prefix}_power"].set(conn.get("power", 0.0))
                if f"{prefix}_stereo" in controls:
                    controls[f"{prefix}_stereo"].set(conn.get("stereo", 0.0))

    def _render_note(self, midi_note: int, velocity: int) -> np.ndarray:
        """Render a single MIDI note and return stereo audio."""
        velocity_float = velocity / 127.0
        audio = self.synth.render(
            midi_note,
            velocity_float,
            self.config.note_duration,
            self.config.render_duration,
        )
        return audio.astype(np.float32)
