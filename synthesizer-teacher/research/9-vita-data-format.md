# Report 9: Vita Data Format Reference

Complete reference for the Vita Python API (v0.0.5), the `.vital` JSON preset format, and how the project uses them.

**Source**: https://github.com/DBraun/Vita (David Braun, nanobind-based Python bindings to Vital's C++ `HeadlessSynth`).

---

## 1. Module-Level API

```python
import vita
from vita import Synth, constants, get_modulation_sources, get_modulation_destinations
```

| Function | Signature | Returns | Description |
|----------|-----------|---------|-------------|
| `Synth()` | `() -> Synth` | Synth instance | Creates a new synthesizer (init preset loaded) |
| `get_modulation_sources()` | `() -> list[str]` | 32 source names | All available modulation sources |
| `get_modulation_destinations()` | `() -> list[str]` | 428 destination names | All available modulation destinations |

---

## 2. Synth Class

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `render()` | `(midi_note: int, velocity: float, note_dur: float, render_dur: float) -> ndarray` | `float32 (2, N)` | Render audio. Velocity is [0, 1] NOT [0, 127]. N = sample_rate * render_dur. |
| `render_file()` | `(path: str, midi_note: int, velocity: float, note_dur: float, render_dur: float) -> bool` | True/False | Render directly to WAV file |
| `set_bpm()` | `(bpm: float) -> None` | None | Set BPM for tempo-synced params |
| `get_controls()` | `() -> dict[str, ControlValue]` | Dict of 772 controls | All parameter controls (cached dict, fast) |
| `get_control_details()` | `(name: str) -> ControlInfo` | Metadata object | Get param metadata (min, max, scale, options) |
| `get_control_text()` | `(name: str) -> str` | Display string | Current formatted display text for a control |
| `connect_modulation()` | `(source: str, dest: str) -> bool` | True/False | Create modulation routing (fills next empty slot) |
| `disconnect_modulation()` | `(source: str, dest: str) -> None` | None | Remove specific modulation routing |
| `clear_modulations()` | `() -> None` | None | Remove all modulation connections |
| `to_json()` | `() -> str` | JSON string | Serialize entire synth state (= .vital file format) |
| `load_json()` | `(json: str) -> bool` | True/False | Restore synth state from JSON |
| `load_preset()` | `(filepath: str) -> bool` | True/False | Load a .vital file from disk |
| `load_init_preset()` | `() -> None` | None | Reset to default init preset |
| `__getstate__()` | `() -> str` | JSON string | Pickle support |
| `__setstate__()` | `(json: str) -> None` | None | Pickle support |

**Notes:**
- v0.0.5 has NO `set_sample_rate()` -- fixed at 44100 Hz. Later versions add it.
- Rendering is **non-deterministic** due to oscillator phase randomization. Use spectral comparison, not waveform correlation.
- Velocity MUST be [0, 1]. The project divides by 127.0: `velocity / 127.0`.

---

## 3. ControlValue Class

Returned by `synth.get_controls()["param_name"]`.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `value()` | `() -> float` | Raw value | Current raw (denormalized) value |
| `set()` | `(value: float) -> None` | None | Set raw value directly. Also accepts enum constants. |
| `get_normalized()` | `() -> float` | [0, 1] | Current value as VST-style normalized value |
| `set_normalized()` | `(value: float) -> None` | None | Set from [0, 1] normalized value. **DANGEROUS -- see below.** |
| `get_text()` | `() -> str` | Display string | Formatted display text |

### CRITICAL: Why We Never Use `set_normalized()`

`set_normalized()` in the Python binding applies non-linear ValueScale transformations for some parameter types. This is NOT a simple linear interpolation:

| ValueScale | `set_normalized(n)` maps to | Linear equivalent |
|------------|------------------------------|-------------------|
| **Linear** | `min + n * (max - min)` | Same (safe) |
| **Indexed** | `round(min + n * (max - min))` | Same (safe) |
| **Quadratic** | `min + sqrt(n) * (max - min)` | **Different!** |
| **Quartic** | `min + n^(1/4) * (max - min)` | **Different!** |
| **Exponential** | Non-linear (2^x based, may be inverted) | **Completely different!** |
| **SquareRoot** | Non-linear mapping | **Different** |

**Concrete examples:**
- `osc_1_level` (Quadratic, range [0, 1]): `set_normalized(0.25)` -> `value() = 0.5` (not 0.25)
- `chorus_frequency` (Exponential, range [-6, 3]): `set_normalized(0.0)` -> `value() = 3.0` (max! direction reversed!)

**This is why the project uses `ctrl.set(raw)` with manual linear normalization:**
```python
# Correct: manual linear denormalization
raw = lo + normalized * (hi - lo)
ctrl.set(raw)

# WRONG: lossy roundtrip for ~72 non-linear params
ctrl.set_normalized(normalized)
```

### C++ Architecture: ValueBridge vs Display Skewing

Source: [value_bridge.h](https://github.com/mtytel/vital/blob/c0694a193777fc97853a598f86378bea625a6d81/src/plugin/value_bridge.h)

Vital's C++ code has **two separate normalization layers**:

**1. Plugin parameter normalization (ValueBridge) -- ALWAYS LINEAR:**
```cpp
// Used for VST/AU host communication and JSON storage
float convertToPluginValue(float synth_value) const {
    return (synth_value - details_.min) / span_;  // linear
}
float convertToEngineValue(float plugin_value) const {
    float value = plugin_value * span_ + details_.min;  // linear
    if (value_scale == kIndexed) return std::round(value);
    return value;
}
```

**2. Display/knob skewing (private helpers) -- NON-LINEAR:**
```cpp
float skewValue(float value) const {
    switch (details_.value_scale) {
        case kQuadratic:   return value * value;            // x^2
        case kCubic:       return value * value * value;    // x^3
        case kQuartic:     value *= value; return value * value; // x^4
        case kExponential: return powf(2.0f, value);        // 2^x (or 1/2^x if inverted)
        case kSquareRoot:  return sqrtf(value);             // sqrt(x)
        default:           return value;                    // linear/indexed
    }
}
float unskewValue(float value) const {
    switch (details_.value_scale) {
        case kQuadratic:   return sqrtf(value);             // sqrt(x)
        case kCubic:       return powf(value, 1.0f/3.0f);  // cbrt(x)
        case kQuartic:     return powf(value, 1.0f/4.0f);  // x^(1/4)
        case kExponential: return log2(value);              // log2(x) (or log2(1/x))
        default:           return value;
    }
}
```

**Key insight**: The `.vital` JSON stores raw values. The plugin's `convertToEngineValue` is pure linear. The skewing is a display/knob UI layer that Vita's Python binding appears to use in `set_normalized()` / `get_normalized()`, making them non-linear for ~72 params. Our project's manual linear normalization (`raw = min + norm * span`) matches the plugin's actual storage representation.

---

## 4. ControlInfo Class

Returned by `synth.get_control_details("param_name")`.

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Parameter name (e.g., `"filter_1_cutoff"`) |
| `min` | `float` | Minimum raw value |
| `max` | `float` | Maximum raw value |
| `default_value` | `float` | Default raw value |
| `scale` | `ValueScale` | Scaling type (see enum below) |
| `display_multiply` | `float` | Multiplier for display formatting |
| `post_offset` | `float` | Offset added after scaling for display |
| `display_name` | `str` | Human-readable name (e.g., `"Oscillator 1 Level"`) |
| `display_units` | `str` | Unit string (e.g., `" semitones"`) |
| `is_discrete` | `bool` | True for categorical/indexed params |
| `options` | `list[str]` | Option labels for discrete params (empty for continuous) |
| `version_added` | `int` | Vital version this param was added in |

### Segfault Warning: OPTIONS_CRASH_CONTROLS

Accessing `details.options` **crashes Vita with a segfault** for 7 controls:

```python
OPTIONS_CRASH_CONTROLS = {
    "filter_1_style", "filter_2_style", "filter_fx_style",
    "osc_1_view_2d", "osc_2_view_2d", "osc_3_view_2d",
    "view_spectrogram",
}
```

**Workaround:** Infer categoricality from integer range instead:
```python
if name in OPTIONS_CRASH_CONTROLS:
    if hi > lo and hi == float(int(hi)):
        is_categorical = True
else:
    is_categorical = len(details.options) > 0
```

---

## 5. Constants and Enums (`vita.constants`)

### ValueScale

```python
class ValueScale:
    Indexed = 0       # Discrete integer params
    Linear = 1        # Simple linear interpolation
    Quadratic = 2     # x^2 skewing -- 12 params
    Cubic = 3         # Not used in practice
    Quartic = 4       # x^4 skewing, envelope times -- 30 params
    SquareRoot = 5    # sqrt(x) skewing, volume only -- 1 param
    Exponential = 6   # 2^x skewing, frequencies/times -- 30 params
```

### Scale Distribution Across Non-Modulation Controls

| Scale | Count | Examples |
|-------|-------|---------|
| Linear | 221 | Filter cutoffs, dry/wet, feedback, most continuous params |
| Indexed | 159 | All `*_on` switches, model selectors, categorical params |
| Quartic | 30 | All envelope times (attack, decay, delay, hold, release x 6 envelopes) |
| Exponential | 30 | LFO/random frequencies, chorus/flanger/phaser frequency, portamento, reverb decay |
| Quadratic | 12 | `osc_{1,2,3}_level`, `osc_{1,2,3}_unison_detune`, `eq_{low,band,high}_resonance`, `reverb_chorus_amount`, `sub_level`, `sample_level` |
| SquareRoot | 1 | `volume` (range [0, 7399.44]) |

### Synthesizer Enums

These can be passed directly to `ctrl.set()`:

| Enum | Values |
|------|--------|
| `FilterModel` | Analog=0, Dirty=1, Ladder=2, Digital=3, Diode=4, Formant=5, Comb=6, Phase=7 |
| `SynthFilterStyle` | k12Db=0, k24Db=1, NotchPassSwap=2, DualNotchBand=3, BandPeakNotch=4, Shelving=5 |
| `SpectralMorph` | NoSpectralMorph=0, Vocode=1, FormScale=2, HarmonicScale=3, InharmonicScale=4, Smear=5, RandomAmplitudes=6, LowPass=7, HighPass=8, PhaseDisperse=9, ShepardTone=10, Skew=11 |
| `DistortionType` | None=0, Sync=1, Formant=2, Quantize=3, Bend=4, Squeeze=5, PulseWidth=6, FmOscillatorA=7, FmOscillatorB=8, FmSample=9, RmOscillatorA=10, RmOscillatorB=11, RmSample=12 |
| `UnisonStackType` | Normal=0 through OddHarmonicSeries=10 |
| `WaveShape` | Sin=0, SaturatedSin=1, Triangle=2, Square=3, Pulse=4, Saw=5 |
| `RetriggerStyle` | Free=0, Retrigger=1, SyncToPlayHead=2 |
| `SyncedFrequency` | k32_1=0 through k1_64=11 (tempo divisions) |
| `SynthLFOSyncOption` | Time=0, Tempo=1, DottedTempo=2, TripletTempo=3, Keytrack=4 |
| `SynthLFOSyncType` | Trigger=0, Sync=1, Envelope=2, SustainEnvelope=3, LoopPoint=4, LoopHold=5 |
| `RandomLFOStyle` | Perlin=0, SampleAndHold=1, SinInterpolate=2, LorenzAttractor=3 |
| `SourceDestination` | Filter1=0, Filter2=1, DualFilters=2, Effects=3, DirectOut=4 |
| `Effect` | Chorus=0, Compressor=1, Delay=2, Distortion=3, Eq=4, FilterFx=5, Flanger=6, Phaser=7, Reverb=8 |
| `VoicePriority` | Newest=0, Oldest=1, Highest=2, Lowest=3, RoundRobin=4 |
| `VoiceOverride` | Kill=0, Steal=1 |
| `CompressorBandOption` | Multiband=0, LowBand=1, HighBand=2, SingleBand=3 |

---

## 6. Control Parameter Space

### Overview: 772 Total Controls

| Category | Count | Description |
|----------|-------|-------------|
| Non-modulation controls | 452 | Oscillators, filters, envelopes, LFOs, effects, global |
| Modulation slot controls | 320 | 64 slots x 5 params (amount, bipolar, bypass, power, stereo) |
| **Total** | **772** | |

### ParamRegistry Classification (448 in registry)

The project's `ParamRegistry` auto-discovers all 772 controls from a live Synth and classifies them:

| Category | Count | In Registry | Description |
|----------|-------|-------------|-------------|
| Continuous | 322 | Yes | Params with non-integer ranges |
| Categorical | 126 | Yes | Params with integer options (including `*_on` switches) |
| Modulation slot params | 320 | No | Handled separately via `_modulation_t3` dense matrix |
| View-only params | 4 | No | `view_spectrogram`, `osc_{1,2,3}_view_2d` (excluded) |
| **Total** | **772** | **448** | |

### Modulation Slot Controls (5 per slot, 64 slots = 320 total)

| Control | Range | Scale | Type | Mod Destination? |
|---------|-------|-------|------|-----------------|
| `modulation_N_amount` | [-1, 1] | Linear | Continuous | Yes |
| `modulation_N_bipolar` | [0, 1] | Indexed | Discrete (on/off) | No |
| `modulation_N_bypass` | [0, 1] | Indexed | Discrete (on/off) | No |
| `modulation_N_power` | [-10, 10] | Linear | Continuous | Yes |
| `modulation_N_stereo` | [0, 1] | Indexed | Discrete (on/off) | No |

All 320 modulation slot controls exist in `get_controls()` at all times, regardless of whether connections are active. They do NOT dynamically appear/disappear.

### Controls NOT Available as Modulation Destinations

344 of 772 controls cannot be modulation destinations. Key categories:
- All `*_on` switches (module enables)
- All `*_sync`, `*_sync_type`, `*_style`, `*_model`, `*_type` selectors
- All envelope curve params (`*_attack_power`, `*_decay_power`, `*_release_power`)
- All modulation slot `bipolar`, `bypass`, `stereo` params
- Various: `bypass`, `legato`, `mpe_enabled`, `oversampling`, `voice_override`, `voice_priority`, etc.

The 428 modulation destinations include all "musically useful" continuous params plus `modulation_N_amount` and `modulation_N_power` (allowing modulations of modulations).

---

## 7. Modulation Sources (32)

```
aftertouch        env_1..env_6       lfo_1..lfo_8
lift              macro_control_1..4  mod_wheel
note              note_in_octave      pitch_wheel
random            random_1..random_4  slide
stereo            velocity
```

---

## 8. Modulation Destinations (428)

All modulatable parameters. The 22 destinations in `MOD_DEST_BLOCKLIST` are valid Vita destinations but **crash the renderer with a segfault**:

```python
MOD_DEST_BLOCKLIST = {
    "chorus_voices",
    "compressor_band_lower_ratio", "compressor_band_lower_threshold",
    "compressor_band_upper_ratio", "compressor_band_upper_threshold",
    "compressor_enabled_bands",
    "compressor_high_lower_ratio", "compressor_high_lower_threshold",
    "compressor_high_upper_ratio", "compressor_high_upper_threshold",
    "compressor_low_lower_ratio", "compressor_low_lower_threshold",
    "compressor_low_upper_ratio", "compressor_low_upper_threshold",
    "delay_style", "distortion_type",
    "flanger_offset", "phaser_offset", "reverb_damping",
    "pitch_wheel", "mod_wheel", "stereo_mode",
}
```

**Safe modulation destinations = 428 - 22 = 406.**

---

## 9. `.vital` JSON File Format

A `.vital` file is plain JSON, identical to the output of `synth.to_json()`.

### Top-Level Structure

```json
{
  "author": "",
  "comments": "",
  "macro1": "MACRO 1",
  "macro2": "MACRO 2",
  "macro3": "MACRO 3",
  "macro4": "MACRO 4",
  "preset_name": "",
  "preset_style": "",
  "synth_version": "99999.9.9",
  "settings": { ... }
}
```

`synth_version` is `"99999.9.9"` in headless mode (placeholder for the embedded build).

### Settings Object

The `settings` object contains:
1. **772 scalar parameter values** -- exactly 1:1 with `get_controls()`, stored as their raw (denormalized) values
2. **4 non-scalar data arrays** -- accessible only via JSON manipulation, not the control API

```json
{
  "settings": {
    // === Scalar parameters (772) ===
    "beats_per_minute": 2.0,
    "osc_1_level": 0.7071067690849304,
    "osc_1_on": 1.0,
    "filter_1_cutoff": 60.0,
    "volume": 0.7071067690849304,
    "modulation_1_amount": 0.0,
    "modulation_1_bipolar": 0.0,
    "modulation_1_bypass": 0.0,
    "modulation_1_power": 0.0,
    "modulation_1_stereo": 0.0,
    // ... all 772 controls ...

    // === Non-scalar data ===
    "modulations": [ ... ],     // 64-element array
    "lfos": [ ... ],            // 8-element array
    "wavetables": [ ... ],      // 3-element array
    "sample": { ... }           // 1 object
  }
}
```

---

## 10. Non-Scalar Data: Modulations

The `modulations` array has **64 entries** (one per modulation slot). Empty slots have empty source/destination strings.

```json
{
  "modulations": [
    {"source": "lfo_1", "destination": "osc_1_level"},
    {"source": "env_2", "destination": "filter_1_cutoff"},
    {"source": "", "destination": ""},
    // ... 64 entries total, most empty ...
  ]
}
```

The per-slot scalar parameters (`modulation_N_amount`, `modulation_N_bipolar`, etc.) are stored as regular scalar params in the settings object, NOT inside the `modulations` array. The array only defines which source connects to which destination; the slot index (1-based) links to the corresponding scalar params.

### Modulation Defaults by Source Type

When `connect_modulation()` creates a connection:
- **LFO sources**: `modulation_N_bipolar` defaults to `1.0` (bipolar modulation)
- **Envelope sources**: `modulation_N_bipolar` defaults to `0.0` (unipolar modulation)
- `modulation_N_amount` defaults to `0.0` (no effect until explicitly set)

---

## 11. Non-Scalar Data: Wavetables

The `wavetables` array has **3 entries** (one per oscillator).

```json
{
  "wavetables": [
    {
      "author": "",
      "full_normalize": true,
      "name": "Init",
      "remove_all_dc": false,
      "version": "0.0.0",
      "groups": [
        {
          "components": [
            {
              "type": "Wave Source",
              "interpolation": 1,
              "interpolation_style": 1,
              "keyframes": [
                {
                  "position": 0,
                  "wave_data": "<base64-encoded float32 array>"
                }
              ]
            }
          ]
        }
      ]
    },
    { ... },  // oscillator 2
    { ... }   // oscillator 3
  ]
}
```

### wave_data Encoding

- **Format**: Base64-encoded array of **2048 float32 values** in range [-1.0, 1.0]
- Represents one cycle of the waveform
- Multiple keyframes allow wavetable morphing (the `position` field controls where in the wavetable this frame sits)
- The `osc_N_wavetable_position` control selects which frame is active during playback

### Wavetable Properties

| Property | Type | Description |
|----------|------|-------------|
| `author` | string | Author name |
| `full_normalize` | bool | Whether to normalize the wavetable to full amplitude |
| `name` | string | Display name (e.g., "Basic Shapes", "Analog") |
| `remove_all_dc` | bool | Whether to remove DC offset |
| `version` | string | Wavetable version |
| `groups` | array | Array of component groups |

### Component Types

| Type | Description |
|------|-------------|
| `"Wave Source"` | Raw waveform data (base64 float32) |
| `"Line Source"` | Parametric line/shape definition |
| `"Audio File Source"` | Imported audio file reference |

### Project Wavetable Handling

The project catalogs wavetables by content-hashing their JSON:
```python
wt_json = json.dumps(wt, separators=(",", ":"), sort_keys=True)
content_hash = hashlib.sha256(wt_json.encode()).hexdigest()
```

During rendering, wavetables are injected via JSON manipulation (not the control API):
1. `synth.to_json()` -> parse JSON
2. Replace `settings.wavetables[osc_idx]` with cataloged wavetable object
3. `synth.load_json(updated_json)`

---

## 12. Non-Scalar Data: LFOs

The `lfos` array has **8 entries** (one per LFO).

```json
{
  "lfos": [
    {
      "name": "Triangle",
      "num_points": 3,
      "points": [0.0, 1.0, 0.5, 0.0, 1.0, 1.0],
      "powers": [0.0, 0.0, 0.0],
      "smooth": false
    },
    { ... },  // LFO 2
    // ... 8 total
  ]
}
```

### LFO Shape Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | string | Shape name (e.g., "Triangle", "Square", "Custom") |
| `num_points` | int | Number of control points in the shape |
| `points` | float[] | Flat array of (x, y) pairs. x = position [0, 1], y = amplitude [0, 1] |
| `powers` | float[] | Curve power per segment (0.0 = linear, positive = concave, negative = convex) |
| `smooth` | bool | Whether to apply smoothing between segments |

The `points` array length is `num_points * 2`. For example, a triangle wave with 3 points:
- Point 0: `(0.0, 1.0)` -- start at top
- Point 1: `(0.5, 0.0)` -- midpoint at bottom
- Point 2: `(1.0, 1.0)` -- end at top

LFO shapes can only be set via JSON manipulation, not the control API. The scalar LFO controls (`lfo_N_frequency`, `lfo_N_sync`, etc.) ARE accessible via `ctrl.set()`.

---

## 13. Non-Scalar Data: Sample

Single object for the sampler oscillator.

```json
{
  "sample": {
    "length": 44100,
    "name": "White Noise",
    "sample_rate": 44100,
    "samples": "<base64-encoded audio data>"
  }
}
```

| Property | Type | Description |
|----------|------|-------------|
| `length` | int | Number of samples |
| `name` | string | Sample name |
| `sample_rate` | int | Sample rate of the audio data |
| `samples` | string | Base64-encoded audio data |

Like wavetables and LFO shapes, the sample can only be set via JSON manipulation.

---

## 14. Rendering Pipeline (How the Project Uses Vita)

### Full Flow (`datagen/render/engine.py`)

```
1. Reset            synth.load_json(default_json)     Reset to init state
2. Apply params     ctrl.set(raw) for each param      Set all 448 scalar registry params
3. Inject WT        Modify JSON, synth.load_json()    Replace wavetables from catalog
4. Inject LFO/etc   Modify JSON, synth.load_json()    Replace LFO shapes, sample data
5. Apply modulation  synth.connect_modulation()        Create mod routings + set slot params
6. Render           synth.render(note, vel, dur, dur)  Get stereo float32 audio
```

### Step 2: Parameter Application

```python
controls = synth.get_controls()  # cached at init
for name, value in preset.items():
    if name.startswith("_"):  # internal keys
        continue
    ctrl = controls.get(name)
    if ctrl is None:
        continue
    lo, hi = control_ranges[name]
    if name in categorical_names:
        ctrl.set(float(int(value)) + lo)          # categorical: index + min
    else:
        ctrl.set(lo + float(value) * (hi - lo))   # continuous: denormalize
```

### Step 5: Modulation Application

```python
synth.clear_modulations()

slot = 0
for conn in modulation_connections:
    if conn["destination"] in MOD_DEST_BLOCKLIST:
        continue
    synth.connect_modulation(conn["source"], conn["destination"])
    slot += 1

# IMPORTANT: Must re-fetch controls after connecting modulations
# because modulation slot controls only become "active" after connections
controls = synth.get_controls()

for i, conn in enumerate(applied_connections):
    slot_1 = i + 1  # 1-based indexing
    controls[f"modulation_{slot_1}_amount"].set(conn["amount"])
    controls[f"modulation_{slot_1}_bipolar"].set(conn["bipolar"])
    controls[f"modulation_{slot_1}_power"].set(conn["power"])
    controls[f"modulation_{slot_1}_stereo"].set(conn["stereo"])
```

### Normalization Convention

All parameters are stored in the dataset and preset dicts as normalized [0, 1] values (continuous) or integer indices (categorical). Denormalization happens at render time:

```
Dataset/Preset value       Vita raw value
─────────────────────      ──────────────
continuous: 0.5       -->  lo + 0.5 * (hi - lo)
categorical: 3        -->  3.0 + lo  (usually lo=0)
```

This linear normalization is used instead of Vita's built-in `set_normalized()` / `get_normalized()` because Vita's mapping is non-linear for Quadratic/Exponential/SquareRoot scale types.

---

## 15. Community Preset Ingestion

When parsing `.vital` files from community presets:

### Parameter Extraction
```python
# Continuous: linear normalize raw value
norm = (raw - min) / (max - min)
norm = max(0.0, min(1.0, norm))

# Categorical: offset by min
index = int(raw) - int(min_val)
```

### Modulation Extraction (Tier 3)
```python
for slot_0, entry in enumerate(settings["modulations"]):
    source = entry["source"]
    destination = entry["destination"]
    slot_1 = slot_0 + 1  # .vital uses 1-based slot params

    # Check bypass
    if settings.get(f"modulation_{slot_1}_bypass", 0.0) > 0.5:
        continue

    # Read slot params
    amount = settings.get(f"modulation_{slot_1}_amount", 0.0)   # [-1, 1]
    bipolar = settings.get(f"modulation_{slot_1}_bipolar", 0.0)  # 0 or 1
    power = settings.get(f"modulation_{slot_1}_power", 0.0)      # [-10, 10]
    stereo = settings.get(f"modulation_{slot_1}_stereo", 0.0)    # 0 or 1
```

---

## 16. Dense Modulation Matrix Format (Project-Specific)

For training/storage, modulation is converted from sparse connections to a dense matrix:

**Shape**: `(4, n_sources, n_destinations)` = `(4, 32, 406)` float32

| Channel | Index | Range | Description |
|---------|-------|-------|-------------|
| Amount | 0 | [-1, 1] | Modulation depth |
| Bipolar | 1 | {0, 1} | Bipolar mode flag |
| Power | 2 | [-10, 10] | Curve shape |
| Stereo | 3 | {0, 1} | Stereo mode flag |

Conversion from sparse:
```python
matrix = np.zeros((4, n_sources, n_destinations), dtype=np.float32)
for conn in connections:
    si, di = conn["source_idx"], conn["dest_idx"]
    matrix[0, si, di] = conn["amount"]
    matrix[1, si, di] = conn["bipolar"]
    matrix[2, si, di] = conn["power"]
    matrix[3, si, di] = conn["stereo"]
```

Most entries are zero (typically 0-20 connections per preset out of 32 x 406 = 12,992 possible).

---

## 17. HDF5 Dataset Storage

### Schema Attributes (in `schema` group)

| Attribute | Type | Description |
|-----------|------|-------------|
| `sample_rate` | int | Audio sample rate (44100) |
| `continuous_names` | str[] | Ordered list of continuous param names (322) |
| `categorical_names` | str[] | Ordered list of categorical param names |
| `wavetable_names` | str[] | Wavetable param names (separate from categorical) |
| `categorical_n_options` | int[] | Number of options per categorical param |
| `continuous_min` | float[] | Min raw value per continuous param |
| `continuous_max` | float[] | Max raw value per continuous param |
| `mod_source_names` | str[] | Ordered modulation source names (32) |

### Datasets

| Path | Shape | Type | Description |
|------|-------|------|-------------|
| `audio/raw` | (N, 2, S) | float32 | Raw stereo audio (gzip compressed) |
| `features/mel_spectrogram` | (N, 128, T) | float32 | Precomputed mels (uncompressed) |
| `params/continuous` | (N, 322) | float32 | Normalized [0, 1] continuous params |
| `params/categorical` | (N, ~126) | int32 | Integer indices |
| `params/modulation_t3` | (N, 4, 32, 406) | float32 | Dense modulation matrix (tier 3 only, gzip) |
| `metadata/midi_note` | (N,) | int32 | MIDI note used for rendering |
| `metadata/source` | (N,) | str | "synthetic" or "community" |
| `metadata/preset_hash` | (N,) | str | SHA256 preset hash |
| `metadata/preset_name` | (N,) | str | Preset name (if community) |
| `metadata/tier` | (N,) | int32 | Tier level (1, 2, or 3) |

---

## 18. C++ Parameter Definitions (Authoritative)

Source: [synth_parameters.cpp](https://github.com/mtytel/vital/blob/c0694a193777fc97853a598f86378bea625a6d81/src/common/synth_parameters.cpp)

These are the ground-truth definitions from Vital's C++ source. Parameter groups are instantiated per-module (e.g., `osc_1_*`, `osc_2_*`, `osc_3_*`).

### Oscillator Parameters (x3: osc_1, osc_2, osc_3)

| Param | Scale | Min | Max | Default |
|-------|-------|-----|-----|---------|
| `on` | Indexed | 0 | 1 | 0 |
| `transpose` | Indexed | -48 | 48 | 0 |
| `transpose_quantize` | Indexed | 0 | 8191 | 0 |
| `tune` | Linear | -1 | 1 | 0 |
| `pan` | Linear | -1 | 1 | 0 |
| `stack_style` | Indexed | 0 | 10 | 0 |
| `unison_detune` | **Quadratic** | 0 | 10 | 4.472 |
| `unison_voices` | Indexed | 1 | 16 | 1 |
| `unison_blend` | Linear | 0 | 1 | 0.8 |
| `detune_power` | Linear | -5 | 5 | 1.5 |
| `detune_range` | Linear | 0 | 48 | 2 |
| `level` | **Quadratic** | 0 | 1 | 0.7071 |
| `midi_track` | Indexed | 0 | 1 | 1 |
| `smooth_interpolation` | Indexed | 0 | 1 | 0 |
| `spectral_unison` | Indexed | 0 | 1 | 1 |
| `wave_frame` | Linear | 0 | 256 | 0 |
| `frame_spread` | Linear | -128 | 128 | 0 |
| `stereo_spread` | Linear | 0 | 1 | 1 |
| `phase` | Linear | 0 | 1 | 0.5 |
| `distortion_phase` | Linear | 0 | 1 | 0.5 |
| `random_phase` | Linear | 0 | 1 | 1 |
| `distortion_type` | Indexed | 0 | 12 | 0 |
| `distortion_amount` | Linear | 0 | 1 | 0.5 |
| `distortion_spread` | Linear | -0.5 | 0.5 | 0 |
| `spectral_morph_type` | Indexed | 0 | 11 | 0 |
| `spectral_morph_amount` | Linear | 0 | 1 | 0.5 |
| `spectral_morph_spread` | Linear | -0.5 | 0.5 | 0 |
| `destination` | Indexed | 0 | 8 | 0 |
| `view_2d` | Indexed | 0 | 2 | 1 |

### Envelope Parameters (x6: env_1 through env_6)

| Param | Scale | Min | Max | Default | Display meaning |
|-------|-------|-----|-----|---------|-----------------|
| `delay` | **Quartic** | 0 | 1.4142 | 0 | max^4 = 4.0 sec |
| `attack` | **Quartic** | 0 | 2.3784 | 0.1495 | max^4 = 32.0 sec |
| `hold` | **Quartic** | 0 | 1.4142 | 0 | max^4 = 4.0 sec |
| `decay` | **Quartic** | 0 | 2.3784 | 1.0 | max^4 = 32.0 sec |
| `release` | **Quartic** | 0 | 2.3784 | 0.5476 | max^4 = 32.0 sec |
| `attack_power` | Linear | -20 | 20 | 0 | |
| `decay_power` | Linear | -20 | 20 | -2 | |
| `release_power` | Linear | -20 | 20 | -2 | |
| `sustain` | Linear | 0 | 1 | 1 | |

The Quartic max values are chosen so `raw^4` gives musically useful maximum times: 2.3784 = 32^(1/4) (32 sec max for ADR), 1.4142 = 4^(1/4) = sqrt(2) (4 sec max for delay/hold).

### LFO Parameters (x8: lfo_1 through lfo_8)

| Param | Scale | Min | Max | Default |
|-------|-------|-----|-----|---------|
| `phase` | Linear | 0 | 1 | 0 |
| `sync_type` | Indexed | 0 | 5 | 0 |
| `frequency` | **Exponential** | -7 | 9 | 1 |
| `sync` | Indexed | 0 | 4 | 1 |
| `tempo` | Indexed | 0 | 12 | 7 |
| `fade_time` | Linear | 0 | 8 | 0 |
| `smooth_mode` | Indexed | 0 | 1 | 1 |
| `smooth_time` | **Exponential** | -10 | 4 | -7.5 |
| `delay_time` | Linear | 0 | 4 | 0 |
| `stereo` | Linear | -0.5 | 0.5 | 0 |
| `keytrack_transpose` | Indexed | -60 | 36 | -12 |
| `keytrack_tune` | Linear | -1 | 1 | 0 |

### Random Generator Parameters (x4: random_1 through random_4)

| Param | Scale | Min | Max | Default |
|-------|-------|-----|-----|---------|
| `style` | Indexed | 0 | 3 | 0 |
| `frequency` | **Exponential** | -7 | 9 | 1 |
| `sync` | Indexed | 0 | 4 | 1 |
| `tempo` | Indexed | 0 | 12 | 8 |
| `stereo` | Indexed | 0 | 1 | 0 |
| `sync_type` | Indexed | 0 | 1 | 0 |
| `keytrack_transpose` | Indexed | -60 | 36 | -12 |
| `keytrack_tune` | Linear | -1 | 1 | 0 |

### Filter Parameters (x3: filter_1, filter_2, filter_fx)

| Param | Scale | Min | Max | Default |
|-------|-------|-----|-----|---------|
| `mix` | Linear | 0 | 1 | 1 |
| `cutoff` | Linear | 8 | 136 | 60 |
| `resonance` | Linear | 0 | 1 | 0.5 |
| `drive` | Linear | 0 | 20 | 0 |
| `blend` | Linear | 0 | 2 | 0 |
| `style` | Indexed | 0 | 9 | 0 |
| `model` | Indexed | 0 | 7 | 0 |
| `on` | Indexed | 0 | 1 | 0 |
| `blend_transpose` | Linear | 0 | 84 | 42 |
| `keytrack` | Linear | -1 | 1 | 0 |
| `formant_x` | Linear | 0 | 1 | 0.5 |
| `formant_y` | Linear | 0 | 1 | 0.5 |
| `formant_transpose` | Linear | -12 | 12 | 0 |
| `formant_resonance` | Linear | 0.3 | 1 | 0.85 |
| `formant_spread` | Linear | -1 | 1 | 0 |

All filter params are **Linear or Indexed**. No non-linear scales.

### Effect Parameters (Global, not per-instance)

**Delay:**

| Param | Scale | Min | Max | Default |
|-------|-------|-----|-----|---------|
| `delay_dry_wet` | Linear | 0 | 1 | 0.333 |
| `delay_feedback` | Linear | -1 | 1 | 0.5 |
| `delay_frequency` | **Exponential** | -2 | 9 | 2 |
| `delay_aux_frequency` | **Exponential** | -2 | 9 | 2 |
| `delay_filter_cutoff` | Linear | 8 | 136 | 60 |
| `delay_filter_spread` | Linear | 0 | 1 | 1 |

**Chorus:**

| Param | Scale | Min | Max | Default |
|-------|-------|-----|-----|---------|
| `chorus_dry_wet` | Linear | 0 | 1 | 0.5 |
| `chorus_feedback` | Linear | -0.95 | 0.95 | 0.4 |
| `chorus_cutoff` | Linear | 8 | 136 | 60 |
| `chorus_frequency` | **Exponential** | -6 | 3 | -3 |
| `chorus_mod_depth` | Linear | 0 | 1 | 0.5 |
| `chorus_delay_1` | **Exponential** | -10 | -5.644 | -9 |
| `chorus_delay_2` | **Exponential** | -10 | -5.644 | -7 |

**Flanger:**

| Param | Scale | Min | Max | Default |
|-------|-------|-----|-----|---------|
| `flanger_dry_wet` | Linear | 0 | 0.5 | 0.5 |
| `flanger_feedback` | Linear | -1 | 1 | 0.5 |
| `flanger_frequency` | **Exponential** | -5 | 2 | 2 |
| `flanger_center` | Linear | 8 | 136 | 64 |
| `flanger_mod_depth` | Linear | 0 | 1 | 0.5 |

**Phaser:**

| Param | Scale | Min | Max | Default |
|-------|-------|-----|-----|---------|
| `phaser_dry_wet` | Linear | 0 | 1 | 1 |
| `phaser_feedback` | Linear | 0 | 1 | 0.5 |
| `phaser_frequency` | **Exponential** | -5 | 2 | -3 |
| `phaser_center` | Linear | 8 | 136 | 80 |
| `phaser_blend` | Linear | 0 | 2 | 1 |
| `phaser_mod_depth` | Linear | 0 | 48 | 24 |

**Reverb:**

| Param | Scale | Min | Max | Default |
|-------|-------|-----|-----|---------|
| `reverb_dry_wet` | Linear | 0 | 1 | 0.25 |
| `reverb_decay_time` | **Exponential** | -6 | 6 | 0 |
| `reverb_chorus_amount` | **Quadratic** | 0 | 1 | 0.2236 |
| `reverb_chorus_frequency` | **Exponential** | -8 | 3 | -2 |
| `reverb_size` | Linear | 0 | 1 | 0.5 |
| `reverb_delay` | Linear | 0 | 0.3 | 0 |

**EQ:**

| Param | Scale | Min | Max | Default |
|-------|-------|-----|-----|---------|
| `eq_low_resonance` | **Quadratic** | 0 | 1 | 0.3163 |
| `eq_band_resonance` | **Quadratic** | 0 | 1 | 0.4473 |
| `eq_high_resonance` | **Quadratic** | 0 | 1 | 0.3163 |
| `eq_*_cutoff` | Linear | 8 | 136 | varies |
| `eq_*_gain` | Linear | varies | varies | 0 |

### Global Parameters

| Param | Scale | Min | Max | Default |
|-------|-------|-----|-----|---------|
| `volume` | **SquareRoot** | 0 | 7399.44 | 5473.04 |
| `portamento_time` | **Exponential** | -10 | 4 | -10 |
| `portamento_slope` | Linear | -8 | 8 | 0 |
| `beats_per_minute` | Linear | 0.333 | 5.0 | 2.0 |
| `voice_tune` | Linear | -1 | 1 | 0 |
| `voice_amplitude` | Linear | 0 | 1 | 1 |
| `stereo_routing` | Linear | 0 | 1 | 1 |
| `pitch_wheel` | Linear | -1 | 1 | 0 |
| `mod_wheel` | Linear | 0 | 1 | 0 |
| `velocity_track` | Linear | -1 | 1 | 0 |
| `macro_control_{1..4}` | Linear | 0 | 1 | 0 |
| `polyphony` | Indexed | 1 | 32 | 8 |
| `voice_transpose` | Indexed | -48 | 48 | 0 |
| `pitch_bend_range` | Indexed | 0 | 48 | 2 |
| `oversampling` | Indexed | 0 | 3 | 1 |

### Sub-Oscillator & Sample

| Param | Scale | Min | Max | Default |
|-------|-------|-----|-----|---------|
| `sub_level` | **Quadratic** | 0 | 1 | 0.7071 |
| `sub_tune` | Linear | -1 | 1 | 0 |
| `sub_pan` | Linear | -1 | 1 | 0 |
| `sample_level` | **Quadratic** | 0 | 1 | 0.7071 |
| `sample_tune` | Linear | -1 | 1 | 0 |
| `sample_pan` | Linear | -1 | 1 | 0 |

### Modulation Slot Parameters (x64: modulation_1 through modulation_64)

| Param | Scale | Min | Max | Default |
|-------|-------|-----|-----|---------|
| `amount` | Linear | -1 | 1 | 0 |
| `power` | Linear | -10 | 10 | 0 |
| `bipolar` | Indexed | 0 | 1 | 0 |
| `stereo` | Indexed | 0 | 1 | 0 |
| `bypass` | Indexed | 0 | 1 | 0 |

All modulation params are **Linear or Indexed**. No non-linear scales.

### Complete Non-Linear Parameter Count

| Scale | Formula (skew/unskew) | Params | Count |
|-------|----------------------|--------|-------|
| Quadratic | x^2 / sqrt(x) | osc_{1,2,3}_level, osc_{1,2,3}_unison_detune, eq_{low,band,high}_resonance, reverb_chorus_amount, sub_level, sample_level | **12** |
| Quartic | x^4 / x^(1/4) | env_{1..6}_{delay,attack,hold,decay,release} | **30** |
| Exponential | 2^x / log2(x) | lfo_{1..8}_frequency, lfo_{1..8}_smooth_time, random_{1..4}_frequency, delay_frequency, delay_aux_frequency, chorus_frequency, chorus_delay_1, chorus_delay_2, flanger_frequency, phaser_frequency, reverb_decay_time, reverb_chorus_frequency, portamento_time | **31** |
| SquareRoot | sqrt(x) / x^2 | volume | **1** |
| **Total non-linear** | | | **74** |

---

## 19. Summary of Gotchas

1. **`set_normalized()` is lossy** for 74 params with Quadratic/Quartic/Exponential/SquareRoot scales. Always use `ctrl.set(raw)` with manual linear denormalization.
2. **Velocity is [0, 1]**, not [0, 127]. Divide by 127.0.
3. **7 controls crash on `details.options`** (`filter_{1,2,fx}_style`, `osc_{1,2,3}_view_2d`, `view_spectrogram`). Infer categoricality from integer range.
4. **22 modulation destinations crash Vita** with segfault. See `MOD_DEST_BLOCKLIST`.
5. **Modulation slot controls exist always** (all 320), but slot params only affect audio when a connection is active in the corresponding `modulations` array entry.
6. **Non-scalar data (wavetables, LFO shapes, sample) requires JSON manipulation**. The control API cannot set these.
7. **Rendering is non-deterministic** due to oscillator phase randomization.
8. **v0.0.5 is fixed at 44100 Hz** -- no `set_sample_rate()`.
9. **Each process needs its own `vita.Synth()`** instance. Cannot share across process boundaries.
10. **`synth.to_json()` output IS the `.vital` file format** -- they are identical.
11. **Settings values are raw (denormalized)** in JSON. The project normalizes to [0, 1] for ML training and denormalizes back at render time.
12. **`get_controls()` must be called again after `connect_modulation()`** in the project's rendering flow because the codebase refreshes its control reference to ensure modulation slot params are properly set.
13. **Envelope time max values look weird but are intentional.** `2.37842 = 32^(1/4)` and `1.4142 = 4^(1/4)` -- when passed through Quartic display skewing (`x^4`), they give 32-second and 4-second max display times respectively.
14. **The C++ plugin normalization (ValueBridge) is always linear**, even for non-linear scale types. The non-linear skewing is a separate display/knob layer. Our project's `raw = min + norm * (max - min)` matches the plugin's internal `convertToEngineValue()`.
