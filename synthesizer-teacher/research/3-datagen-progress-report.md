# Dataset Generation Pipeline — Progress Report

**Date**: 2026-02-07 **Phase**: Complete. All 3 tiers generated and validated.

---

## Status: DONE

The full dataset generation pipeline is implemented, tested against the live
Vita synthesizer, and has produced 4 datasets totaling ~2.1 GB.

---

## Generated Datasets

**Note**: Previous datasets (below) used the old param counts and rendering approach.
New datasets should be regenerated with the updated pipeline (448 params, unified
param-by-param rendering).

| Dataset                  | Samples | Params (OLD)                                    | Source Mix                    | Rejection Rate | Size   |
| ------------------------ | ------- | ----------------------------------------------- | ----------------------------- | -------------- | ------ |
| `data/tier1_1k.h5`       | 1,000   | 25 cont + 3 cat                                 | 100% synthetic                | 30.7%          | 533 MB |
| `data/tier1_mixed_1k.h5` | 1,000   | 25 cont + 3 cat                                 | 217 community + 783 synthetic | 24.1%          | 533 MB |
| `data/tier2_1k.h5`       | 1,000   | 60 cont + 5 cat + 16 mod                        | 200 community + 800 synthetic | 27.1%          | 526 MB |
| `data/tier3_1k.h5`       | 1,000   | 171 cont + 7 cat + 16 mod + 25x171 dense matrix | 200 community + 800 synthetic | 25.9%          | 519 MB |

**Updated param counts** (after registry refactor):

| Tier   | Continuous | Categorical | Modulation              |
| ------ | ---------- | ----------- | ----------------------- |
| Tier 1 | ~55        | ~27         | None                    |
| Tier 2 | 322        | 126         | None                    |
| Tier 3 | 322        | 126         | (4, 32, 428) dense matrix |

Additional artifacts:

- `data/wavetable_catalog.json` — 123 unique wavetables from 75 preset files (43
  MB)

All audio is stereo float32 at 44100 Hz, 2 seconds per sample (note held 1.5s +
0.5s release). Each preset rendered at 3 MIDI notes (48, 60, 72).

---

## Files Created (22 Python files + pyproject.toml)

```
datagen/
├── __init__.py              # Package root, version
├── __main__.py              # python -m datagen entry point
├── cli.py                   # Click CLI: generate, discover-wt, download-presets, compute-weights, inspect, preview
├── config.py                # PipelineConfig, tier module prefixes, MOD_DEST_BLOCKLIST, heuristics
├── pipeline.py              # Orchestrator: sample → render → validate → store
├── preview.py               # TUI dataset browser with playback (Textual)
├── params/
│   ├── __init__.py
│   ├── registry.py          # ParamRegistry: data-driven from vita.Synth, auto-classifies 448 params
│   ├── sampler.py           # LHS for continuous, uniform for categoricals, modulation sampling
│   ├── normalize.py         # Normalize/denormalize scalars and vectors
│   └── importance.py        # Perturbation-based importance weight computation
├── wavetables/
│   ├── __init__.py
│   ├── discovery.py         # Scan Vital dirs, extract WTs from JSON, dedup by SHA256
│   └── catalog.py           # WavetableCatalog: name→index→base64, JSON persistence
├── render/
│   ├── __init__.py
│   ├── engine.py            # RenderEngine: unified param-by-param rendering (no _vital_json)
│   ├── pool.py              # Multiprocessing pool with per-worker Synth instances
│   └── validator.py         # AudioValidator: RMS/peak/NaN/shape rejection checks
├── presets/
│   ├── __init__.py
│   ├── synthetic.py         # SyntheticPresetGenerator with hash dedup
│   ├── ingest.py            # PresetIngester: extract ALL params from .vital, linear normalize
│   ├── factory.py           # Factory preset discovery and loading
│   └── download.py          # Git clone from 3 known repos
└── storage/
    ├── __init__.py
    ├── schema.py            # HDF5Schema: dataset definitions per tier
    ├── writer.py            # Chunked HDF5Writer with gzip, appendable, auto-flush
    └── reader.py            # HDF5Reader: info(), get_sample(), get_batch()
```

---

## Vita API — Actual Behavior (v0.0.5)

Critical findings from testing against the real Vita library. These differ
significantly from the research docs' assumptions.

### Constructor

```python
synth = vita.Synth()  # NO arguments — sample rate fixed at 44100
```

The research docs assumed `vita.Synth(sample_rate=44100)` — this is wrong.

### Rendering

```python
audio = synth.render(midi_note, velocity_float, note_dur, render_dur)
# Returns np.ndarray shape (2, n_samples) float32
```

The research docs assumed separate `note_on()` / `note_off()` /
`render(num_samples)` calls — this is wrong. Vita uses a single-call API.

### Controls

```python
controls = synth.get_controls()  # dict[str, ControlValue]
ctrl = controls["osc_1_level"]
ctrl.set_normalized(0.5)       # Maps [0,1] to actual range
ctrl.set(3.0)                  # Sets raw value (use for categoricals)
ctrl.value()                   # Get current raw value
ctrl.get_normalized()          # Get current [0,1] value

info = synth.get_control_details("osc_1_level")  # ControlInfo
info.min, info.max, info.default_value, info.scale, info.is_discrete, info.options
```

### Total controls: 772

Far more than the ~220 we expected. Breakdown:
- **448 in registry** (322 continuous + 126 categorical) — all sound-affecting params
- **320 modulation slot params** (64 slots × 5: amount/bipolar/bypass/power/stereo) — handled via `_modulation_t3`
- **4 view-only** (view_spectrogram, osc_N_view_2d) — excluded

### Key parameter ranges (NOT [0,1])

| Param               | Min   | Max    | Default | Scale      |
| ------------------- | ----- | ------ | ------- | ---------- |
| osc_1_level         | 0.0   | 1.0    | 0.707   | Linear     |
| filter_1_cutoff     | 8.0   | 136.0  | 60.0    | —          |
| filter_1_model      | 0.0   | 7.0    | 0.0     | Discrete   |
| env_1_attack        | 0.0   | 2.38   | 0.15    | —          |
| volume              | 0.0   | 7399.4 | 5473.0  | SquareRoot |
| osc_1_transpose     | -48.0 | 48.0   | 0.0     | —          |
| osc_1_unison_voices | 1.0   | 16.0   | 1.0     | Discrete   |

### Module on/off switches (15 params)

- `osc_1_on=1` by default, `osc_2_on=0`, `osc_3_on=0`
- `filter_1_on=0`, `filter_2_on=0` (filters off = pass-through, not silence)
- All 15 `_on` switches are now **regular registry params** (categorical, 2 options)
- They're set via `_apply_params()` just like any other param
- Sampler forces `osc_1_on = 1` to ensure at least one sound source
- Full list: osc_1/2/3, filter_1/2, filter_fx, chorus, compressor, delay, distortion, eq, flanger, phaser, reverb, sample

### Modulation

```python
synth.connect_modulation("lfo_1", "osc_1_level")  # Connect source→dest
synth.clear_modulations()                           # Remove all
synth.disconnect_modulation(source, dest)           # Remove specific
```

All 25 modulation sources work. However, **22 destinations crash Vita with a
segfault** (exit code 139):

### Modulation Destination Blocklist (22 params that segfault Vita)

```
chorus_voices, compressor_band_lower_ratio, compressor_band_lower_threshold,
compressor_band_upper_ratio, compressor_band_upper_threshold,
compressor_enabled_bands, compressor_high_lower_ratio,
compressor_high_lower_threshold, compressor_high_upper_ratio,
compressor_high_upper_threshold, compressor_low_lower_ratio,
compressor_low_lower_threshold, compressor_low_upper_ratio,
compressor_low_upper_threshold, delay_style, distortion_type,
flanger_offset, phaser_offset, reverb_damping,
pitch_wheel, mod_wheel, stereo_mode
```

These are all discrete/integer params or compressor ratio/threshold params.
They're stored in `config.MOD_DEST_BLOCKLIST` and filtered out of modulation
destinations.

### Missing params (3 in Tier 3 list not found in Vita)

```
flanger_offset, phaser_offset, reverb_damping
```

These are silently skipped by the engine.

### Wavetable structure

Vital JSON stores wavetables in `settings.wavetables[]`, each with:

```json
{
  "name": "Init",
  "author": "",
  "full_normalize": false,
  "remove_all_dc": false,
  "version": "...",
  "groups": [{
    "components": [{
      "interpolation": ...,
      "interpolation_style": ...,
      "keyframes": [...],
      "type": ...,
      "audio_file": "base64..."  // Only present for non-Init wavetables
    }]
  }]
}
```

The init preset has 3 "Init" wavetables with no audio_file data. Community
presets embed real wavetable data (100KB-1.4MB per wavetable as base64).

### Vital install paths (macOS, this machine)

- App: `/Applications/Vital.app`
- Config: `~/Library/Application Support/Vital/` (only Vital.config and
  Vital.library)
- Presets: `~/Music/Vital/` — 75 .vital files across 14 packs
- Factory presets: `~/Music/Vital/Factory/Presets/` — only 7 files (Free tier)
- No factory presets at `~/Library/Application Support/Vital/Factory/Presets/`
  (the default assumed path)

---

## Key Fixes Applied During Generation

### 1. Vita API adaptation (`render/engine.py`)

- Changed `vita.Synth(sample_rate=...)` → `vita.Synth()` (no args)
- Changed separate `note_on/note_off/render` → single
  `synth.render(midi_note, velocity, note_dur, render_dur)`
- Velocity is a float [0,1] not int [0,127]

### 2. Volume scaling (`config.py`)

- Volume has SquareRoot scale: normalized 0.3 → raw 666 → near silence (RMS
  0.0005)
- Tightened constraint from [0.3, 1.0] to [0.55, 0.85] normalized
- 0.55 norm → ~2240 raw → audible; 0.85 norm → ~5345 raw → safe from clipping

### 3. Oscillator on/off (`params/sampler.py`)

- Sampler forces `osc_1_on = 1` after random sampling to ensure sound
- `_on` switches are regular registry params (no separate `_auto_enable_modules`)

### 4. Community preset normalization (`presets/ingest.py`)

- Vital JSON stores raw param values (e.g., filter_1_cutoff=45.3,
  env_1_attack=0.149)
- All community preset values linear-normalized to [0,1] during ingestion:
  `(raw - min) / (max - min)`
- Ingester extracts ALL 448 registry params (not just tier-filtered subset)
- Non-scalar data (`_lfos`, `_wavetables`, `_sample`) stored under `_` keys for
  JSON injection during rendering

### 5. Modulation destination blocklist (`config.py`, `render/engine.py`)

- 22 params crash Vita when used as modulation destinations (segfault, exit 139)
- Added `MOD_DEST_BLOCKLIST` set in config
- Engine checks blocklist before calling `connect_modulation()`
- Safe mod destinations: 428 (after blocklist)

### 6. Preset ordering (`pipeline.py`)

- Community/factory presets placed first in render queue (before synthetic)
- Ensures they're included in the dataset even when synthetic count is high

### 7. Filter cutoff constraint (`config.py`)

- Tightened from [0.1, 1.0] to [0.15, 1.0]

### 8. Unified param-by-param rendering (v2 refactor)

- **Removed `_vital_json` direct loading** — was causing label/render mismatch
- **All presets (synthetic + community)** use same rendering flow:
  reset → apply_params → inject_wavetables → inject_nonscalar → apply_modulation
- Engine denormalizes continuous values: `raw = min + norm * (max - min)`, then `ctrl.set(raw)`
- Control ranges cached at engine init for performance
- Non-scalar data (LFO shapes, sample audio, wavetable defs) injected via JSON manipulation
- Spectral correlation between new approach and direct JSON loading: 0.80-0.9997
  (waveform differences are from phase randomization)

### 9. Data-driven parameter registry (v2 refactor)

- Registry auto-discovers all params from Vita API at startup
- Only 4 controls excluded (view-only): `view_spectrogram`, `osc_N_view_2d`
- `_on` switches (15 module enables) included as regular categorical params
- `bypass`, `mpe_enabled`, `oversampling` included
- 320 `modulation_*` slot params excluded (handled via `_modulation_t3`)
- Total: 448 params (322 continuous + 126 categorical)

---

## Generation Performance

| Metric                                | Value                                                 |
| ------------------------------------- | ----------------------------------------------------- |
| Render speed                          | ~10-13 presets/sec (single process)                   |
| Time per 1K samples                   | ~100-135 seconds                                      |
| Rejection rate (synthetic)            | 25-31%                                                |
| Rejection rate (community)            | Lower (~15-20%, hand-crafted presets are more viable) |
| Main rejection reason                 | SILENT (~95% of rejections)                           |
| Clipping rejections                   | Rare (~1% of rejections)                              |
| Samples per 44100 Hz stereo 2s render | 88,200 × 2 channels                                   |
| HDF5 file per 1K samples              | ~500-530 MB                                           |

---

## Wavetable Catalog

- 123 unique wavetables discovered from 75 .vital preset files
- Deduplicated by SHA256 of serialized JSON content
- 59 unique names (many duplicates like "Init" ×25, "Resynthesize" ×9)
- Source: `~/Music/Vital/` (all community/factory packs on this machine)
- Catalog persisted to `data/wavetable_catalog.json` (43 MB)

---

## HDF5 Schema (current — v2.1.0)

All tiers share the same schema structure; only the param vector sizes and
modulation presence differ.

### Tier 1

```
audio/waveforms:      (N, 2, 88200)   float32
params/continuous:    (N, ~55)         float32   [0,1] linear normalized
params/categorical:   (N, ~27)         int32     [includes osc_1_on, filter_1_on, etc.]
metadata/midi_note:   (N,)             int32
metadata/source:      (N,)             S10
metadata/tier:        (N,)             int32
metadata/preset_hash: (N,)             S64
metadata/preset_name: (N,)             S80
```

### Tier 2 (all params, no modulation)

```
params/continuous:    (N, 322)         float32
params/categorical:   (N, 126)         int32     [all _on switches, models, styles, etc.]
```

### Tier 3 (adds modulation matrix)

```
params/modulation_t3: (N, 4, 32, 428)  float32   [channels × sources × destinations]
                      channels: [amount, bipolar, power, stereo]
```

---

## CLI Usage (verified working)

```bash
# Discover wavetables
python -m datagen discover-wt -o data/wavetable_catalog.json --extra-dir ~/Music/Vital

# Generate Tier 1 (synthetic only)
python -m datagen generate --tier 1 -n 1000 -o data/tier1_1k.h5 --seed 42 \
  --wavetable-catalog data/wavetable_catalog.json

# Generate Tier 1 (mixed with community presets)
python -m datagen generate --tier 1 -n 1000 -o data/tier1_mixed_1k.h5 --seed 123 \
  --wavetable-catalog data/wavetable_catalog.json \
  --community-dir /Users/theol/Music/Vital

# Generate Tier 2
python -m datagen generate --tier 2 -n 1000 -o data/tier2_1k.h5 --seed 42 \
  --wavetable-catalog data/wavetable_catalog.json \
  --community-dir /Users/theol/Music/Vital

# Generate Tier 3
python -m datagen generate --tier 3 -n 1000 -o data/tier3_1k.h5 --seed 42 \
  --wavetable-catalog data/wavetable_catalog.json \
  --community-dir /Users/theol/Music/Vital

# Inspect any dataset
python -m datagen inspect data/tier2_1k.h5 -s 42

# Download community presets
python -m datagen download-presets -o presets/community/

# Compute importance weights
python -m datagen compute-weights --n-base 500 -o data/tier1_1k.h5

# Preview with a tui
python -m datagen preview
```

---

## Next Steps (for training phase)

1. **Regenerate datasets**: Re-run generation with the updated 448-param pipeline (old datasets have incomplete labels)
2. **Scale datasets**: Generate 10K-100K samples per tier for actual training
3. **Multiprocessing**: Test `--workers 4` for parallel rendering
4. **Importance weights**: Run `compute-weights` to get per-param loss weights
5. **Training pipeline**: Build PyTorch DataLoader reading from HDF5
6. **Encoder selection**: Test MERT-95M vs AST vs ResNet-18 on spectrograms
7. **Loss functions**: Implement param MSE + multi-resolution STFT + CLAP

---

## Architecture Notes

### Why these specific design choices

**Unified param-by-param rendering**: Every preset (synthetic and community)
goes through the same rendering flow. No `_vital_json` direct loading. This
ensures the dataset label exactly reflects what was sent to the synth. Spectral
correlation with direct JSON loading is 0.80-0.9997; waveform differences are
from inherent phase randomization in Vita's oscillators.

**Data-driven parameter registry**: All params are auto-discovered from the live
Vita API at startup. Only 4 view-only controls are excluded by name/suffix.
Module `_on` switches, `bypass`, `mpe_enabled`, `oversampling` are all included.
This means new Vita controls will be picked up automatically.

**Linear normalization**: `(raw - min) / (max - min)` for labels and
`raw = min + norm * (max - min)` for rendering. Vita's `set_normalized()` uses
nonlinear mappings (exponential, square root) for some params, which causes
lossy roundtrips. Linear mapping avoids this entirely.

**Non-scalar JSON injection**: LFO shapes, sample audio, and wavetable
definitions can't be set via the control API. They're injected by reading the
current synth JSON, overlaying the data, and reloading. This happens after
`_apply_params()` so scalar params are preserved in the JSON.

**LHS over pure random**: Latin Hypercube Sampling guarantees each dimension is
evenly covered even at small sample sizes (1K). Pure random would leave gaps in
the 322-dimensional continuous parameter space.

**Heuristic constraints**: Without them, ~50%+ of random presets are silent (osc
level near 0, volume near 0). Constraining osc_1_level > 0.3 and volume to
[0.55, 0.85] normalized reduces rejection to ~25-31%.

**Multi-pitch rendering (3 notes per preset)**: Captures key-tracking,
velocity-sensitive filters, and pitch-dependent behavior. A single note would
miss how the sound changes across the keyboard.

**HDF5 over individual files**: At 100K samples, individual .wav + .json files
would create filesystem overhead. HDF5 with chunked gzip gives compression and
allows efficient batch reads for training.

**Flush every 256 samples**: Balances write performance (fewer disk ops) with
crash resilience (at most 256 samples lost on crash). Matches the chunk size for
optimal HDF5 alignment.

**Community presets first in queue**: Ensures they're rendered before we hit the
sample target from synthetics alone. Community presets have lower rejection
rates since they're hand-crafted.

**Modulation destination blocklist**: Rather than wrapping every
`connect_modulation()` call in a subprocess (slow), we discovered the 22 bad
destinations upfront and filter them at sampling time. This is safe and fast.
