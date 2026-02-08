# Plan: Dataset Generation Pipeline for Vital Inverse Synthesis

## Context

This is Step 1 of the QHacks 2026 inverse synthesis project. We need a pipeline that generates (audio, parameters) training pairs by rendering random Vital presets using the Vita Python library, validates them, ingests community presets, and stores everything in HDF5. The pipeline supports three tiers of parameter complexity so the same code serves the MVP through the full research model.

**Constraints**: Free-tier Vital (25 factory wavetables), starting with ~1K samples for pipeline validation, HDF5 storage, auto-ingest from folder + separate download component.

---

## Module Structure

```
datagen/
├── __init__.py
├── __main__.py             # python -m datagen support
├── cli.py                  # Click CLI: generate, download-presets, discover-wt, compute-weights, inspect
├── config.py               # PipelineConfig dataclass, tier param lists, fixed Tier 2 mod routings, paths
├── params/
│   ├── __init__.py
│   ├── registry.py         # ParamRegistry: discovers all params from Vita, tags by tier
│   ├── sampler.py          # LHS for continuous, uniform for categoricals, modulation sampling
│   ├── normalize.py        # Normalize/denormalize using Vita control ranges
│   └── importance.py       # Perturbation-based importance weight computation
├── wavetables/
│   ├── __init__.py
│   ├── discovery.py        # Scan Vital install dirs for factory WT data
│   └── catalog.py          # WavetableCatalog: name→index→base64 data, persist to JSON
├── render/
│   ├── __init__.py
│   ├── engine.py           # RenderEngine: wraps vita.Synth, reset/apply/render per preset
│   ├── pool.py             # Multiprocessing pool of RenderEngine workers
│   └── validator.py        # Rejection sampling: RMS, peak, NaN checks
├── presets/
│   ├── __init__.py
│   ├── synthetic.py        # Wraps ParamSampler into preset dicts ready for rendering
│   ├── ingest.py           # Scan folder of .vital files, extract params, check wavetables
│   ├── factory.py          # Locate/load Vital factory presets from install dir
│   └── download.py         # Decoupled: clone GitHub repos, extract .vital files
├── storage/
│   ├── __init__.py
│   ├── schema.py           # HDF5 dataset schema definition
│   ├── writer.py           # Chunked HDF5 writer with gzip, appendable
│   └── reader.py           # HDF5 reader for verification and downstream use
└── pipeline.py             # Orchestrator: sampling → rendering → validation → storage
```

---

## Key Design Decisions

### Tiered Parameter Schema

Parameters are auto-discovered from the live Vita API at startup (`ParamRegistry.from_synth()`).
Tier assignment is based on module prefix matching against `TIER1_MODULES` in `config.py`.

**Registry totals: 448 params** (322 continuous + 126 categorical), auto-classified from Vita's 772 controls.
Excluded: 320 `modulation_*` slot params (handled via `_modulation_t3`), 4 view-only controls.

**Tier 1 (82 params, no modulation)**:
- Osc 1-2: level, wave_frame, transpose, tune, unison_voices, unison_detune, pan, destination, distortion, spectral_morph, frame_spread, phase, stereo_spread, etc. + `osc_1_on`, `osc_2_on`
- Filter 1: cutoff, resonance, drive, blend, mix, model, style, formant, keytrack, etc. + `filter_1_on`
- Envelope 1: attack, decay, sustain, release, delay, hold, power curves
- Volume
- Wavetable categoricals (tier 2+): osc_N_wavetable

**Tier 2 (448 params, no modulation)**:
- All of Tier 1 + Osc 3, Filter 2, filter_fx, Envelopes 2-6, LFOs 1-8, Random 1-4
- All 9 effects (chorus, compressor, delay, distortion, eq, flanger, phaser, reverb) with `*_on` switches
- Sample playback, macros, voice settings, portamento, global params
- `bypass`, `mpe_enabled`, `oversampling`, etc.

**Tier 3 (448 params + modulation matrix)**:
- All of Tier 2 + dense (4, 32 sources × 428 destinations) modulation matrix
- 4 channels per connection: amount, bipolar, power, stereo
- Sparse: connections sampled from Geometric(p=0.15), clamped to [0, 20]

Each tier is a strict superset. A Tier 3 dataset can be filtered to Tier 1 by column subset.

### Wavetable Handling

Vita has **no API to enumerate factory wavetables**. Wavetable data lives as base64 in .vital JSON files.

Discovery strategy:
1. Scan known Vital install paths (`~/Library/Application Support/Vital/Factory/Presets/` on macOS)
2. Parse each .vital file, extract wavetable data from `settings.wavetables`
3. Deduplicate by content hash (SHA256 of base64 blob), not just name
4. Build `WavetableCatalog`: name → (index, base64_data), persist to JSON for reuse
5. For community presets: compare embedded WT data against catalog hash; flag custom WTs

To inject a wavetable during rendering:
1. Get current synth JSON via `synth.to_json()`
2. Replace the wavetable entry for the target oscillator
3. Reload via `synth.load_json()`

### Modulation Handling

- **Tier 1-2**: `synth.clear_modulations()` before every render. No modulation stored.
- **Tier 3**: Sparse connections via `connect_modulation(source, dest)` + per-slot `modulation_N_amount/bipolar/power/stereo`. Stored as dense `(N, 4, 32, 428)` matrix (4 channels: amount/bipolar/power/stereo; 32 mod sources; 428 destinations). Zeros compress well with gzip.
- For synthetic presets: connections sampled from `Geometric(p=0.15)` clamped to [0, 20].
- For community presets: modulation extracted from `.vital` file's `modulations` array + per-slot scalar params. Bypassed slots (`modulation_N_bypass > 0.5`) are skipped.
- 22 destinations in `MOD_DEST_BLOCKLIST` crash Vita and are filtered out.

### Rendering

All presets (synthetic and community) follow the same param-by-param flow.
No `_vital_json` direct loading — every param is set individually so the dataset
label exactly reflects what was sent to the synth.

Each `RenderEngine` (one per worker process) wraps a `vita.Synth` instance:
1. **Reset** to init state (load default preset JSON)
2. **Apply params** — continuous: denormalize `raw = min + norm * (max - min)`, then `ctrl.set(raw)`. Categoricals (including `_on` switches): `ctrl.set(float(index))`. Control ranges cached at engine init.
3. **Inject wavetables** from catalog via JSON manipulation
4. **Inject non-scalar data** (`_lfos`, `_sample`, `_wavetables`) via JSON — these can't be set via controls
5. **Apply modulation** — `connect_modulation(source, dest)` + per-slot amount/bipolar/power/stereo
6. **Render** at each MIDI note (48, 60, 72) — 3 samples per preset
7. **Validate** each render (reject silent, clipping, NaN)

Multi-pitch rendering captures key-tracking, velocity-sensitive, and filter behavior variation.

### HDF5 Schema

```
dataset.h5
├── audio/
│   └── waveforms          (N, 2, 88200)  float32  stereo, chunked (64,2,88200), gzip-4
├── params/
│   ├── continuous         (N, 322)        float32  normalized [0,1] via linear (raw-min)/(max-min)
│   ├── categorical        (N, 126)        int32    category indices (includes _on switches)
│   └── modulation_t3      (N, 4, 32, 428) float32  tier 3: dense (channels × sources × dests)
├── metadata/
│   ├── midi_note          (N,)            int32
│   ├── source             (N,)            S10      "synthetic"/"community"/"factory"
│   ├── tier               (N,)            int32
│   ├── preset_hash        (N,)            S64      SHA256 of param dict (for dedup)
│   └── preset_name        (N,)            S80
├── schema/                                (HDF5 attributes)
│   ├── sample_rate, render_duration, note_duration
│   ├── continuous_names, categorical_names, wavetable_names
│   ├── mod_source_names
│   └── version            "2.1.0"
└── importance_weights/                    (optional, computed separately)
    └── weights            (C,)            float32
```

---

## Implementation Phases

### Phase 1: Foundation (skeleton + config + params + wavetables + rendering)

- **Step 1.1**: `datagen/` package skeleton, all `__init__.py`, `pyproject.toml`, `config.py`
- **Step 1.2**: `ParamRegistry` with `from_synth()` and `from_config()`, `normalize.py`
- **Step 1.3**: Wavetable discovery + `WavetableCatalog` with JSON persistence
- **Step 1.4**: `RenderEngine` wrapping vita.Synth, `AudioValidator`

### Phase 2: End-to-End Tier 1

- **Step 2.1**: `ParamSampler` with LHS + heuristic constraints
- **Step 2.2**: HDF5 `schema.py` + `writer.py` + `reader.py`
- **Step 2.3**: `Pipeline` orchestrator (single-process)
- **Step 2.4**: Click CLI (`generate`, `inspect`, `discover-wt`, `download-presets`, `compute-weights`)
- **Step 2.5**: `RenderPool` multiprocessing with `maxtasksperchild=500`
- **Step 2.6**: 1K validation run

### Phase 3: Preset Ingestion

- **Step 3.1**: `PresetIngester.scan()` for community .vital files
- **Step 3.2**: Factory preset ingestion from Vital install dirs
- **Step 3.3**: `download_presets` from 3 known GitHub repos
- **Step 3.4**: Mixed generation run (synthetic + community/factory)

### Phase 4: Tier 2 + 3

- **Step 4.1**: Tier 2 fixed modulation (8 routings, amounts + active flags)
- **Step 4.2**: Tier 3 full params + dense modulation matrix
- **Step 4.3**: Perturbation-based importance weights

### Phase 5: Scale and Harden

- **Step 5.1**: Checkpointing (resume from last flushed chunk), retry logic, logging
- **Step 5.2**: 20K run: measure time, rejection rate, file size, profile bottlenecks
- **Step 5.3**: 100K run: consider 22050 Hz or shorter renders, optional HDF5 sharding

---

## Verification Plan

After each phase:
1. **Unit tests**: Param counts, normalization roundtrips, HDF5 read/write roundtrip, validator catches silent audio
2. **Integration**: Generate N samples, read back from HDF5, re-render 10 random samples from stored params
3. **1K validation run** (Phase 2.6): Rejection rate should be 30-40%. Spot-listen to confirm non-degenerate audio.
4. **Inspect command**: `python -m datagen inspect data/val_1k.h5`

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Wavetable discovery fails (paths differ per OS/version) | Fallback: user provides --extra-file pointing to any .vital file with factory WTs |
| Vita crashes on edge-case parameter combos | Per-render 5s timeout, `maxtasksperchild=500`, log and skip |
| High rejection rate >50% | Heuristic sampling constraints (osc level > 0.3, filter cutoff > 0.1) |
| HDF5 corruption on crash | Flush every 256 samples; resume from last valid index |
| Community presets use different WT naming conventions | Compare by content hash (SHA256 of base64 blob), not name |

---

## CLI Usage

```bash
# Discover wavetables
python -m datagen discover-wt -o data/wavetable_catalog.json

# Download community presets
python -m datagen download-presets -o presets/community/

# Generate Tier 1 dataset (1K samples)
python -m datagen generate --tier 1 -n 1000 -o data/val_1k.h5

# Generate with community presets and multiprocessing
python -m datagen generate --tier 1 -n 5000 -o data/train.h5 \
    --community-dir presets/community/ --workers 4

# Inspect a dataset
python -m datagen inspect data/val_1k.h5

# Compute importance weights
python -m datagen compute-weights --n-base 500 -o data/train.h5
```
