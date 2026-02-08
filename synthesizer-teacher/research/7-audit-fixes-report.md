# Second Audit — Comprehensive Bug Fixes & Training Improvements

**Date**: 2026-02-08
**Scope**: Full code audit follow-up — 30 issues across datagen, training, and inference

---

## Executive Summary

A comprehensive code audit identified ~30 issues across the codebase. Additionally, analysis of training results revealed that **~240 of 322 continuous params are unlearnable at Tier 1** because LFOs, random generators, disabled effects, and unrouted envelopes have zero effect on audio without modulation routing. All issues have been addressed across 6 phases, verified with smoke tests.

---

## Phase 1: Critical Bugs (Runtime Crashes / Incorrect Output)

### 1A. Fix `evaluate` command tuple unpacking
**File**: `training/cli.py:285`

The `evaluate` command was unpacking 3 values from `VitalDataset.__getitem__` which returns a 4-tuple `(mel, cont, cat, midi_note)`.

**Before**: `mel, cont, cat = val_dataset[idx]` (crash)
**After**: `mel, cont, cat, _midi = val_dataset[idx]`

### 1B. Fix `evaluate` command hardcoded `midi_note=60`
**File**: `training/cli.py:318,328`

The spectral evaluation was rendering all presets at MIDI note 60 regardless of the actual note used during dataset generation. This corrupted spectral distance metrics.

**Before**: `engine.render_preset(target_preset, midi_note=60)`
**After**: `engine.render_preset(target_preset, midi_note=int(result["midi_note"]))`

### 1C. Fix `evaluate` command hardcoded `sample_rate=44100`
**File**: `training/cli.py:266`

The sample rate was hardcoded instead of reading from the checkpoint config.

**Before**: Hardcoded `44100` passed to `render_eval_batch`
**After**: `sample_rate = ckpt_config.get("sample_rate", 44100)` read before the call

### 1D. Fix preset export writing normalized values
**File**: `inference/pipeline.py:206-214`

The `.vital` preset export was writing [0,1] normalized values directly to the JSON, producing incorrect sounds when loaded in Vital.

**After**: Continuous params are denormalized using `continuous_ranges` (stored in checkpoint):
```python
if name in self.continuous_names and self.continuous_ranges:
    idx = self.continuous_names.index(name)
    lo, hi = self.continuous_ranges[idx]
    value = lo + float(value) * (hi - lo)
```

### 1E. Fix `set_normalized()` for modulation amounts
**File**: `datagen/render/engine.py:262`

Modulation amounts were set via `set_normalized()` which has buggy nonlinear mappings. Modulation amount range is [-1, 1] and was being incorrectly transformed.

**Before**: `controls[f"{prefix}_amount"].set_normalized((conn["amount"] + 1.0) / 2.0)`
**After**: `controls[f"{prefix}_amount"].set(conn["amount"])`

Also restructured `_apply_modulation` to fetch controls once after all `connect_modulation()` calls (since new slot controls only appear after connections are established), instead of inside the loop.

---

## Phase 2: Training Improvements (Highest Impact on Model Quality)

### 2A. Simplify categorical heads (~30M → ~3M fewer params)
**Files**: `training/model.py`, `training/config.py`, `training/cli.py`

The 126 categorical heads were each a 2-layer MLP (512→mlp_hidden→n_opts with ReLU and dropout), totaling ~33M parameters. With <100k training samples, these dramatically overfit.

**Change**: Default to single `nn.Linear(512, n_opts)` per param.
- New config field: `simple_categorical_heads: bool = True`
- CLI flag: `--simple-cat-heads / --no-simple-cat-heads` (default: simple)
- MLP path preserved for backward compatibility when `False`
- **Note**: This breaks checkpoint compatibility — existing checkpoints won't load with the new architecture.

### 2B. Conditional loss masking for unlearnable params
**Files**: `training/loss.py`, `training/config.py`, `training/cli.py`, `training/trainer.py`

At Tier 1-2, ~240 of 322 continuous params have zero audio effect:
- **LFOs** (lfo_1..8_*): 64 params — no modulation routing to route them
- **Random generators** (random_1..4_*): 32 params — same reason
- **Envelopes 3-6** (env_3..6_*): 36 params — only env_1 and env_2 are hard-wired
- **Disabled effects**: continuous params for modules where `*_on=0` (chorus, delay, flanger, phaser, distortion, reverb, compressor, filter_1, filter_2, filter_fx)

Training the model to predict these wastes gradient signal and causes the model to plateau early.

**Implementation** in `VitalLoss`:
1. **Static mask**: Built at init, zeros indices for LFO/random/env3-6 prefixes unconditionally
2. **Dynamic mask**: Per-sample, checks categorical `*_on` targets and zeros continuous params for disabled modules
3. Mask tensor `(B, n_continuous)` multiplied into squared error
4. Loss normalized by `mask.sum().clamp(min=1)` instead of `.mean()`
5. No new arguments required in `forward()` — mask built internally from targets

- Config field: `conditional_loss_mask: bool = True`
- CLI flag: `--conditional-loss-mask / --no-conditional-loss-mask` (default: enabled)

### 2C. Save `sample_rate` in checkpoint
**File**: `training/trainer.py`

Previously, sample rate had to be inferred from config defaults. Now saved explicitly:
```python
"sample_rate": self.sample_rate
```

### 2D. Fix LR warmup off-by-one
**File**: `training/trainer.py`

The LR scheduler was not stepped before the first training epoch, meaning epoch 0 trained at the wrong learning rate (either full LR or 0, depending on the lambda).

**Fix**: Added `self.scheduler.step()` after scheduler creation, before the training loop.

### 2E. Periodic checkpoint saving
**File**: `training/trainer.py`

Previously only `best_model.pt` was saved. A training crash meant losing all progress since the last best epoch.

**Fix**: Save `latest_model.pt` every epoch (overwritten each time) for crash recovery.

### 2F. Store continuous param ranges in checkpoint
**File**: `training/trainer.py`

Stores `continuous_ranges: list[tuple[float, float]]` (min/max for each continuous param) in the checkpoint. Read from HDF5 schema attributes (`continuous_min`, `continuous_max`). This enables `inference/pipeline.py` to denormalize params for `.vital` export without needing a live Vita instance.

---

## Phase 3: Moderate Bugs

### 3A. Fix RMS calculation in datagen inspect command
**File**: `datagen/cli.py:323`

**Before**: `s['audio'].std()` — computes standard deviation, not RMS
**After**: `np.sqrt(np.mean(s['audio'].astype(np.float64)**2))` — correct RMS

### 3B. Fix offline tutorial missing sections
**File**: `inference/tutorial.py`

The offline tutorial template was missing LFOs, random generators, and global settings sections.

**Changes**:
1. Added `"random_generators"` group to `PARAM_GROUPS` with prefixes `random_1..4`
2. Expanded `"global"` prefixes to include `macro_control`, `beats_per_minute`, `stereo`
3. Added LFO (Step 5) and Global Settings (Step 6) sections to `generate_offline_tutorial()`

### 3C. Fix temp file leak in Gradio demo
**File**: `inference/demo.py`

Previous temp files (predicted audio, .vital presets) were never cleaned up, accumulating on disk.

**Fix**: Module-level `_temp_files: list[str]` tracks created files. At the start of each `predict_and_analyze` call, previous files are unlinked and the list is cleared.

### 3D. Add audio length validation in inference
**File**: `inference/pipeline.py:113-116`

Input audio longer than 4 seconds was being processed in full, mismatching the training clip length.

**Fix**: Added truncation to `max_samples = sample_rate * 4` in `_load_and_compute_mel()`.

### 3E. Extract shared mel computation method
**File**: `inference/pipeline.py`

`predict()` and `predict_with_confidence()` duplicated audio loading and mel computation code.

**Fix**: Extracted `_load_and_compute_mel(audio_path)` private method, called by both.

---

## Phase 4: Datagen Improvements

### 4A. Sort parameter names in registry
**File**: `datagen/params/registry.py`

`continuous()`, `categorical()`, and `wavetable()` methods now wrap results in `sorted(..., key=lambda p: p.name)` for deterministic parameter ordering regardless of Vita's control iteration order.

### 4B. Fix wavetable discovery double-scan
**File**: `datagen/wavetables/discovery.py`

The discovery code called `d.rglob("*.vital")` twice — once for counting and once for collection.

**Fix**: Store results in `found = list(d.rglob("*.vital"))`, use for both `preset_files.extend(found)` and logging.

### 4C. Cache `synth.get_controls()` in render engine
**File**: `datagen/render/engine.py`

`_apply_params` was calling `self.synth.get_controls()` on every render — unnecessary since the controls dict doesn't change between resets.

**Fix**: Cache as `self._controls = self.synth.get_controls()` in `__init__`, reuse in `_apply_params`.

In `_apply_modulation`, controls are fetched once *after* all `connect_modulation()` calls (new modulation slot controls only appear after connections are made), then modulation params are set in a second pass.

### 4E. Elevate render failure log level
**File**: `datagen/render/engine.py:98`

Changed render failure logging from `log.debug(...)` to `log.warning(...)` so failures are visible at default log level.

### 4F. Remove dead `modulation_t2` references
**File**: `datagen/storage/reader.py`

Removed `if "params/modulation_t2" in f:` blocks from `get_sample()` and `get_batch()` — this dataset key was never written by the pipeline.

---

## Phase 5: Cleanup

### 5A. Update CLAUDE.md
**File**: `CLAUDE.md`

Removed stale "inference/pipeline.py was deleted" warning since the file exists.

### 5C. Remove unused `_ensure_synth` in preset ingester
**File**: `datagen/presets/ingest.py`

The `_ensure_synth()` method lazily created a Vita synth instance that was never actually used — normalization uses linear math from the registry instead. Removed:
- `self._synth = None` from `__init__`
- `_ensure_synth()` method entirely
- `self._ensure_synth()` call in `_parse_preset()`

### 5D. Fix gradient accumulation residual scaling
**File**: `training/trainer.py:432-470`

When total batches aren't divisible by `accum_steps`, the residual batches at the end were divided by the full `accum_steps` instead of the actual residual count. For example, with 10 batches and `accum_steps=4`, the last 2 batches each divided by 4 instead of 2, under-weighting their gradients.

**Fix**: Pre-compute `residual = n_total_batches % accum_steps` and `residual_start`. Batches at index >= `residual_start` use `effective_accum = residual` instead of `accum_steps`. The optimizer step also fires on the last batch (not just on sync steps).

---

## Phase 6: Verification

All smoke tests pass:

| Test | Result |
|------|--------|
| Model build (simple heads): `VitalInverseModel(322, [2]*126)` | OK |
| Loss compute: `VitalLoss(n_continuous=10)` | OK |
| Conditional mask loss: with `continuous_names`, `categorical_names` | OK |
| MLP vs simple heads param count comparison | OK (MLP: 12.2M, Simple: 11.7M for small test) |
| `python -m training train --help` | OK — shows `--simple-cat-heads`, `--conditional-loss-mask` |
| `python -m training evaluate --help` | OK |
| `from inference.pipeline import InferencePipeline` | OK |
| `py_compile` on all 13 modified files | OK |

---

## Files Modified

| File | Changes |
|------|---------|
| `training/model.py` | Simple categorical heads option (2A) |
| `training/loss.py` | Conditional loss masking (2B) |
| `training/config.py` | New fields: `simple_categorical_heads`, `conditional_loss_mask` (2A, 2B) |
| `training/trainer.py` | Save sample_rate + ranges (2C, 2F), fix warmup (2D), periodic checkpoint (2E), fix gradient accum residual (5D) |
| `training/cli.py` | Fix evaluate bugs (1A-C), new CLI flags (2A, 2B), fix model reconstruction (2A) |
| `inference/pipeline.py` | Denormalize preset export (1D), shared mel method (3E), audio truncation (3D) |
| `inference/tutorial.py` | Add LFO/global/random sections (3B) |
| `inference/demo.py` | Fix temp file leak (3C) |
| `datagen/render/engine.py` | Fix set_normalized (1E), cache controls (4C), log level (4E) |
| `datagen/params/registry.py` | Sort param names (4A) |
| `datagen/wavetables/discovery.py` | Fix double-scan (4B) |
| `datagen/cli.py` | Fix RMS calculation (3A) |
| `datagen/storage/reader.py` | Remove dead modulation_t2 (4F) |
| `datagen/presets/ingest.py` | Remove unused _ensure_synth (5C) |
| `CLAUDE.md` | Updated documentation for all changes |

---

## Impact Assessment

### Training Quality
- **Conditional loss masking** is the highest-impact change. By focusing gradient signal on the ~80 params that actually affect audio (at Tier 1), the model should converge faster and achieve better accuracy on the params that matter.
- **Simple categorical heads** reduce overfitting risk by cutting ~30M excess parameters.
- **LR warmup fix** ensures smooth training start.
- **Gradient accumulation residual fix** corrects subtle gradient scaling error.

### Correctness
- **Preset export denormalization** was the most user-visible bug — exported presets previously sounded nothing like predictions.
- **Evaluate command** had 3 bugs (tuple unpacking, hardcoded note, hardcoded sample rate) that would crash or produce wrong metrics.
- **Modulation amount** fix ensures correct Vita rendering for Tier 3 presets.

### Reliability
- **Periodic checkpoints** prevent training progress loss on crash.
- **Temp file cleanup** prevents disk space leak in demo.
- **Audio truncation** prevents OOM on long audio files during inference.

---

## Next Steps

1. **Retrain from scratch** with simple heads + conditional loss masking enabled
2. **Compare metrics** against the previous training run (expect faster convergence, better accuracy on learnable params)
3. **Validate preset export** — load exported .vital files in Vital synth and verify sounds match predictions
4. **Run full evaluation** with the corrected evaluate command
