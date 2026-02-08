# Report 8: Bug Fixes (2-6) + Tier 3 CMA-ES Implementation

## Summary

This report documents the resolution of 5 bugs identified in the second audit (report 7) and the implementation of Tier 3 CMA-ES inference-time optimization. Bug 1 was confirmed as a false positive and received only a clarifying comment.

---

## Part A: Bug Fixes

### Bug 1: Gradient Accumulation Metric (FALSE POSITIVE)

**File**: `training/trainer.py:474`
**Status**: Confirmed correct; added clarifying comment.

The metric tracking uses `loss.item()` (the unscaled loss), not `scaled_loss`. This is correct because each micro-batch loss has the same magnitude as a validation batch loss (same batch size). `scaled_loss = loss / effective_accum` is only used for gradient accumulation backward passes, not for metric reporting.

**Change**: Added a comment explaining the rationale.

### Bug 2: Short Audio Crashes Inference

**File**: `inference/pipeline.py`, `_load_and_compute_mel()`
**Root Cause**: Audio shorter than `n_fft` (2048 samples, ~46ms at 44.1kHz) would crash the mel spectrogram transform with an opaque error.

**Fix**:
- Added minimum length check after mono conversion and 4-second truncation
- Requires `waveform.shape[-1] >= n_fft` (2048 samples)
- Raises `ValueError` with user-friendly message: actual duration, minimum required, and sample counts
- Also added `log.info()` when 4-second truncation occurs (previously silent)

### Bug 3: DDP Validation Metrics Bias

**File**: `training/trainer.py`, `_validate_epoch()`
**Root Cause**: `compute_per_group_mse()` and `compute_categorical_accuracy()` were guarded by `is_main_process()`, so they only reflected rank 0's validation shard (~1/N of data) rather than the full validation set.

**Fix**:
- All ranks now compute group MSE and categorical accuracy on their local shards
- **Group MSE**: Values packed into a tensor, `dist.all_reduce(SUM)`, divided by `world_size` (average of per-shard MSEs approximates global MSE since `DistributedSampler` distributes evenly)
- **Categorical accuracy**: Correct counts and total counts packed into tensors, both all-reduced with SUM, then divided to get exact global accuracy
- Results only assigned to `metrics` dict on rank 0 (for W&B logging)
- Non-DDP path unchanged

### Bug 4: Silent Export of Unnormalized Values

**File**: `inference/pipeline.py`, `export_vital_preset()`
**Root Cause**: When `continuous_ranges` was `None` (old checkpoint without ranges) or incomplete, the export silently wrote normalized [0,1] values into the preset instead of denormalized raw Vital values. The resulting preset would sound wrong with no indication of why.

**Fix**:
- Added `warned_missing_ranges` flag to log warnings once per export
- When `continuous_ranges` is `None`: warns that values are exported as normalized [0,1]
- When `continuous_ranges` exists but index is out of range: warns that ranges are incomplete
- Does not spam — only one warning per export call

### Bug 5: Silent Modulation Skipping

**File**: `datagen/presets/ingest.py`, `_extract_modulation()`
**Root Cause**: Blocked destinations and unknown source/destination names were logged at `DEBUG` level, making them invisible in normal operation. Users ingesting community presets had no indication that modulation connections were being silently dropped.

**Fix**: Changed `log.debug(...)` to `log.info(...)` for:
- Blocked modulation destinations (from `MOD_DEST_BLOCKLIST`)
- Unknown modulation sources
- Unknown modulation destinations

### Bug 6: Hardcoded Sample Rate in Preview

**File**: `datagen/preview.py`
**Root Cause**: `_play()` hardcoded `samplerate=44100` regardless of dataset sample rate. Datasets generated at other rates (e.g. 22050Hz) would play at wrong speed.

**Fix**:
- Added `self._sample_rate = 44100` as instance variable default
- In `_open()`: reads `sample_rate` from `schema.attrs.get("sample_rate", 44100)`
- In `_play()`: uses `self._sample_rate` instead of hardcoded `44100`

---

## Part B: Tier 3 CMA-ES Implementation

### Overview

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a derivative-free optimization algorithm used here for inference-time refinement. After the neural network predicts initial parameters, CMA-ES iteratively mutates the continuous parameters, renders each candidate through Vita, and compares the rendered audio against the target using multi-resolution STFT loss.

This bridges the gap between parameter-space prediction and perceptual audio quality, which is critical because:
1. The many-to-one mapping (multiple parameter settings produce identical sounds) means exact parameter match is less important than perceptual match
2. The model does NOT predict modulation routing (only continuous + categorical params), so CMA-ES compensates
3. Derivative-free optimization works with the non-differentiable Vita renderer

### New File: `inference/cma_optimizer.py`

`CMAESOptimizer` class (~120 lines):

- **`__init__(engine, continuous_names, sample_rate)`**: Stores a reusable `RenderEngine`, parameter names, and sample rate. MRSTFT loss is lazy-initialized.

- **`_get_mrstft()`**: Lazily creates and caches an `auraloss.freq.MultiResolutionSTFTLoss` with FFT sizes [1024, 2048, 8192], mel-scale bins, and perceptual weighting.

- **`_objective(x, categorical_params, target_mono, midi_note)`**: Core fitness function. Clips candidate to [0,1], builds preset dict, renders via engine, computes MRSTFT loss against target. Returns 1e6 for failed renders or audio < 1024 samples.

- **`optimize(initial_continuous, categorical_params, target_mono, ...)`**: Runs `cma.CMAEvolutionStrategy` with bounds [0,1], configurable `max_evals` (default 500), `timeout_sec` (default 60), and `sigma0` (default 0.1). Returns `(optimized_vector, info_dict)` where info contains initial/final loss, n_evals, elapsed time, and whether improvement occurred.

### Modified: `inference/pipeline.py`

Added `predict_with_refinement()` method:

1. Calls `predict_with_confidence()` for initial params + categorical confidence
2. Loads raw target audio via torchaudio, converts to mono numpy
3. Creates `RenderEngine` and `CMAESOptimizer` (import-guarded for Vita availability)
4. Extracts continuous vector + categorical dict from predicted params
5. Runs `optimizer.optimize()` with configurable sigma, max_evals, timeout
6. Rebuilds params dict with refined continuous values
7. Returns `(params, confidence, refinement_info)`

Gracefully degrades: returns `{"skipped": True, "reason": "vita_unavailable"}` if Vita can't be imported.

### Modified: `inference/cli.py`

Added 3 new options to `infer` command:
- `--refine / --no-refine` (default: False) — enables CMA-ES refinement
- `--refine-evals INT` (default: 500) — max function evaluations
- `--refine-timeout FLOAT` (default: 60.0) — timeout in seconds
- `--refine-sigma FLOAT` (default: 0.1) — initial step size

When `--refine` is set, calls `predict_with_refinement()` and logs improvement metrics.

### Modified: `inference/demo.py`

- Added `gr.Checkbox(label="CMA-ES Refinement (slower, more accurate)")` to UI
- Updated `predict_and_analyze()` to accept `use_cmaes` parameter
- When enabled, calls `predict_with_refinement()` with try/except fallback
- Displays refinement stats in params output: initial loss, final loss, n_evals, time, improved

### Modified: `inference/__init__.py`

Added `CMAESOptimizer` to module exports.

### Modified: `pyproject.toml`

Added `cma` and `auraloss` to inference extras (both were already in train extras but inference needs them too for CMA-ES).

---

## Files Changed

| File | Change Type | Lines Changed | Description |
|------|------------|--------------|-------------|
| `training/trainer.py` | Edit | +50 | Bug 1 comment, Bug 3 DDP metrics all-reduce |
| `inference/pipeline.py` | Edit | +70 | Bug 2 min length check, Bug 4 export warnings, `predict_with_refinement()` |
| `datagen/presets/ingest.py` | Edit | +3 | Bug 5 log level elevation (debug -> info) |
| `datagen/preview.py` | Edit | +3 | Bug 6 sample rate from schema |
| `inference/cma_optimizer.py` | **New** | 130 | CMA-ES optimizer with MRSTFT objective |
| `inference/cli.py` | Edit | +25 | `--refine` options and branching logic |
| `inference/demo.py` | Edit | +30 | CMA-ES checkbox and refinement stats display |
| `inference/__init__.py` | Edit | +2 | Export `CMAESOptimizer` |
| `pyproject.toml` | Edit | +2 | `cma` and `auraloss` in inference extras |

---

## Tier 3 Training Recommendations

The model architecture is identical across all tiers. For tier 3:

### Data Generation
```bash
python -m datagen generate --tier 3 -n 200000 -o data/tier3_200k.h5 --workers 16
python -m training precompute-mels -d data/tier3_200k.h5 --device cuda
```

### Training (Single GPU)
```bash
python -m training train \
  -d data/tier3_200k.h5 \
  --epochs 200 \
  --batch-size 32 \
  --lr 1e-4 \
  --label-smoothing 0.1 \
  --no-freeze \
  --dropout 0.1 \
  --early-stopping-patience 15 \
  --gradient-accumulation-steps 4 \
  --num-workers 8 \
  --compute-spectral-metrics \
  --log-audio-every 5 \
  --device cuda
```

### Training (Multi-GPU, 4x)
```bash
torchrun --nproc_per_node=4 -m training train \
  -d data/tier3_200k.h5 \
  --epochs 200 \
  --batch-size 32 \
  --lr 4e-4 \
  --label-smoothing 0.1 \
  --no-freeze \
  --dropout 0.1 \
  --early-stopping-patience 15 \
  --gradient-accumulation-steps 4 \
  --num-workers 8 \
  --compute-spectral-metrics \
  --log-audio-every 5
```

### Key Flag Rationale
- **`--no-freeze`**: Tier 3 has ~322 continuous + 126 categorical outputs; unfreezing the backbone helps extract richer features
- **`--label-smoothing 0.1`**: 126 categorical params with varying class counts benefit from smoothing
- **`--gradient-accumulation-steps 4`**: Effective batch 128 stabilizes gradients over the large output space
- **`--early-stopping-patience 15`**: Tier 3 converges slower; needs more patience
- **`--lr 4e-4` (multi-GPU)**: LR is NOT auto-scaled with DDP; manually scale proportional to world_size
- **Conditional loss masking** (on by default): Less impactful for tier 3 since modulation routing makes LFO/envelope/random params audible, but still masks params gated by disabled `*_on` switches

### Inference with CMA-ES
```bash
python -m inference infer \
  -c checkpoints/best_model.pt \
  -i target.wav \
  --refine --refine-evals 500 --refine-timeout 60 \
  --render --tutorial
```

---

## Verification Checklist

1. **Bug 2**: Inference on audio <50ms raises `ValueError` with duration info
2. **Bug 3**: DDP training produces globally accurate group_mse and cat_accuracy
3. **Bug 4**: Old checkpoint without `continuous_ranges` logs a warning on export
4. **Bug 5**: Ingesting community preset with blocked mod shows INFO log
5. **Bug 6**: Dataset at non-44100 sample rate plays at correct speed in preview
6. **CMA-ES CLI**: `python -m inference infer -c ckpt.pt -i test.wav --refine --render`
7. **CMA-ES Demo**: Gradio checkbox triggers refinement, shows initial/final loss stats
