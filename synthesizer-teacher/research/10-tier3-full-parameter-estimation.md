# 10: Tier-3 Full Parameter Estimation Implementation Report

## Problem Statement

Despite training convergence (loss decreasing), rendered predictions didn't match targets. Root cause analysis identified four gaps:

1. **Wavetable selection predicted but never applied at inference.** Model trains on `osc_N_wavetable` categorical params but `InferencePipeline` never injected wavetable data into presets.
2. **Modulation routing stored in HDF5 but ignored by training.** `VitalDataset.__getitem__` returned only `(mel, cont, cat, midi)`. The `params/modulation_t3` dataset `(N, 4, 32, 406)` sat unused.
3. **Loss masking actively prevented learning mod-relevant params.** `_STATIC_MASK_PREFIXES` zeroed loss for `lfo_*`, `random_*`, `env_3-6_*` even for tier-3 data where modulation makes them audible.
4. **Render evaluation built presets without wavetables or modulation**, so W&B audio logs showed misleading comparisons.

## Changes Made

### Phase 1: Fix Loss Masking for Tier-3

**Files:** `training/config.py`, `training/loss.py`, `training/trainer.py`

- Added `tier: int = 0` to `TrainConfig` (0 = auto-detect from HDF5 schema).
- `Trainer.__init__` reads tier from `schema.get("tier", 1)` and passes to `VitalLoss`.
- `VitalLoss.__init__` accepts `tier` param. When `tier >= 3`, `_STATIC_MASK_PREFIXES` are skipped entirely (LFOs, randoms, env 3-6 are audible via modulation routing). The dynamic `_module_on_map` gating (for `*_on` switches) is kept for all tiers since disabled-module params are always inaudible.

### Phase 2: Fix Wavetable Application at Inference

**Files:** `datagen/wavetables/catalog.py`, `training/trainer.py`, `inference/pipeline.py`

**Catalog serialization:**
- Added `WavetableCatalog.to_json_string()` and `WavetableCatalog.from_json_string()` for embedding catalogs in checkpoints without requiring the original JSON file.

**Trainer changes:**
- `__init__`: loads wavetable catalog from `--wavetable-catalog` CLI flag or auto-detects from HDF5 schema `wavetable_catalog_path` attr.
- `_save_checkpoint`: embeds `wavetable_catalog_json` (full catalog as string) in checkpoint state dict.
- `_render_and_log_audio`: injects `_wavetable_catalog` into both target and predicted preset dicts so RenderEngine applies correct wavetables during W&B audio eval.

**Inference pipeline changes:**
- `from_checkpoint`: loads `wavetable_catalog_json` from checkpoint, reconstructs `WavetableCatalog`. Stored as `self.wavetable_catalog`.
- `render_comparison`: passes `_wavetable_catalog` to RenderEngine via the preset dict.
- `export_vital_preset`: for each oscillator (1-3), looks up predicted wavetable index in catalog, parses the JSON-serialized wavetable object, and injects into `settings["wavetables"]` array.
- `predict_with_refinement`: attaches `_wavetable_catalog` to CMA-ES categorical params so renders use correct wavetables.

### Phase 3: Add Modulation Prediction Head

**This is the core new capability.** Uses a factored bilinear head (~7.2M params).

#### 3a: Model Architecture (`training/model.py`)

Added `ModulationHead` class:
- Source/destination embeddings via linear projections from 512-dim backbone features.
- 4 bilinear scoring matrices (one per channel: amount, bipolar, power, stereo).
- Output: `(B, 4, n_sources, n_destinations)`.
- Parameter count: ~7.2M (source_proj: 512x32x32=524K, dest_proj: 512x406x32=6.6M, bilinear: 4x32x32=4K).

Modified `VitalInverseModel`:
- New args: `n_mod_sources: int = 0`, `n_mod_destinations: int = 0`, `mod_embed_dim: int = 32`.
- When both > 0, creates `self.modulation_head = ModulationHead(...)`.
- Forward now always returns 3-tuple: `(continuous_pred, categorical_logits, modulation_pred)`. `modulation_pred` is `None` when no head exists.

#### 3b: Dataset Loading (`training/dataset.py`)

- `VitalDataset.__getitem__` now returns 5-tuple: `(mel, cont, cat, midi_note, modulation)`.
- `_has_modulation` flag detected lazily on first file open by checking for `features/modulation_t3` or `params/modulation_t3` in HDF5.
- Prefers uncompressed `features/modulation_t3` (fast random access), falls back to compressed `params/modulation_t3`.
- When no modulation data exists, returns `torch.empty(0)` as sentinel.

Added `precompute_modulation()`:
- Copies gzip-compressed `params/modulation_t3` to uncompressed `features/modulation_t3` with per-sample chunking.
- Same pattern as `precompute_mels` — run once before training.
- For 200K samples x 4 x 32 x 406 x 4 bytes = ~41 GB uncompressed. Loading all into RAM is infeasible; uncompressed HDF5 with per-sample chunks gives fast random access.

#### 3c: Modulation Loss (`training/loss.py`)

Added `ModulationLoss` class with sparsity-aware design:
- **Presence loss**: BCE with `pos_weight=20.0` on `|amount|` vs binary active mask. High pos_weight compensates for extreme sparsity (~99.9% of connections are inactive).
- **Amount MSE**: Only on active connections (where `|target_amount| > 1e-6`).
- **Power MSE**: Only on active connections.
- **Bipolar BCE**: Binary classification, only on active connections.
- **Stereo BCE**: Binary classification, only on active connections.
- All active-connection losses normalized by `n_active.clamp(min=1)`.

Integrated into `VitalLoss`:
- New params: `modulation_loss_weight` (default 0.0), `modulation_pos_weight` (20.0), `modulation_warmup_epochs` (5).
- `forward()` now accepts `modulation_pred`, `modulation_target`, and `current_epoch`.
- Returns 4-tuple: `(total, cont_loss, cat_loss, mod_loss)`.
- Total = `cont_weight * cont + cat_weight * cat + effective_mod_weight * mod`.
- **Warmup**: `effective_mod_weight = modulation_loss_weight * min(current_epoch / warmup_epochs, 1.0)`. At epoch 0, mod weight is 0; linearly ramps to full weight over `warmup_epochs`.

#### 3d: Trainer Integration (`training/trainer.py`)

- `__init__`: reads `n_mod_sources`, `n_mod_destinations` from HDF5 by inspecting the modulation dataset shape. Reads `mod_source_names` and `mod_destination_names` from schema attrs. Passes to `VitalInverseModel` constructor. Sets `modulation_loss_weight` to configured value only when modulation data exists (0.0 otherwise).
- `_train_epoch(epoch)`: unpacks 5-tuple from dataloader. Passes `mod_target` and `mod_pred` to loss. Passes `current_epoch=epoch` for warmup. Logs `train/mod_loss`.
- `_validate_epoch(epoch)`: same 5-tuple unpacking. Accumulates `all_mod_pred` and `all_mod_target` for metrics. After validation, calls `compute_modulation_metrics()` for precision/recall/amount_mae. Logs `val/mod_loss`, `val/mod_precision`, `val/mod_recall`, `val/mod_amount_mae`.
- `_save_checkpoint`: adds `tier`, `wavetable_catalog_json`, `n_mod_sources`, `n_mod_destinations`, `mod_source_names`, `mod_destination_names` to state dict.
- `_render_and_log_audio`: uses `_dense_to_connections()` helper to convert dense `(4, n_src, n_dst)` matrices to connection dicts. Attaches `_modulation_t3` to both target and predicted presets for rendering.
- `_dense_to_connections(mod_matrix, threshold=0.05)`: iterates over amount channel, extracts connections where `|amount| > threshold`, maps source/dest indices to names.

### Phase 4: Update Inference for Modulation

**Files:** `inference/pipeline.py`, `inference/cma_optimizer.py`

**Pipeline:**
- `from_checkpoint`: detects modulation head from `n_mod_sources` in checkpoint. Creates model with modulation head if present. Loads `mod_source_names` and `mod_destination_names`.
- `_extract_modulation(mod_matrix, threshold=0.05, top_k=20)`: converts dense `(4, n_src, n_dst)` tensor to connections dict. Thresholds by `|amount|`, sorts by magnitude, keeps top-K. Each connection includes: `source`, `destination`, `source_idx`, `dest_idx`, `amount`, `bipolar`, `power`, `stereo`. Return dict includes `n_sources`, `n_destinations`.
- `predict()` and `predict_with_confidence()`: after forward pass, extract modulation connections and store as `params["_modulation_t3"]`.
- `render_comparison`: `_modulation_t3` is already in params dict, passed through to RenderEngine.
- `export_vital_preset`: builds `modulations` array (64 slots, most empty), populates active slots with source/dest names, sets `modulation_N_amount/bipolar/power/stereo` in settings.
- `predict_with_refinement`: passes `_modulation_t3` and `_wavetable_catalog` to CMA-ES via `categorical_params` dict.

**CMA-ES optimizer:**
- Type hint updated from `dict[str, int]` to `dict[str, Any]` for `categorical_params`.
- Docstring documents that `_wavetable_catalog` and `_modulation_t3` may be present in `categorical_params` and are passed through to RenderEngine.
- Modulation routing stays fixed from neural net prediction; CMA-ES optimizes continuous params only.

### Phase 5: Update Evaluation

**Files:** `training/evaluate.py`

- Added `compute_modulation_metrics(pred, target, threshold=0.05)`: computes connection precision, recall, and amount MAE on truly active connections.
- `render_eval_batch`: handles 5-tuple dataset output and 3-tuple model output. Returns `target_modulation` and `pred_modulation` in result dicts.

### Phase 6: CLI & Workflow Integration

**Files:** `training/cli.py`, `training/config.py`

New `TrainConfig` fields:
- `wavetable_catalog: str = ""` — path to catalog JSON, or empty for auto-detect.
- `modulation_loss_weight: float = 0.3`
- `modulation_pos_weight: float = 20.0`

New CLI command: `precompute-modulation` — copies compressed modulation data to uncompressed HDF5.

New train flags: `--tier`, `--wavetable-catalog`, `--modulation-loss-weight`, `--mod-pos-weight`.

Updated `evaluate` command: loads model with modulation head from checkpoint, uses indexed tuple access for dataset samples.

Updated `eval_tui.py`: unpacks 3-tuple model output `(cont, cat, _mod)`.

## Architecture Summary

### Data Flow (Training)

```
HDF5 Dataset
  params/continuous       (N, 322)     float32
  params/categorical      (N, 126)     int64
  params/modulation_t3    (N, 4, 32, 406) float32 [gzip]
  features/mel_spectrogram (N, 1, 128, T)  float32 [uncompressed]
  features/modulation_t3   (N, 4, 32, 406) float32 [uncompressed]
       |
       v
VitalDataset.__getitem__ -> (mel, cont, cat, midi, mod) 5-tuple
       |
       v
VitalInverseModel.forward(mel) -> (cont_pred, cat_logits, mod_pred) 3-tuple
       |
       v
VitalLoss.forward(cont, cat, mod, epoch) -> (total, cont_loss, cat_loss, mod_loss) 4-tuple
```

### Data Flow (Inference)

```
Audio file -> mel spectrogram
       |
       v
VitalInverseModel.forward(mel) -> (cont_pred, cat_logits, mod_pred)
       |
       v
_extract_modulation(mod_pred, threshold=0.05, top_k=20)
       |
       v
params dict with _modulation_t3 connections + _wavetable_catalog
       |
       +---> export_vital_preset() -> .vital JSON with wavetables + modulations
       |
       +---> render_comparison() -> Vita renders with full modulation + wavetables
       |
       +---> predict_with_refinement() -> CMA-ES with fixed modulation routing
```

### Model Parameter Budget

| Component | Params | Notes |
|-----------|--------|-------|
| ResNet-18 backbone | ~11.2M | conv1 adapted to 1ch |
| Continuous head (3-layer MLP) | ~530K | 512->512->512->322 + sigmoid |
| Categorical heads (126 linear) | ~65K-300K | 512->n_opts each |
| **Modulation head** | **~7.2M** | source_proj + dest_proj + 4 bilinear |
| **Total** | **~19M** | Well within 16GB VRAM |

### Loss Function Breakdown

```
L_total = w_cont * L_cont + w_cat * L_cat + warmup(epoch) * w_mod * L_mod

L_cont = importance-weighted MSE with conditional masking
L_cat  = mean per-param cross-entropy with label smoothing
L_mod  = presence_BCE + amount_MSE + power_MSE + bipolar_BCE + stereo_BCE
         (all active-connection losses normalized by n_active)

warmup(epoch) = min(epoch / warmup_epochs, 1.0)
```

## Checkpoint Format (New Fields)

The following keys were added to the checkpoint state dict:

```python
{
    # ... existing keys ...
    "tier": int,                        # Training tier (1, 2, or 3)
    "wavetable_catalog_json": str | None,  # Full catalog JSON for portability
    "n_mod_sources": int,               # 0 for no modulation head
    "n_mod_destinations": int,          # 0 for no modulation head
    "mod_source_names": list[str],      # Modulation source parameter names
    "mod_destination_names": list[str], # Modulation destination parameter names
}
```

## Verification Results

All syntax checks pass. Unit tests confirm:

- **Tier masking**: Tier 1 correctly masks 5 LFO params (indices 0-4). Tier 3 has empty static mask (LFOs/randoms/envs are learnable).
- **Model output**: Without mod head: `(B, 10)`, `[list]`, `None`. With mod head: `(B, 10)`, `[list]`, `(B, 4, 32, 406)`.
- **Modulation loss**: Zero when no mod head. Non-zero with mod head + target. Warmup correctly ramps: epoch 0 -> w=0.0, epoch 1 -> w=0.06, epoch 5 -> w=0.3, epoch 10 -> w=0.3 (for warmup_epochs=5, configured weight=0.3).
- **Modulation metrics**: precision=0.5, recall=0.5, amount_mae=0.2 for test case with 1 TP, 1 FP, 1 FN.
- **Total model params**: ~18.9M with modulation head (7.2M from mod head = 38.1%).
