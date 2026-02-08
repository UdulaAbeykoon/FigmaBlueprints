# Tier-1 Training Pipeline — Implementation Report

## Summary

Implemented a complete training pipeline for the Tier-1 inverse synthesis model: ResNet-18 on log-mel spectrograms predicting ~62 continuous + ~20 categorical Vital synthesizer parameters. The pipeline reads from the datagen HDF5 dataset, trains with importance-weighted MSE + per-param cross-entropy, logs to W&B, and supports checkpoint resume.

## Module Structure

```
training/
├── __init__.py
├── __main__.py          # python -m training support
├── cli.py               # Click CLI: precompute-mels, train, evaluate
├── config.py            # TrainConfig dataclass
├── dataset.py           # Precompute mels to disk, PyTorch Dataset, train/val split
├── model.py             # ResNet-18 backbone + continuous/categorical heads
├── loss.py              # Importance-weighted MSE + per-param cross-entropy
├── trainer.py           # Training loop with W&B logging + checkpoint resume
└── evaluate.py          # Per-group MSE, categorical accuracy, render eval, W&B audio
```

## Changes to Existing Files

### `datagen/storage/schema.py`
Added `categorical_n_options: list[int]` field to `HDF5Schema`. Populated in `from_config()` from the registry and included in `schema_attributes()` so it's written to HDF5. This lets the training pipeline know how many classes each categorical param has without requiring a live Vita instance.

### `pyproject.toml`
- Added `torchvision>=0.15` to `[project.optional-dependencies].train`
- Added `train = "training.cli:main"` to `[project.scripts]`
- Added `"training*"` to `[tool.setuptools.packages.find].include`

## Architecture Decisions

### Model: VitalInverseModel

```
Input: (B, 1, 128, 173) log-mel spectrogram

ResNet-18 backbone (ImageNet pretrained, conv1 averaged to 1ch):
    conv1 → layer1 → layer2  [frozen by default]
    layer3 → layer4 → avgpool [trainable]
    → (B, 512) feature vector

Continuous head (shared MLP):
    512 → 512 → 512 → n_continuous + Sigmoid → [0,1]

Categorical heads (nn.ModuleList, one per param):
    Default (simple): 512 → n_options → logits
    MLP mode:         512 → 512 → n_options → logits
```

- Separate categorical heads because each param has different n_options (2 for on/off, up to 8 for filter model, etc.)
- Default: simple linear heads (`--simple-cat-heads`, ~11.7M params). Use `--no-simple-cat-heads` for MLP heads (~12.2M params, higher overfitting risk with <100k samples).
- Conv1 adapted from 3ch by averaging pretrained RGB weights into 1 channel

### Loss: VitalLoss

```
L_total = continuous_weight × mean(importance_weights × mask × (pred - target)²)
        + categorical_weight × mean([CE(logits_i, target_i) for i])
```

- Default weights: continuous=1.0, categorical=0.5 (CE starts at higher magnitude)
- Importance weights loaded from HDF5 if present (from `datagen compute-weights`), else uniform
- **Conditional loss masking** (`--conditional-loss-mask`, default on): zeros loss for unlearnable params — LFOs, random generators, env 3-6 (static mask), and continuous params gated by disabled module `*_on` switches (dynamic per-sample mask). Focuses training on the ~80 params that actually affect audio at Tier 1.

### Data Pipeline: Precomputed Mels

**Critical finding:** HDF5 gzip compression + shuffled random access is catastrophically slow. Each random read decompresses an entire gzip chunk (~45MB for audio). With 60k shuffled samples per epoch, training couldn't complete a single epoch in 10+ minutes.

**Solution:** Two-step pipeline:
1. `precompute-mels`: Reads audio sequentially (fast with gzip), computes log1p mel spectrograms in batched chunks, writes to `features/mel_spectrogram` as an **uncompressed** HDF5 dataset with `chunks=(1, 1, 128, 173)` for optimal random access.
2. `train`: Reads precomputed 86KB mels directly — no decompression, fast random access.

Memory/disk trade-offs:
| Samples | Uncompressed Mels on Disk |
|---------|--------------------------|
| 60k     | ~5.1 GB                  |
| 300k    | ~25 GB                   |
| 1.5M    | ~128 GB                  |

This scales to any dataset size since data stays on disk. `num_workers=4` with `persistent_workers=True` for parallel reads.

### Train/Val Split

Split by **preset hash**, not sample index. Each preset renders 3 MIDI notes (48, 60, 72), and all 3 must go to the same split to prevent data leakage. Uses `metadata/preset_hash` from HDF5.

### LR Schedule

Linear warmup (5 epochs) → cosine decay to ~0. Configurable via `--warmup-epochs`.

### Checkpoint Resume

Saves: model, optimizer, scheduler state dicts + epoch + best_val_loss + config + param metadata.

Resume: `--resume checkpoints/best_model.pt` restores all state and continues from `epoch + 1`.

## CLI Usage

```bash
# 1. Precompute mels (one-time, after datagen)
python -m training precompute-mels -d data/tier1_20k.h5

# 2. Train
python -m training train \
    -d data/tier1_20k.h5 \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --device mps \          # or cuda
    --wandb-project vital-inverse-synthesis

# 3. Resume training
python -m training train \
    -d data/tier1_20k.h5 \
    --epochs 200 \
    --resume checkpoints/best_model.pt \
    --device cuda

# 4. Evaluate
python -m training evaluate \
    -c checkpoints/best_model.pt \
    -d data/tier1_20k.h5 \
    --n-samples 16
```

## W&B Logging

Per epoch:
- `train/loss`, `train/cont_loss`, `train/cat_loss`
- `val/loss`, `val/cont_loss`, `val/cat_loss`
- `val/mse_<group>` — per-module MSE (osc_1, filter_1, env_1, etc.)
- `val/acc_<param>` — per-param categorical accuracy
- `lr`, `best_val_loss`

Periodic (every `--log-audio-every` epochs):
- `audio/target_N`, `audio/predicted_N` — rendered via Vita, logged as `wandb.Audio`

## Existing Code Reused

| What | From | Purpose |
|------|------|---------|
| `HDF5Reader` | `datagen/storage/reader.py` | Schema attr reading pattern |
| `ParamRegistry.from_synth()` | `datagen/params/registry.py` | Get param metadata for render eval |
| `denormalize_vector()` | `datagen/params/normalize.py` | Convert [0,1] predictions → raw Vita values |
| `RenderEngine` | `datagen/render/engine.py` | Render predicted params to audio for eval |
| `PipelineConfig` | `datagen/config.py` | Configure RenderEngine during eval |

## Verified Behavior

- Shapes: mel output `(1, 128, 173)` matches model input
- Split: no preset hash overlap between train/val
- Forward pass: correct output shapes for continuous and all categorical heads
- Loss: gradients flow, loss is scalar, decreases on overfit
- W&B: metrics, per-group MSE, categorical accuracy, audio all logged
- Render eval: Vita renders target/predicted audio pairs successfully
- Checkpoint resume: epoch, optimizer, scheduler, best_val_loss all restored correctly
- MPS and CUDA device support both tested

## Known Limitations

- `params/continuous` and `params/categorical` are still gzip-compressed (tiny overhead, 248+80 bytes per sample)
- Audio eval requires a live Vita instance — gracefully skipped if unavailable

### Resolved Limitations (see `7-audit-fixes-report.md`)

- ~~Only saves best checkpoint~~ → Now saves `latest_model.pt` every epoch + `best_model.pt`
- ~~No early stopping~~ → Added `--early-stopping-patience` (see `6-changes-report.md`)
- Categorical heads simplified from 2-layer MLP to `nn.Linear` (default) to reduce overfitting
- Conditional loss masking added for ~240 unlearnable params at Tier 1
- LR warmup off-by-one fixed
- Gradient accumulation residual scaling corrected
- Checkpoint now stores `sample_rate` and `continuous_ranges` for inference
