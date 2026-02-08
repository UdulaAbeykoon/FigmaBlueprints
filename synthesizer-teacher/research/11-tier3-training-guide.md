# 11: Tier-3 Full Training Guide (CUDA GPU)

Complete instructions for training the tier-3 inverse synthesis model with wavetable prediction, modulation routing, and multi-GPU support.

## Prerequisites

- CUDA GPU(s) with >= 16 GB VRAM (tested on A100/H100; works on 3090/4090)
- Python environment with: `torch`, `torchaudio`, `torchvision`, `h5py`, `tqdm`, `click`, `wandb`, `auraloss`, `cma`
- Working Vita installation (Python bindings to Vital C++ engine)
- Community presets downloaded to `presets/community/` (recommended, not required)
- Wavetable catalog built or provided as `data/wavetable_catalog.json`

## Step 1: Generate Dataset

### Discover wavetables (if not already done)

```bash
python -m datagen discover-wavetables -o data/wavetable_catalog.json
```

This scans your Vital preset directories, deduplicates by content hash, and writes a JSON catalog. Typical result: ~70 unique wavetables.

### Generate tier-3 data

```bash
python -m datagen generate \
  --tier 3 \
  -n 200000 \
  -o data/tier3_200k.h5 \
  --workers 16 \
  --wavetable-catalog data/wavetable_catalog.json \
  --community-dir presets/community \
  --duration 2.0 \
  --midi-notes 48,60,72 \
  --seed 42
```

**What this does:**
- Generates 200K samples (preset x MIDI note combinations).
- Each preset is rendered at 3 MIDI notes (48, 60, 72) for pitch diversity.
- `--workers 16` parallelizes rendering across 16 processes. Scale to your core count.
- Community presets are ingested and mixed with synthetic presets.
- Synthetic presets use Latin Hypercube Sampling for continuous params, module-level breeding (crossover/interpolation/mutation) when seed presets are available.
- Tier-3 presets get sparse modulation connections sampled from Geometric(p=0.15), averaging ~6.7 connections per preset, clamped to [0, 20].
- Expect 30-40% rejection rate (silent or clipping presets). The pipeline adapts batch sizes accordingly.
- At ~50-200ms per render, 200K samples takes roughly 3-10 hours with 16 workers.

**Scaling guidance:**
- 100K samples: Minimum viable for tier-3. Training will converge but generalize poorly.
- 200K samples: Recommended baseline. Good balance of coverage and generation time.
- 500K samples: Best results. Worth it if you have the compute for both generation and training.

### Verify dataset

```bash
python -c "
import h5py
with h5py.File('data/tier3_200k.h5', 'r') as f:
    print('Schema tier:', f['schema'].attrs.get('tier', 'missing'))
    print('Samples:', f['audio/waveforms'].shape[0])
    print('Continuous params:', f['params/continuous'].shape)
    print('Categorical params:', f['params/categorical'].shape)
    print('Modulation:', f['params/modulation_t3'].shape if 'params/modulation_t3' in f else 'MISSING')
    print('Sample rate:', f['schema'].attrs['sample_rate'])
"
```

Expected output for 200K:
```
Schema tier: 3
Samples: 200000
Continuous params: (200000, 322)
Categorical params: (200000, 126)
Modulation: (200000, 4, 32, 406)
Sample rate: 44100
```

## Step 2: Precompute Features

Both steps are mandatory before training. They convert compressed HDF5 datasets to uncompressed, per-sample-chunked datasets for fast random access in DataLoader workers.

### Precompute mel spectrograms

```bash
python -m training precompute-mels \
  -d data/tier3_200k.h5 \
  --device cuda
```

This computes 128-bin log-mel spectrograms on GPU and writes them to `features/mel_spectrogram` (uncompressed). For 200K samples, expect ~15-30 GB disk space and a few minutes of compute.

### Precompute modulation (tier-3 only)

```bash
python -m training precompute-modulation \
  -d data/tier3_200k.h5
```

Copies `params/modulation_t3` (gzip compressed) to `features/modulation_t3` (uncompressed). This is critical — reading gzip-compressed random-access HDF5 in a DataLoader is catastrophically slow (100-1000x slower).

For 200K samples: ~41 GB uncompressed (4 channels x 32 sources x 406 destinations x 4 bytes x 200K).

## Step 3: Train

### Single GPU

```bash
python -m training train \
  -d data/tier3_200k.h5 \
  --epochs 200 \
  --batch-size 32 \
  --lr 1e-4 \
  --no-freeze \
  --label-smoothing 0.1 \
  --dropout 0.1 \
  --early-stopping-patience 15 \
  --gradient-accumulation-steps 4 \
  --modulation-loss-weight 0.3 \
  --mod-pos-weight 20.0 \
  --wavetable-catalog data/wavetable_catalog.json \
  --compute-spectral-metrics \
  --num-workers 8 \
  --device cuda
```

### Multi-GPU (same node)

```bash
torchrun --nproc_per_node=4 -m training train \
  -d data/tier3_200k.h5 \
  --epochs 200 \
  --batch-size 32 \
  --lr 4e-4 \
  --no-freeze \
  --label-smoothing 0.1 \
  --dropout 0.1 \
  --early-stopping-patience 15 \
  --gradient-accumulation-steps 4 \
  --modulation-loss-weight 0.3 \
  --mod-pos-weight 20.0 \
  --wavetable-catalog data/wavetable_catalog.json \
  --compute-spectral-metrics \
  --num-workers 8
```

### Multi-node (2 nodes x 4 GPUs)

```bash
# Node 0
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
  --master_addr=10.0.0.1 --master_port=29500 \
  -m training train \
  -d data/tier3_200k.h5 \
  --epochs 200 --batch-size 32 --lr 8e-4 \
  --no-freeze --label-smoothing 0.1 --dropout 0.1 \
  --early-stopping-patience 15 --gradient-accumulation-steps 4 \
  --modulation-loss-weight 0.3 --wavetable-catalog data/wavetable_catalog.json \
  --compute-spectral-metrics --num-workers 8

# Node 1
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
  --master_addr=10.0.0.1 --master_port=29500 \
  -m training train \
  -d data/tier3_200k.h5 \
  --epochs 200 --batch-size 32 --lr 8e-4 \
  --no-freeze --label-smoothing 0.1 --dropout 0.1 \
  --early-stopping-patience 15 --gradient-accumulation-steps 4 \
  --modulation-loss-weight 0.3 --wavetable-catalog data/wavetable_catalog.json \
  --compute-spectral-metrics --num-workers 8
```

### Flag rationale

| Flag | Value | Why |
|------|-------|-----|
| `--no-freeze` | always for tier-3 | 322 continuous + 126 categorical + modulation needs richer backbone features than frozen early layers provide |
| `--batch-size 32` | per-GPU | With `--gradient-accumulation-steps 4`, effective batch = 128 per GPU. Stabilizes large output space. |
| `--lr 1e-4` | single GPU | Scale linearly with world_size. 4 GPUs -> `4e-4`, 8 GPUs -> `8e-4`. |
| `--label-smoothing 0.1` | always | Many categorical params have varying class counts (2-70). Smoothing prevents overconfident predictions on rare classes. |
| `--dropout 0.1` | always | Regularization for the 19M param model. |
| `--early-stopping-patience 15` | tier-3 | Tier-3 converges slower than tier-1 due to modulation head. 15 epochs patience prevents premature stopping. |
| `--modulation-loss-weight 0.3` | default | Balances modulation learning against continuous/categorical. See tuning section below if modulation metrics stall. |
| `--mod-pos-weight 20.0` | default | Compensates for extreme sparsity (~0.1% of modulation slots are active). |
| `--compute-spectral-metrics` | recommended | Renders both target and predicted params through Vita and computes MRSTFT loss. The most meaningful metric for audio quality. |
| `--num-workers 8` | scale to cores | DataLoader workers. More = faster data loading, but each opens its own HDF5 handle. 8 is a good default. |

### LR scaling rule

**Effective batch size = `batch_size` x `gradient_accumulation_steps` x `world_size`.**

LR is NOT auto-scaled. When you increase `world_size`, manually multiply `--lr`:

| GPUs | batch_size | accum_steps | effective_batch | --lr |
|------|-----------|-------------|-----------------|------|
| 1 | 32 | 4 | 128 | 1e-4 |
| 2 | 32 | 4 | 256 | 2e-4 |
| 4 | 32 | 4 | 512 | 4e-4 |
| 8 | 32 | 4 | 1024 | 8e-4 |

If training becomes unstable at high LR, try reducing by 0.5x or increasing warmup epochs.

### Learning rate schedule

The scheduler uses linear warmup + cosine annealing decay:
- Epochs 0 to `warmup_epochs` (default 5): LR ramps linearly from 0 to `--lr`.
- Epochs `warmup_epochs` to end: cosine decay back to 0.

### Modulation loss warmup

The modulation loss weight ramps linearly from 0 to `--modulation-loss-weight` over the first `warmup_epochs` (default 5). This prevents the modulation head from dominating gradients before the backbone has learned basic audio features.

### What to watch in W&B

Key metrics logged every epoch:

| Metric | What it means | Healthy range |
|--------|--------------|---------------|
| `train/loss` | Total training loss | Decreasing, stabilizing |
| `val/loss` | Total validation loss | Decreasing, gap with train/loss is small |
| `val/cont_loss` | Continuous param MSE | Should decrease steadily |
| `val/cat_loss` | Categorical cross-entropy | Should decrease, may plateau |
| `val/mod_loss` | Modulation loss | Should decrease after warmup (epoch 5+) |
| `val/mod_precision` | Connection detection precision | Should increase from 0 toward 0.3-0.5 |
| `val/mod_recall` | Connection detection recall | Should increase from 0 toward 0.2-0.4 |
| `val/mod_amount_mae` | Amount prediction error | Should decrease |
| `val/mse_osc_1` | Oscillator 1 param MSE | Should be among lowest group MSEs |
| `val/mse_lfo_1` | LFO 1 param MSE | Should decrease (was masked to 0 before this fix) |
| `val/acc_overall` | Overall categorical accuracy | 0.4-0.7 is good for 126 params |
| `val/spectral/mrstft_loss` | Rendered audio spectral distance | Most meaningful metric. Lower is better. |

Audio samples are logged every `--log-audio-every` epochs (default 5).

## Step 4: Resume Training (if interrupted)

```bash
python -m training train \
  -d data/tier3_200k.h5 \
  --resume checkpoints/latest_model.pt \
  --epochs 300
```

- `latest_model.pt` is saved every epoch. `best_model.pt` is saved when `val/loss` improves.
- Resume restores model weights, optimizer state, scheduler state, and epoch counter.
- You can change `--epochs` to extend training beyond the original target.

## Step 5: Evaluate

```bash
python -m training evaluate \
  -c checkpoints/best_model.pt \
  -d data/tier3_200k.h5 \
  --n-samples 64 \
  --device cuda
```

Prints per-group MSE, categorical accuracy, and spectral metrics (if Vita available).

### Interactive evaluation

```bash
python -m training eval-tui \
  -c checkpoints/best_model.pt \
  -d data/tier3_200k.h5 \
  --device cuda
```

TUI for browsing individual predictions with audio playback (requires `sounddevice`).

## Step 6: Inference

### Basic prediction

```bash
python -m inference infer \
  -c checkpoints/best_model.pt \
  -i target.wav \
  -o output.vital \
  --render
```

### With CMA-ES refinement (recommended for best results)

```bash
python -m inference infer \
  -c checkpoints/best_model.pt \
  -i target.wav \
  -o output.vital \
  --refine \
  --refine-evals 500 \
  --refine-timeout 60 \
  --render \
  --tutorial
```

CMA-ES keeps modulation routing fixed from the neural net prediction and optimizes continuous params only via spectral loss.

### Gradio demo

```bash
python -m inference demo \
  -c checkpoints/best_model.pt \
  --device cuda \
  --share
```

Web UI with audio upload, CMA-ES toggle, parameter display, tutorial generation, and `.vital` preset download.

---

## Troubleshooting

### Training loss is not decreasing

**Symptom:** `train/loss` flat or increasing from epoch 0.

**Check:**
1. Did you run `precompute-mels`? Without it, the DataLoader reads raw audio and will error or produce garbage.
2. Is LR too high? Try `1e-5` as a sanity check.
3. Check `train/cont_loss` and `train/cat_loss` independently. If one is NaN, the issue is in that head.

### Modulation loss is NaN

**Symptom:** `train/mod_loss` or `val/mod_loss` is NaN.

**Likely cause:** The modulation matrix is empty or the BCE with pos_weight diverges.

**Fix:**
1. Verify `params/modulation_t3` actually has non-zero entries: `python -c "import h5py; f=h5py.File('data/tier3_200k.h5','r'); print(f['params/modulation_t3'][:100,0].sum())"`. If zero, your datagen didn't produce modulation data — regenerate with `--tier 3`.
2. Reduce `--mod-pos-weight` from 20 to 10 or 5. High pos_weight can cause instability with very sparse matrices.
3. Increase `--modulation-loss-weight` warmup by editing `modulation_warmup_epochs` in `TrainConfig` (default is 5, try 10).

### Modulation precision/recall stuck at 0

**Symptom:** `val/mod_precision` and `val/mod_recall` remain 0 even after 20+ epochs.

**Possible causes:**
1. **Warmup too short:** The modulation head hasn't had enough gradient signal yet. By default, modulation loss is 0 at epoch 0 and ramps linearly to full weight at epoch 5. If other losses dominate, try increasing `--modulation-loss-weight` to 0.5-1.0.
2. **Sparsity mismatch:** If your dataset has very few active connections (avg < 2 per preset), the model may not get enough positive examples. Check average connections: `python -c "import h5py, numpy as np; f=h5py.File('data/tier3_200k.h5','r'); m=f['params/modulation_t3'][:1000,0]; print('Avg connections:', (np.abs(m)>1e-6).sum(axis=(1,2)).mean())"`. Should be ~6-7 for Geometric(p=0.15) sampling.
3. **Threshold too high:** The default prediction threshold is 0.05. If the model predicts small magnitudes, try lowering to 0.02 in `_extract_modulation` at inference time.

### val/loss stops improving but spectral metrics are bad

**Symptom:** Parameter MSE is low, but rendered audio doesn't match (high `val/spectral/mrstft_loss`).

**Root cause:** Many-to-one problem — multiple parameter settings produce the same sound.

**Mitigations:**
1. **Use CMA-ES at inference time.** Even with imperfect parameter prediction, CMA-ES optimizes continuous params against the actual spectral target. `--refine --refine-evals 500` typically cuts MRSTFT loss by 30-60%.
2. **Train longer.** Spectral metrics often improve well past the point where parameter MSE plateaus, because the model is learning which parameters actually affect the sound (and which are degenerate).
3. **Increase dataset size.** More diverse presets help the model learn the actual parameter-to-sound mapping rather than memorizing specific settings.

### Rendered predictions sound wrong despite good metrics

**Symptom:** Continuous MSE is low, categorical accuracy is high, but audio doesn't match.

**Most likely causes:**
1. **Wavetable catalog not loaded.** Check W&B logs for "Loaded wavetable catalog: N wavetables". If missing, pass `--wavetable-catalog data/wavetable_catalog.json`. Old checkpoints without `wavetable_catalog_json` will use the default Init wavetable for all oscillators.
2. **Modulation not applied in renders.** Check that the checkpoint has `n_mod_sources > 0`. If 0, the checkpoint was trained without the modulation head (perhaps from an old run).
3. **Rendering tier mismatch.** The RenderEngine only applies modulation when `config.tier >= 3`. If the render config defaults to tier 1, modulation is silently ignored even if predicted. The pipeline creates a fresh `PipelineConfig(sample_rate=...)` which defaults to tier 1. This is fine because `render_preset` reads `_modulation_t3` from the preset dict directly and the engine's `_apply_modulation` checks for the dict key, not the config tier.

### OOM (Out of Memory)

**Symptom:** `CUDA out of memory` error.

**Fixes:**
1. Reduce `--batch-size` from 32 to 16 or 8. Compensate with `--gradient-accumulation-steps 8` or `16` to maintain effective batch size.
2. Reduce `--num-workers`. Each worker opens its own HDF5 handle and loads data into shared memory.
3. For 12 GB GPUs: use `--batch-size 16 --gradient-accumulation-steps 8`.
4. For 8 GB GPUs: use `--batch-size 8 --gradient-accumulation-steps 16`. You may also need to reduce `--mlp-hidden` to 256.

Memory budget for batch_size=32:
- Model weights: ~76 MB (19M params x 4 bytes)
- Activations: ~200-400 MB (backbone + modulation bilinear products)
- Mel batch: ~50 MB (32 x 1 x 128 x ~340 frames)
- Modulation batch: ~640 MB (32 x 4 x 32 x 406 x 4 bytes)
- Gradient buffers: ~150 MB
- **Total estimate: ~1.5-2 GB.** Should fit comfortably on any 16 GB GPU.

### DataLoader slow / CPU bottleneck

**Symptom:** GPU utilization low, training throughput limited by data loading.

**Fixes:**
1. **Did you run `precompute-mels` AND `precompute-modulation`?** Without these, every sample read decompresses gzip on the fly.
2. Increase `--num-workers` to match your CPU core count (up to 16-32).
3. If running on NFS/network storage, copy the HDF5 file to local SSD first.
4. Check if pinned memory is enabled: it's on for CUDA, off for MPS (harmless warning).

### DDP hangs or crashes

**Symptom:** Multi-GPU training hangs at startup or during the first all-reduce.

**Fixes:**
1. Verify NCCL is working: `python -c "import torch; print(torch.cuda.nccl.version())"`.
2. Check that all GPUs are visible: `nvidia-smi` should show all expected devices.
3. If using multiple nodes, verify network connectivity between `master_addr:master_port`.
4. Set `NCCL_DEBUG=INFO` for detailed communication logs: `NCCL_DEBUG=INFO torchrun ...`.
5. If one GPU is different (e.g., mixed architectures), NCCL may fail. Use homogeneous GPUs.

### Early stopping triggers too early

**Symptom:** Training stops at epoch 30-40 with `--early-stopping-patience 15`.

**Explanation:** Tier-3 has a complex loss landscape. The modulation warmup (first 5 epochs) means the total loss trajectory isn't smooth — loss may temporarily increase when modulation weight kicks in.

**Fix:** Increase to `--early-stopping-patience 20` or `25`. Alternatively, disable early stopping entirely with `--early-stopping-patience 0` and train for a fixed number of epochs.

### Categorical accuracy is low for specific params

**Symptom:** Overall accuracy is reasonable but certain categorical params (e.g., `filter_1_model`, `distortion_type`) are stuck near random chance.

**Explanation:** Some categorical params have many options (e.g., filter model has 10+ types) but the audio differences between options are subtle.

**Mitigations:**
1. This is expected behavior. The model naturally learns "audible" categoricals (like oscillator wavetable selection) much faster than "subtle" ones.
2. `--label-smoothing 0.1` helps prevent the model from being overconfident on easy categoricals at the expense of hard ones.
3. CMA-ES at inference time can compensate — even if the categorical is wrong, CMA-ES optimizes continuous params to minimize spectral distance.

---

## Recommended Training Configurations

### Quick experiment (development/debugging)

```bash
python -m datagen generate --tier 3 -n 5000 -o data/tier3_5k.h5 --workers 8 --wavetable-catalog data/wavetable_catalog.json
python -m training precompute-mels -d data/tier3_5k.h5 --device cuda
python -m training precompute-modulation -d data/tier3_5k.h5
python -m training train -d data/tier3_5k.h5 --epochs 30 --batch-size 32 \
  --lr 1e-4 --no-freeze --label-smoothing 0.1 --dropout 0.1 \
  --modulation-loss-weight 0.3 --wavetable-catalog data/wavetable_catalog.json \
  --early-stopping-patience 0 --device cuda
```

Good for verifying the pipeline works end-to-end. Not enough data for meaningful generalization.

### Production single-GPU

```bash
python -m datagen generate --tier 3 -n 200000 -o data/tier3_200k.h5 --workers 16 --wavetable-catalog data/wavetable_catalog.json
python -m training precompute-mels -d data/tier3_200k.h5 --device cuda
python -m training precompute-modulation -d data/tier3_200k.h5
python -m training train -d data/tier3_200k.h5 --epochs 200 --batch-size 32 \
  --lr 1e-4 --no-freeze --label-smoothing 0.1 --dropout 0.1 \
  --early-stopping-patience 15 --gradient-accumulation-steps 4 \
  --modulation-loss-weight 0.3 --wavetable-catalog data/wavetable_catalog.json \
  --compute-spectral-metrics --num-workers 8 --device cuda
```

### Production 4-GPU

```bash
python -m datagen generate --tier 3 -n 200000 -o data/tier3_200k.h5 --workers 16 --wavetable-catalog data/wavetable_catalog.json
python -m training precompute-mels -d data/tier3_200k.h5 --device cuda
python -m training precompute-modulation -d data/tier3_200k.h5
torchrun --nproc_per_node=4 -m training train -d data/tier3_200k.h5 \
  --epochs 200 --batch-size 32 --lr 4e-4 --no-freeze \
  --label-smoothing 0.1 --dropout 0.1 --early-stopping-patience 15 \
  --gradient-accumulation-steps 4 --modulation-loss-weight 0.3 \
  --wavetable-catalog data/wavetable_catalog.json \
  --compute-spectral-metrics --num-workers 8
```

### Maximum quality (8-GPU, large dataset)

```bash
python -m datagen generate --tier 3 -n 500000 -o data/tier3_500k.h5 --workers 32 --wavetable-catalog data/wavetable_catalog.json --community-dir presets/community
python -m training precompute-mels -d data/tier3_500k.h5 --device cuda
python -m training precompute-modulation -d data/tier3_500k.h5
torchrun --nproc_per_node=8 -m training train -d data/tier3_500k.h5 \
  --epochs 300 --batch-size 32 --lr 8e-4 --no-freeze \
  --label-smoothing 0.1 --dropout 0.1 --early-stopping-patience 20 \
  --gradient-accumulation-steps 4 --modulation-loss-weight 0.3 \
  --wavetable-catalog data/wavetable_catalog.json \
  --compute-spectral-metrics --num-workers 8
```

---

## Hyperparameter Tuning Priority

If you need to tune, adjust these in order of impact:

1. **Dataset size** — More data > any hyperparameter change. 200K -> 500K is the single biggest improvement.
2. **Learning rate** — Too high = instability, too low = slow convergence. Start with the linear scaling rule and adjust +/- 0.5x.
3. **Modulation loss weight** — If `val/mod_precision` is stuck at 0 after 30 epochs, try 0.5 or 1.0. If it's unstable, try 0.1.
4. **Modulation pos_weight** — If modulation loss is NaN, reduce from 20 to 5-10.
5. **Dropout** — If val loss diverges from train loss (overfitting), increase to 0.2. If underfitting, reduce to 0.05.
6. **Label smoothing** — 0.1 is robust. Try 0.05 if categorical accuracy is too low, 0.2 if certain classes dominate.
7. **Gradient accumulation** — Keep effective batch >= 128. Below that, training may be noisy.
