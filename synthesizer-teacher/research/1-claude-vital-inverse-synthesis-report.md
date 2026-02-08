# Inverse Synthesis for Vital: ML System Design Report

## 1. Problem Statement

Given an audio clip, infer the synthesizer parameters that produced it in the
Vital wavetable synthesizer, then generate a human-readable tutorial explaining
how to recreate the sound. This is a regression/classification problem over a
structured, high-dimensional parameter space, with a secondary natural language
generation step.

No published work targets Vital specifically. Existing inverse synthesis
literature focuses on simpler synths (Dexed/DX7, Diva, analog-modeled
subtractive). Vital's wavetable engine, deep modulation matrix, and open JSON
preset format create both a harder problem and a better tooling story than
anything in the literature.

---

## 2. Why Vital, and Why Now

Three things make this project feasible in 2025–2026 when it wasn't before:

**Vita** (github.com/DBraun/Vita) provides Python bindings to Vital's C++ audio
engine. You can render audio from JSON preset definitions programmatically,
without hosting a VST plugin. This eliminates the DawDreamer/VST hosting
complexity that has historically made synth dataset generation fragile. Vita
exposes the full parameter space including the modulation matrix, which
DawDreamer cannot reliably access for Vital.

**Vital's `.vital` format is plain JSON.** Every parameter is a named key-value
pair. You can template a preset, slot in predicted values, and export a valid
`.vital` file that opens directly in the synth. This is a massive advantage for
both dataset generation and output formatting.

**Pretrained audio encoders** (AST, MERT, AudioMAE) have matured to the point
where you don't need to train a spectrogram feature extractor from scratch. A
frozen encoder + learned MLP head is a viable architecture that trains in hours
on a single GPU.

---

## 3. Vital's Parameter Space

Understanding what you're predicting is essential before designing anything.

### 3.1 Full Parameter Inventory

Vital exposes roughly 220–250 named parameters in its JSON preset format. These
break down into:

| Group             | Parameters                                                                                                               | Count    | Type                     |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------ | -------- | ------------------------ |
| Oscillators 1–3   | Level, pan, transpose, tune, unison voices, unison detune, wavetable position, wave frame, phase, distortion type/amount | ~45      | Continuous + categorical |
| Sample oscillator | Level, pan, transpose, tune                                                                                              | ~8       | Continuous               |
| Filters 1–2       | Cutoff, resonance, drive, mix, blend, model (12 types), key-tracking                                                     | ~20      | Continuous + categorical |
| Envelopes 1–3     | Attack, decay, sustain, release, power (curve shape)                                                                     | ~15      | Continuous               |
| LFOs 1–4          | Frequency, phase, fade-in, delay, stereo, smooth                                                                         | ~16      | Continuous               |
| Effects (9 slots) | Each has type + 3–8 params (chorus, compressor, delay, distortion, EQ, filter FX, flanger, phaser, reverb)               | ~80      | Continuous + categorical |
| Modulation matrix | Source, destination, amount, power, stereo, bipolar per slot (up to 64 slots)                                            | Variable | Structured               |
| Macros (4)        | Value                                                                                                                    | 4        | Continuous               |
| Global            | Voice count, pitch bend range, velocity tracking, stereo mode                                                            | ~20      | Mixed                    |

Additionally, each oscillator carries embedded **wavetable data** — the actual
waveform shapes stored as binary data within the JSON. LFO and envelope shapes
can also be custom-drawn.

### 3.2 The Wavetable Problem

This is the single most important constraint the original research report
missed.

Vital's oscillators use wavetable synthesis. The `osc_1_wave_frame` parameter
(wavetable position) sweeps through frames of a specific wavetable file. But the
_identity_ of that wavetable determines everything about the harmonic content.
Predicting `wave_frame: 0.5` is meaningless without knowing which wavetable is
loaded.

Vital ships with approximately 70 factory wavetables organized into categories
(analog, digital, fourier, etc.). Users can also import custom wavetables.

**Solution:** Constrain the system to factory wavetables only. Treat wavetable
selection as a categorical parameter (one of ~70 classes per oscillator). This
is a hard constraint — if the target sound was made with a custom wavetable, the
model cannot reconstruct it. This is an acceptable limitation for a hackathon
and should be stated explicitly.

### 3.3 The Modulation Problem

Vital's modulation matrix supports up to 64 routing slots, each specifying:
source (any LFO, envelope, macro, mod wheel, velocity, aftertouch, etc.),
destination (any parameter), amount, power curve, stereo offset, and bipolar
toggle.

This is a **variable-length structured prediction problem**, fundamentally
different from regressing scalar parameters. The same final sound can be
achieved with different modulation topologies. An LFO routed to filter cutoff
with amount 0.7 produces a completely different sound from a static filter
cutoff set to the time-averaged equivalent value.

This problem is addressed in detail in Section 8.

---

## 4. Dataset: Where to Find It and How to Make It

There is no existing dataset for Vital inverse synthesis. You must generate your
own. This is actually an advantage — you control the parameter distribution.

### 4.1 Synthetic Dataset Generation with Vita

This is your primary dataset source.

```bash
pip install vita
```

```python
import vita
import numpy as np
import json
import soundfile as sf
from pathlib import Path

# Initialize
synth = vita.Synth(sample_rate=44100)
controls = synth.get_controls()  # dict of all parameter names + ranges

# Get factory wavetable list
factory_wavetables = synth.get_wavetable_names()  # ~70 entries

def generate_random_preset(controls, wavetable_names):
    """Generate a random but non-degenerate preset."""
    preset = {}

    for name, info in controls.items():
        if info['type'] == 'continuous':
            preset[name] = np.random.uniform(info['min'], info['max'])
        elif info['type'] == 'categorical':
            preset[name] = np.random.choice(info['options'])

    # Force at least one oscillator to be audible
    preset['osc_1_level'] = np.random.uniform(0.3, 1.0)

    # Assign factory wavetables only
    for osc in ['osc_1', 'osc_2', 'osc_3']:
        preset[f'{osc}_wavetable'] = np.random.choice(wavetable_names)

    return preset

def is_valid_render(audio, sr=44100):
    """Reject silent or clipping renders."""
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    return rms > 0.01 and peak < 0.99  # not silent, not clipping

# Generation loop
output_dir = Path("dataset")
output_dir.mkdir(exist_ok=True)

generated = 0
target = 20_000

while generated < target:
    preset = generate_random_preset(controls, factory_wavetables)
    synth.load_preset(preset)

    for midi_note in [48, 60, 72]:  # C3, C4, C5
        audio = synth.render(
            pitch=midi_note, velocity=100,
            note_duration=1.5, render_duration=2.0
        )

        if not is_valid_render(audio):
            continue

        sf.write(
            output_dir / f"{generated:06d}_n{midi_note}.wav",
            audio.T, 44100
        )

    # Save parameters as JSON
    with open(output_dir / f"{generated:06d}_params.json", 'w') as f:
        json.dump(preset, f)

    generated += 1
    if generated % 1000 == 0:
        print(f"Generated {generated}/{target}")
```

**Key decisions in the generation script:**

**Multi-pitch rendering.** Synth sounds change dramatically with pitch due to
key-tracking, velocity-sensitive envelopes, and filter behavior. Rendering at
MIDI 48, 60, and 72 (C3, C4, C5) gives the model exposure to pitch-dependent
variation. At inference time, you'll need to know (or detect) the approximate
pitch of the input audio.

**Rejection sampling.** Pure random parameter sampling produces many degenerate
presets: filter fully closed produces silence, max distortion + max resonance
produces noise, zero oscillator levels produce nothing. The `is_valid_render`
check discards these. Expect ~30–40% rejection rate, so oversample accordingly.

**Latin Hypercube Sampling.** For better parameter space coverage than pure
random, use `scipy.stats.qmc.LatinHypercube` over the continuous parameters.
This ensures you don't accidentally cluster samples in one region of the space.

```python
from scipy.stats.qmc import LatinHypercube

n_continuous = len(continuous_param_names)
sampler = LatinHypercube(d=n_continuous, seed=42)
samples = sampler.random(n=50_000)  # oversample for rejection

# Scale each dimension to its parameter range
for i, name in enumerate(continuous_param_names):
    lo, hi = controls[name]['min'], controls[name]['max']
    samples[:, i] = lo + samples[:, i] * (hi - lo)
```

**Render time estimates.** Simple patches render at ~50ms each. Complex patches
with all 9 effects active and heavy modulation can take 200–500ms. Budget 2–3x
what a naive estimate suggests. For 20K samples × 3 pitches = 60K renders,
expect 2–6 hours on 8 CPU cores with multiprocessing.

### 4.2 Community Preset Augmentation

There are 4,000+ free community-made Vital presets available from:

- **Vital Audio Forum** (forum.vital.audio) — the official community
- **PresetShare** — user-uploaded preset packs
- **GitHub repositories** — search "vital presets"

These are useful for two purposes: (a) as a held-out test set representing
"sounds people actually make" (vs. random parameter sampling), and (b) as
examples of realistic modulation routing topologies for the modulation modeling
stage (Section 8).

To use community presets as training data:

```python
import json

def load_vital_preset(path):
    """Load a .vital preset file and extract parameters + render audio."""
    with open(path) as f:
        preset_data = json.load(f)

    settings = preset_data['settings']

    # Check for custom wavetables
    for osc in ['osc_1', 'osc_2', 'osc_3']:
        wavetable_key = f'{osc}_wavetable'
        if wavetable_key in preset_data:
            wt_data = preset_data[wavetable_key]
            if is_custom_wavetable(wt_data):
                return None  # skip — we can't predict custom wavetables

    synth.load_json(json.dumps(preset_data))
    audio = synth.render(pitch=60, velocity=100,
                        note_duration=1.5, render_duration=2.0)

    return settings, audio
```

**Important caveat:** Community presets will heavily use modulation and custom
wavetables. Many will be unusable for the static-parameter-only MVP. Filter for
presets that use only factory wavetables and have relatively simple modulation.

### 4.3 Dataset Size Recommendations

| Stage               | Samples        | Purpose                                                              |
| ------------------- | -------------- | -------------------------------------------------------------------- |
| Pipeline validation | 500–1,000      | Verify end-to-end: generate → train → predict → render → compare     |
| Hackathon MVP       | 10,000–20,000  | Sufficient for ResNet/MLP baseline on 40 parameters                  |
| Research quality    | 50,000–100,000 | Needed for transformer encoders and full parameter space             |
| With augmentation   | 100,000+       | Pitch shifting, time stretching, noise injection on existing renders |

---

## 5. Model Architecture: Simple to Complex

The single most important principle: **get the simplest version working
end-to-end before adding complexity.** Each tier below should be fully validated
before moving to the next.

### 5.1 Tier 1 — ResNet Baseline (Build This First)

**Architecture:** Pretrained ResNet-18 (ImageNet) on 128-bin log-mel
spectrograms, treated as single-channel images. Replace the classification head
with an MLP regression/classification head.

```
Audio (2s, 44.1kHz)
  → torchaudio.MelSpectrogram(n_mels=128, n_fft=2048, hop=512)
  → log1p() normalization
  → [1, 128, 173] tensor (like a grayscale image)
  → ResNet-18 (pretrained, frozen early layers, finetune layer3+layer4)
  → 512-dim feature vector
  → MLP(512 → 256 → N_continuous + N_categorical)
  → sigmoid (continuous) / softmax (categorical)
```

**Why this first:** It trains in 20–30 minutes on a single GPU, uses a single
`torchvision` import, and gives you a working baseline to measure everything
else against. If this doesn't work at all (e.g., the mel spectrogram doesn't
capture relevant information, or the parameter space is too large), you'll know
in an hour rather than a day.

**Training details:**

```python
import torch
import torch.nn as nn
import torchaudio
from torchvision.models import resnet18

class VitalInverseModel(nn.Module):
    def __init__(self, n_continuous=35, n_filter_types=12, n_wavetables=70):
        super().__init__()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100, n_mels=128, n_fft=2048, hop_length=512
        )

        # Pretrained ResNet, modify input for 1 channel
        backbone = resnet18(pretrained=True)
        backbone.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Identity()
        self.encoder = backbone  # outputs 512-dim

        # Regression head (continuous params)
        self.reg_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_continuous), nn.Sigmoid()
        )

        # Classification heads (categorical params)
        self.filter_type_head = nn.Linear(512, n_filter_types)
        self.wavetable_heads = nn.ModuleList([
            nn.Linear(512, n_wavetables) for _ in range(3)  # 3 oscillators
        ])

    def forward(self, audio):
        # audio: [B, samples]
        mel = self.mel(audio)          # [B, 128, T]
        mel = torch.log1p(mel)
        mel = mel.unsqueeze(1)         # [B, 1, 128, T]

        features = self.encoder(mel)   # [B, 512]

        continuous = self.reg_head(features)
        filter_type = self.filter_type_head(features)
        wavetables = [head(features) for head in self.wavetable_heads]

        return continuous, filter_type, wavetables
```

**Loss function:**

```python
def compute_loss(pred_continuous, pred_filter, pred_wts,
                 true_continuous, true_filter, true_wts, weights):
    """
    weights: per-parameter importance weights from perturbation analysis
    """
    # Weighted parameter MSE
    param_loss = (weights * (pred_continuous - true_continuous)**2).mean()

    # Cross-entropy for categoricals
    filter_loss = F.cross_entropy(pred_filter, true_filter)
    wt_loss = sum(F.cross_entropy(pw, tw) for pw, tw in zip(pred_wts, true_wts))

    return param_loss + 0.5 * filter_loss + 0.3 * wt_loss
```

**Expected performance:** Don't expect great perceptual results from this tier.
The purpose is to validate the pipeline and establish a baseline. Parameter MSE
will be meaningful; perceptual quality will be rough.

**GPU requirements:** 2–3 GB VRAM. Trains in 20–60 min on RTX 3060 or better.

### 5.2 Tier 2 — Pretrained Audio Encoder (Upgrade When Baseline Works)

Replace ResNet-18 with a pretrained audio transformer. This is where the encoder
choice matters.

**Option A: AST (Audio Spectrogram Transformer)**

- Pretrained on AudioSet (2M clips, 527 classes)
- Architecture: ViT applied to spectrogram patches
- Validated for synth parameter inference in DAFx 2024 literature
- 87M parameters, 768-dim output
- `MIT/ast-finetuned-audioset-10-10-0.4593`

**Option B: MERT-v1-95M**

- Pretrained on music with CQT-teacher + masked prediction
- Designed for music information retrieval tasks
- 95M parameters, 768-dim output per layer (13 layers)
- Better at pitch/key/rhythm; possibly less sensitive to timbral microstructure
- `m-a-p/MERT-v1-95M`

**Option C: AudioMAE**

- Pretrained on AudioSet via masked autoencoding of spectrograms
- Reconstruction-based pretraining may capture spectral detail well
- 86M parameters, 768-dim output
- `facebook/audiomae-base` (check availability — may need to load from repo)

**Recommendation:** Prototype AST and MERT. AST has the strongest evidence from
the synth inference literature. MERT has the strongest music-specific inductive
bias. Run both with identical MLP heads on your training data and compare
validation loss + spectral distance of rendered predictions. This comparison is
itself a modest contribution if you write it up.

**Architecture with pretrained encoder:**

```python
from transformers import AutoModel, AutoFeatureExtractor

class VitalInverseTransformer(nn.Module):
    def __init__(self, encoder_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                 n_continuous=35):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_name)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Learned weighted average over transformer layers
        self.layer_weights = nn.Parameter(torch.ones(13) / 13)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, n_continuous), nn.Sigmoid()
        )

    def forward(self, audio):
        # Extract features using model's expected input format
        with torch.no_grad():
            outputs = self.encoder(audio, output_hidden_states=True)

        # Learned weighted layer pool
        hidden_states = torch.stack(outputs.hidden_states)  # [13, B, T, 768]
        weights = F.softmax(self.layer_weights, dim=0)
        pooled = (weights[:, None, None, None] * hidden_states).sum(0)  # [B, T, 768]
        pooled = pooled.mean(dim=1)  # [B, 768] — average over time

        return self.head(pooled)
```

**Key detail: learned layer weighting.** Different transformer layers capture
different information — early layers have more local spectral detail, later
layers have more abstract features. Learning a weighted combination of all
layers typically outperforms using only the last layer for regression tasks.

**GPU requirements:** 4–6 GB VRAM with frozen encoder. Training the MLP head
takes 1–3 hours on 20K samples.

### 5.3 Tier 3 — Perceptual Loss via Inference-Time Optimization (Advanced)

Tiers 1 and 2 train with parameter-space loss only — they never "hear" how the
predicted parameters sound. The predicted and target audio are only compared at
evaluation time.

Adding a perceptual/spectral loss during training would require a
**differentiable synthesizer** or a **neural proxy** of Vital's rendering
engine. Training such a proxy is a substantial subproject (see Section 9.2). For
a hackathon, skip it.

Instead, use **inference-time optimization**: take the model's predicted
parameters as a starting point, then run a black-box optimizer through Vita's
non-differentiable renderer to minimize spectral distance.

**CMA-ES (Covariance Matrix Adaptation Evolution Strategy)** is the right choice
here:

- Derivative-free (works through Vita directly, no proxy needed)
- Effective in 10–100 dimensional continuous spaces
- Handles the correlation structure between parameters naturally
- Well-tested Python implementation (`pip install cma`)

```python
import cma
import auraloss

mrstft = auraloss.freq.MultiResolutionSTFTLoss()

def refine_prediction(initial_params, target_audio, synth, n_iters=100):
    """
    Take model's predicted parameters and optimize through Vita
    to minimize spectral distance to target audio.
    """
    def objective(params_array):
        # Convert array back to parameter dict
        preset = array_to_preset(params_array)
        synth.load_preset(preset)
        predicted_audio = synth.render(pitch=60, velocity=100,
                                       note_duration=1.5, render_duration=2.0)

        # Multi-resolution STFT distance
        pred_tensor = torch.from_numpy(predicted_audio).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_audio).float().unsqueeze(0)

        return mrstft(pred_tensor, target_tensor).item()

    x0 = preset_to_array(initial_params)
    sigma0 = 0.1  # initial step size — 10% of normalized range

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'maxiter': n_iters,
        'bounds': [0, 1],  # all params normalized to [0,1]
        'popsize': 16
    })

    es.optimize(objective)
    return array_to_preset(es.result.xbest)
```

**Performance note:** Each CMA-ES iteration requires `popsize` Vita renders.
With popsize=16 and 100 iterations, that's 1,600 renders (~80 seconds for simple
patches, ~5 minutes for complex ones). This is viable for a demo but too slow
for batch evaluation. For the demo, show the spectral distance decreasing in
real-time — it's a compelling visual.

### 5.4 Tier 4 — Differentiable Training with Neural Proxy (Research Extension)

If you go beyond a hackathon into a research project, you can train a neural
network to approximate Vital's rendering, making the synthesis process
differentiable. This enables spectral/perceptual loss during training rather
than only at inference time.

**Architecture:** A WaveNet or Transformer-based model that takes parameter
vector → predicted audio spectrogram. Train on 50–100K parameter→audio pairs
from Vita.

**The training pipeline becomes:**

```
Input Audio → Encoder → Predicted Params → Neural Proxy → Predicted Spectrogram
                                                    ↕ (spectral loss)
                              Target Audio → Target Spectrogram
```

This is a meaningful subproject — budget 1–2 weeks for the proxy alone. The
proxy must be accurate enough that optimizing against it actually improves the
real Vita renders, which requires careful validation.

**Alternative: REINFORCE trick.** Instead of a proxy, use policy gradient
methods. Render through Vita (non-differentiable), compute spectral distance as
reward, backpropagate through the encoder/MLP only using REINFORCE. This is what
SynthRL (IJCAI 2025) does. It's simpler than a proxy but has high variance and
requires careful reward shaping.

---

## 6. Training Process

### 6.1 Data Preprocessing

```python
# Mel spectrogram configuration
MEL_CONFIG = {
    'sample_rate': 44100,
    'n_mels': 128,
    'n_fft': 2048,
    'hop_length': 512,
    'f_min': 20,
    'f_max': 16000
}

# For Tier 1 (ResNet): precompute and cache mel spectrograms
# For Tier 2 (AST/MERT): use each model's own feature extractor

# Parameter normalization: all continuous params to [0, 1]
# using min/max from Vita's control definitions
def normalize_params(params, control_defs):
    normalized = {}
    for name, value in params.items():
        lo = control_defs[name]['min']
        hi = control_defs[name]['max']
        normalized[name] = (value - lo) / (hi - lo + 1e-8)
    return normalized
```

### 6.2 Perceptual Importance Weights

Before training, compute per-parameter perceptual importance. This is the single
highest-ROI preprocessing step for improving output quality.

```python
def compute_importance_weights(synth, base_preset, n_presets=500, epsilon=0.1):
    """
    For each parameter, measure how much a ±10% perturbation
    changes the output audio (measured by MRSTFT distance).
    Average over many base presets.
    """
    mrstft = auraloss.freq.MultiResolutionSTFTLoss()
    importance = {name: 0.0 for name in param_names}

    for _ in range(n_presets):
        preset = generate_random_preset(...)
        synth.load_preset(preset)
        base_audio = synth.render(pitch=60, velocity=100, ...)
        base_tensor = torch.from_numpy(base_audio).float().unsqueeze(0)

        for name in param_names:
            # Perturb this parameter up
            perturbed = preset.copy()
            perturbed[name] = min(1.0, preset[name] + epsilon)
            synth.load_preset(perturbed)
            pert_audio = synth.render(pitch=60, velocity=100, ...)
            pert_tensor = torch.from_numpy(pert_audio).float().unsqueeze(0)

            dist = mrstft(pert_tensor, base_tensor).item()
            importance[name] += dist / n_presets

    # Normalize so max weight = 1.0
    max_imp = max(importance.values())
    return {k: v / max_imp for k, v in importance.items()}
```

Expected results: filter cutoff, oscillator levels, and resonance will have
importance weights near 1.0. Effect parameters like `chorus_feedback` or
`phaser_offset` will be 0.01–0.1. Use these weights directly in the loss
function.

**Compute budget:** 500 presets × ~40 parameters × 2 renders each = ~40,000
renders. At 50ms each, that's ~30 minutes.

### 6.3 Training Recipe

```python
# Hyperparameters
BATCH_SIZE = 32
LR = 3e-4
EPOCHS = 100  # ResNet converges fast; transformer head may need 200
WEIGHT_DECAY = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

# Training loop (simplified)
for epoch in range(EPOCHS):
    for audio_batch, param_batch in dataloader:
        pred = model(audio_batch)
        loss = importance_weighted_mse(pred, param_batch, weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scheduler.step()

    # Validation: render predictions through Vita, compute spectral distance
    if epoch % 10 == 0:
        evaluate_rendered_quality(model, val_set, synth)
```

### 6.4 Evaluation Metrics

| Metric                         | What it measures                      | Library                    |
| ------------------------------ | ------------------------------------- | -------------------------- |
| Parameter MSE                  | Raw prediction accuracy               | PyTorch                    |
| Multi-resolution STFT distance | Spectral similarity of rendered audio | `auraloss`                 |
| MFCC distance                  | Timbral similarity                    | `librosa`                  |
| Log-spectral distance (LSD)    | Frequency-domain error                | Manual (simple)            |
| CLAP cosine similarity         | Perceptual/semantic similarity        | `laion/clap-htsat-unfused` |

**The metric that matters most is MRSTFT distance on rendered audio.** Parameter
MSE is a proxy — two different parameter sets can sound identical (many-to-one),
and small parameter errors in perceptually important parameters can sound
terrible while large errors in unimportant parameters don't matter.

Always evaluate by rendering predictions through Vita and listening. Automated
metrics are necessary for iteration speed, but they don't substitute for ears.

---

## 7. The LLM Tutorial Generation Step

### 7.1 Parameter-to-Language Mapping

The non-trivial part is converting normalized parameter values to human-readable
Vital interface values. Vital's parameters use various nonlinear scaling curves
internally.

```python
# Vital's parameter scaling (extracted from source code / Vita API)
PARAM_SCALING = {
    'filter_1_cutoff': {
        'type': 'logarithmic',
        'display': lambda v: f"{20 * (20000/20)**v:.0f} Hz",
        'range': '20 Hz – 20 kHz'
    },
    'osc_1_level': {
        'type': 'linear_db',
        'display': lambda v: f"{20 * np.log10(v + 1e-6):.1f} dB",
        'range': '-inf – 0 dB'
    },
    'env_1_attack': {
        'type': 'exponential',
        'display': lambda v: f"{v * 4.0:.3f}s" if v * 4 < 1 else f"{v * 4.0:.2f}s",
        'range': '0 – 4 seconds'
    },
    'filter_1_resonance': {
        'type': 'linear',
        'display': lambda v: f"{v * 100:.0f}%",
        'range': '0 – 100%'
    }
}

def format_params_for_tutorial(predicted_params):
    """Convert model output to human-readable parameter descriptions."""
    readable = {}
    for name, value in predicted_params.items():
        if name in PARAM_SCALING:
            info = PARAM_SCALING[name]
            readable[name] = {
                'value': value,
                'display': info['display'](value),
                'range': info['range']
            }
        else:
            readable[name] = {'value': value, 'display': f"{value:.2f}"}
    return readable
```

**Vita's `get_control_text()` method** can help here — it returns the display
string for a given parameter value as Vital's own UI would show it. Use this
where available to avoid reimplementing scaling curves.

### 7.2 LLM Prompt Design

```python
import anthropic

client = anthropic.Anthropic()

def generate_tutorial(readable_params, audio_description=None):
    prompt = f"""You are an expert Vital synthesizer sound designer writing a tutorial.

Given these predicted parameters for a Vital preset, write a step-by-step
tutorial explaining how to recreate this sound from the Init preset.

Parameters:
{json.dumps(readable_params, indent=2)}

Instructions:
1. Start with a 1-2 sentence description of what this sound is
   (bass, lead, pad, pluck, etc.) and its tonal character.
2. Walk through each parameter group in the order a sound designer
   would work: oscillators first, then filters, then envelopes,
   then effects.
3. For each parameter, state the exact value to set in Vital's UI
   and briefly explain what it contributes to the sound.
4. End with 2-3 creative suggestions for variations
   (e.g., "try increasing resonance for a more aggressive tone").
5. Use Vital-specific UI terminology (e.g., "the cutoff knob in
   the Filter 1 section", not "filter_1_cutoff").
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

### 7.3 LLM as Post-Processor (Stretch Goal)

Beyond tutorial generation, the LLM can sanity-check predicted parameters for
contradictions. For example: filter fully closed (`cutoff: 0.0`) with high
resonance (`resonance: 0.9`) and no envelope modulation will produce silence or
a barely audible self-oscillation. An LLM with sound design knowledge can flag
and correct these cases.

This is speculative — it depends on the LLM having reliable enough synthesis
knowledge to not make things worse. Worth trying but not worth depending on.

---

## 8. Handling Modulation: A Practical Approach

This is the hardest subproblem and the one most papers punt on. Here's a staged
approach from tractable to ambitious.

### 8.1 Stage 0 — Ignore Modulation (MVP)

Predict only static parameter values. No modulation routing, no LFO-driven
movement, no envelope-to-filter sweeps. The predicted preset will sound "frozen"
compared to the target — a static snapshot rather than a living sound.

This is the right starting point. A model that accurately predicts static
parameters is a genuine contribution and a necessary foundation for everything
else.

### 8.2 Stage 1 — Fixed Topology, Predict Amounts Only

Define a small set of standard modulation routings that cover the most common
sound design patterns. Hardcode these routings and only predict the modulation
amount (and optionally power/curve).

**Recommended fixed routings (8 slots):**

| Slot | Source     | Destination                     | Why                                     |
| ---- | ---------- | ------------------------------- | --------------------------------------- |
| 1    | Envelope 2 | Filter 1 Cutoff                 | The single most common synth modulation |
| 2    | LFO 1      | Oscillator 1 Wavetable Position | Wavetable movement / animation          |
| 3    | LFO 2      | Filter 1 Cutoff                 | Filter wobble (dubstep, techno)         |
| 4    | Envelope 2 | Oscillator 1 Level              | Amplitude shaping beyond main env       |
| 5    | LFO 1      | Pitch (all oscillators)         | Vibrato                                 |
| 6    | Velocity   | Filter 1 Cutoff                 | Velocity-sensitive brightness           |
| 7    | LFO 2      | Pan                             | Stereo movement                         |
| 8    | Envelope 3 | Oscillator 1 Wavetable Position | Timbral envelope                        |

For each slot, predict: amount (continuous, -1 to 1), and an on/off toggle
(binary — is this routing active?).

```python
# Add to model output
class ModulationHead(nn.Module):
    def __init__(self, input_dim, n_fixed_routes=8):
        super().__init__()
        self.amount_head = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, n_fixed_routes), nn.Tanh()  # [-1, 1]
        )
        self.active_head = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, n_fixed_routes), nn.Sigmoid()  # [0, 1] probability
        )

    def forward(self, features):
        amounts = self.amount_head(features)
        active_probs = self.active_head(features)
        return amounts, active_probs
```

**Training data for this stage:** Generate presets using only these 8 fixed
routings (with random amounts). This constrains the dataset but ensures the
model sees the sonic effect of each routing. Generate 20K+ samples for this
stage specifically.

**This approach is workable in a hackathon.** It captures the most perceptually
important modulation patterns and converts the variable-length structured
prediction problem into 16 scalar predictions (8 amounts + 8 on/off).

### 8.3 Stage 2 — Modulation Template Retrieval

Instead of predicting modulation from scratch, retrieve the closest matching
modulation topology from a library of templates.

**Approach:**

1. Collect 500–1,000 community presets with interesting modulation.
2. Extract their modulation matrices and cluster them into ~50 topology
   archetypes (using the routing structure, ignoring amounts).
3. Train a classifier to predict which archetype best matches the input audio.
4. Apply the retrieved topology template, then use Stage 1's approach to predict
   amounts.

```python
# Modulation topology as a structural fingerprint
def extract_topology(preset):
    """Convert modulation matrix to a hashable topology string."""
    routes = preset.get('modulations', [])
    topology = []
    for r in routes:
        topology.append((r['source'], r['destination']))
    return frozenset(topology)

# Cluster topologies
from collections import Counter
topology_counts = Counter()
for preset in community_presets:
    topo = extract_topology(preset)
    topology_counts[topo] += 1

# Top 50 most common topologies become your template library
templates = [topo for topo, count in topology_counts.most_common(50)]
```

This is a reasonable research contribution: it frames modulation prediction as
retrieval + adaptation rather than generation, which is much more tractable.

### 8.4 Stage 3 — Full Modulation Prediction (Research Project)

Predicting arbitrary modulation routing from audio is genuinely hard and
probably requires:

- A set prediction architecture (like DETR for object detection) that outputs a
  variable-length set of (source, destination, amount) tuples
- Gumbel-Softmax for differentiable categorical sampling of sources and
  destinations
- A learned modulation vocabulary embedding that represents source/destination
  pairs in a continuous space
- Contrastive or reconstruction-based training that learns the _audible
  signature_ of specific modulation routings

This is a multi-month research problem. The key difficulty: many modulation
routings are perceptually indistinguishable at certain parameter settings (e.g.,
slow LFO→cutoff vs. slow LFO→resonance at low amounts). The mapping from audio
to modulation topology is severely many-to-many.

**If pursuing this as thesis work:** Start with Stage 1/2 as baselines, then
investigate whether a transformer decoder with learned query tokens (similar to
DETR) can predict modulation slots as a set. Each query token attends to the
audio encoding and outputs (source_logits, destination_logits, amount,
is_active). This is architecturally sound but training it effectively is the
research challenge.

---

## 9. Resources

### 9.1 Essential Tools

| Tool             | Purpose                                        | Install                    |
| ---------------- | ---------------------------------------------- | -------------------------- |
| **Vita**         | Python bindings to Vital's C++ engine          | `pip install vita`         |
| **torchaudio**   | Mel spectrograms, audio I/O                    | `pip install torchaudio`   |
| **auraloss**     | Multi-resolution STFT loss                     | `pip install auraloss`     |
| **transformers** | AST, MERT, AudioMAE encoders                   | `pip install transformers` |
| **cma**          | CMA-ES optimizer for inference-time refinement | `pip install cma`          |
| **wandb**        | Experiment tracking                            | `pip install wandb`        |
| **gradio**       | Demo interface                                 | `pip install gradio`       |
| **anthropic**    | Claude API for tutorial generation             | `pip install anthropic`    |
| **librosa**      | MFCC computation, audio analysis               | `pip install librosa`      |
| **soundfile**    | Audio I/O                                      | `pip install soundfile`    |

### 9.2 Pretrained Models

| Model       | HuggingFace ID                            | Params | Notes                                  |
| ----------- | ----------------------------------------- | ------ | -------------------------------------- |
| AST         | `MIT/ast-finetuned-audioset-10-10-0.4593` | 87M    | Validated for synth tasks in DAFx 2024 |
| MERT-v1-95M | `m-a-p/MERT-v1-95M`                       | 95M    | Music-specific, 24kHz, CQT-teacher     |
| AudioMAE    | `facebookresearch/AudioMAE` (GitHub)      | 86M    | Masked autoencoding, spectral detail   |
| CLAP        | `laion/clap-htsat-unfused`                | ~150M  | For perceptual similarity evaluation   |

### 9.3 Existing Codebases to Study

| Repository                                  | What it does                                                                                                |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **github.com/DBraun/Vita**                  | Python↔Vital bridge. Read the source to understand parameter naming.                                        |
| **github.com/gudgud96/syntheon**            | Inverse synthesis for DawDreamer-hosted synths. Good reference architecture, potential baseline to beat.    |
| **github.com/DBraun/DawDreamer**            | VST hosting in Python. Fallback if Vita doesn't cover your needs.                                           |
| **github.com/torchsynth/torchsynth**        | Differentiable synth in PyTorch (16,200× realtime). Useful reference for differentiable synthesis concepts. |
| **github.com/acids-ircam/flow_synthesizer** | Flow-based model for synth parameter inference. Older but clean implementation.                             |

### 9.4 Key Papers

| Paper                      | Year | Relevance                                                               |
| -------------------------- | ---- | ----------------------------------------------------------------------- |
| InverSynth (Barkan et al.) | 2019 | CNN-based synth parameter estimation, foundational work                 |
| DDSP (Engel et al.)        | 2020 | Differentiable digital signal processing, core concept                  |
| DiffMoog                   | 2024 | Differentiable Moog emulation, full-chain gradient flow                 |
| SynthRL                    | 2025 | RL formulation of inverse synthesis (REINFORCE through non-diff. synth) |
| InverSynth II              | 2023 | Neural proxy + inference-time finetuning, closest to this project       |
| Hayes et al. survey        | 2023 | "A review of differentiable DSP" (Frontiers), comprehensive overview    |
| intro2ddsp.github.io       | 2024 | Tutorial on differentiable audio synthesis                              |

### 9.5 Community Preset Sources

| Source                  | Volume | Notes                                             |
| ----------------------- | ------ | ------------------------------------------------- |
| Vital Audio Forum       | 2,000+ | Official community, mixed quality                 |
| PresetShare collections | 1,000+ | Curated packs                                     |
| GitHub preset repos     | Varies | Search "vital presets"                            |
| Vital's factory presets | ~300   | Ship with Vital, high quality, use for evaluation |

---

## 10. Hackathon Build Schedule

### Day 1: Data Pipeline (8–10 hours)

1. Install Vita, verify it works: render a preset, listen to it, export JSON.
2. Enumerate parameter space: write `get_controls()` output to a reference file.
3. Select your ~40 target parameters. Document which you chose and why.
4. Run perturbation analysis to compute importance weights (~30 min).
5. Generate 15–20K samples with rejection sampling and multi-pitch rendering.
6. Verify end-to-end: pick a random sample, load its params, re-render, confirm
   audio matches.
7. Precompute mel spectrograms and cache to disk for fast training.

### Day 2: Model Training (8–10 hours)

1. Train ResNet-18 baseline. Target: converged in 30–60 min. Log to wandb.
2. Evaluate: render top-10 validation predictions through Vita, listen, compute
   MRSTFT.
3. Train AST or MERT + MLP head (frozen encoder). Compare to ResNet baseline.
4. Iterate: adjust importance weights, learning rate, head architecture if
   needed.
5. If time: implement CMA-ES refinement on a few examples, measure improvement.

### Day 3: Demo and Polish (6–8 hours)

1. Build Gradio interface: upload audio → predict → render → A/B compare →
   export .vital.
2. Add LLM tutorial generation tab with proper parameter-to-language mapping.
3. If CMA-ES works: add "refine" button that runs 50–100 iterations and shows
   improvement.
4. Record demo video. Prepare slides if presenting.
5. Write up results: parameter MSE, MRSTFT distance, A/B listening observations.

---

## 11. What Would Make This Publishable

A hackathon demo is not a paper. To reach publication quality:

1. **Quantitative evaluation** against a baseline (Syntheon or a naive
   nearest-neighbor in CLAP embedding space) across 500+ held-out samples,
   reporting parameter MSE, MRSTFT, MFCC distance, LSD, and CLAP similarity with
   confidence intervals.

2. **Encoder ablation study** comparing ResNet-18, AST, MERT, and AudioMAE with
   identical heads and training data. This alone is a contribution if done
   rigorously.

3. **Listening study** with 10+ participants rating A/B similarity on a Likert
   scale, plus a "which is the reconstruction?" forced choice task.

4. **Analysis of failure modes**: which parameter categories are hardest to
   predict? Does the model confuse filter type X for filter type Y? Are errors
   correlated (e.g., always underestimates cutoff when resonance is high)?

5. **Modulation handling** at Stage 2+ level, with analysis of how much
   perceptual quality improves when modulation is included vs. static-only.

Venue targets: DAFx (focused, peer audience), ISMIR (if framing around MIR/music
understanding), ICASSP (broader signal processing), NeurIPS workshop on machine
learning for audio.
