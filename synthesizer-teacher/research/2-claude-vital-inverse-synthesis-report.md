# Inverse synthesis for Vital: a practical implementation reference

**Building an ML model to predict Vital synthesizer parameters from audio is
tractable — the existing baseline (Syntheon) only predicts a single wavetable
plus ADSR, leaving ~95% of Vital's parameter space untouched.** This guide
covers every technical detail needed to build a significantly better system: the
Vita rendering API, audio encoder architectures (MERT, AST), spectral loss
functions, modulation prediction strategies, dataset generation via Latin
Hypercube Sampling, and perceptual parameter weighting — all with exact code,
tensor shapes, and known gotchas.

## Vita renders Vital headlessly with full modulation control

The **Vita** Python library (v0.0.5, `pip install vita`, GPLv3) wraps Vital's
C++ DSP engine via pybind11, giving programmatic access to every parameter
including the modulation matrix — something that standard VST parameter APIs
cannot do.

**Core rendering workflow:**

```python
import vita
from scipy.io import wavfile

synth = vita.Synth()
synth.set_sample_rate(44100)
synth.set_bpm(120.0)

# Render: returns numpy array shaped (2, num_samples) — stereo
audio = synth.render(60, 0.7, 1.0, 3.0)  # pitch, velocity, note_dur, render_dur
wavfile.write("out.wav", 44100, audio.T)
```

**Parameter access** uses `get_controls()` returning a `dict[str, Control]`
where each Control exposes `.set(value)`, `.value()`, `.set_normalized(0-1)`,
and `.get_normalized()`. The `get_control_details(name)` method returns a
`ControlDetails` object with an `.options` list for discrete parameters (e.g.,
`get_control_details("delay_style").options` →
`["Mono", "Stereo", "Ping Pong", "Mid Ping Pong"]`). The
`get_control_text(name)` returns the human-readable display string for the
current value.

**Modulation routing** is handled through three module-level functions and one
instance method:

```python
sources = vita.get_modulation_sources()       # list[str], constant
destinations = vita.get_modulation_destinations()  # list[str], constant
synth.connect_modulation("lfo_1", "filter_1_cutoff")  # returns bool
controls = synth.get_controls()
controls["modulation_1_amount"].set(0.75)     # range: -1.0 to 1.0
synth.clear_modulations()                     # reset all routing
```

To enumerate every parameter and its normalized range programmatically:

```python
for name, ctrl in synth.get_controls().items():
    info = synth.get_control_details(name)
    opts = info.options if hasattr(info, 'options') and info.options else None
    print(f"{name}: norm={ctrl.get_normalized():.3f}, raw={ctrl.value():.3f}"
          + (f", options={opts}" if opts else ""))
```

Key parameter names include `osc_1_level`, `osc_1_wave_frame` (wavetable
position), `filter_1_cutoff`, `filter_1_resonance`, `modulation_N_amount`,
envelope parameters (`env_1_attack`, etc.), and effects (`reverb_dry_wet`,
`chorus_dry_wet`). The `.set_normalized()` method maps 0–1 to the full parameter
range, making it ideal for ML output normalization. **No official rendering
benchmarks exist**, but the 99.1% C++ codebase with SSE optimization and zero
GUI overhead means rendering is significantly faster than real-time for single
notes.

**Critical limitation**: Vita has no API to enumerate or load factory wavetables
directly. Wavetable data lives inside `.vital` preset files as base64-encoded
JSON. To change wavetables, load a preset containing the desired wavetable via
`synth.load_json()` or `synth.load_preset()`, or inject raw wavetable data into
the JSON blob. Vital ships **25 factory wavetables** (free tier), **70**
(Plus/$25), or **150** (Pro/$80). Factory wavetables are not redistributable per
Matt Tytel's licensing terms.

## MERT-95M extracts 768-dim music representations at 75 Hz

**MERT-v1-95M** (`m-a-p/MERT-v1-95M` on HuggingFace) provides strong music audio
representations from a self-supervised model trained on **160,000 hours** of
music. The architecture uses **12 transformer layers** with **768-dim** hidden
states, a 7-layer CNN feature extractor with total stride **320**, and a CQT
musical feature branch.

**Input requirements**: mono audio at **24,000 Hz**, pre-trained on **5-second**
clips (120,000 samples). Feature rate is **75 frames/second** (24000 ÷ 320). For
5s input, output sequence length is 375. Hidden states are a tuple of **13
tensors** (1 CNN output + 12 transformer layers), each shaped
`[batch, seq_len, 768]`.

**Frozen encoder with learnable layer-weighted pooling:**

```python
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torch, torch.nn as nn

class MERTEncoder(nn.Module):
    def __init__(self, output_dim, freeze=True):
        super().__init__()
        self.mert = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True)
        if freeze:
            for p in self.mert.parameters(): p.requires_grad = False
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.head = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 256),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(256, output_dim))

    def forward(self, x):  # x: [B, num_samples] at 24kHz
        with torch.no_grad():
            out = self.mert(x, output_hidden_states=True)
        stacked = torch.stack(out.hidden_states)  # [13, B, T, 768]
        w = torch.softmax(self.layer_weights, 0).view(-1, 1, 1, 1)
        pooled = (stacked * w).sum(0).mean(1)     # [B, 768]
        return self.head(pooled)                    # [B, output_dim]
```

**Known gotchas** that will bite you:

- `trust_remote_code=True` is **mandatory** — MERT uses custom modeling code
- **Incompatible with `transformers >= 4.44.0`**; pin to `transformers==4.38`
- **BFloat16 crashes** (`weight_norm` not implemented for BF16) — use FP32 or
  FP16 only
- Requires the `nnAudio` library for the CQT feature branch
  (`pip install nnAudio`)
- Audio must be resampled to exactly 24kHz; feeding 44.1kHz audio produces
  garbage
- Different layers excel at different tasks: lower layers capture local acoustic
  features (beat, pitch), higher layers capture global patterns (genre, mood)

## AST provides 768-dim audio embeddings at 16 kHz

The **Audio Spectrogram Transformer**
(`MIT/ast-finetuned-audioset-10-10-0.4593`) processes **16 kHz** mono audio
through a 128-bin log-Mel spectrogram, patched with stride 10×10 into **1,214
tokens** (12 frequency × 101 time patches + CLS + distillation), each with
**768-dim** embeddings across 12 transformer layers.

```python
from transformers import ASTModel, ASTFeatureExtractor

feat_ext = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593")
ast = ASTModel.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593")

# inputs["input_values"] shape: [1, 1024, 128]
inputs = feat_ext(waveform_16k, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    out = ast(**inputs, output_hidden_states=True)
# out.pooler_output: [B, 768] — mean of CLS+distillation tokens
# out.hidden_states: 13 × [B, 1214, 768]
```

AST inference is fast: **6ms per sample** with SDPA attention on an A100
(batch=1). Use `attn_implementation="sdpa"` with `torch>=2.1.1`. The model uses
AudioSet-specific normalization (mean=-4.2677, std=4.5689), so compute your own
dataset statistics for best results on synth audio.

**MERT vs AST tradeoff**: MERT was trained on music audio at 24kHz with CQT
features, making it inherently more suited to musical timbre tasks. AST was
trained on AudioSet (environmental sounds) at 16kHz. For synthesizer sound
matching, **MERT is likely the stronger choice** due to music-specific
pretraining, but AST's faster inference and native HuggingFace integration (no
`trust_remote_code`) make it a safer hackathon pick.

## auraloss provides drop-in multi-resolution spectral losses

Install with `pip install auraloss` (v0.3.0, Apache-2.0). The library provides
time-domain losses (ESR, SI-SDR, SNR), frequency-domain losses (STFT, MelSTFT,
MultiResolutionSTFT), and perceptual transforms (sum-and-difference, FIR
pre-emphasis).

**Recommended configuration for synthesizer audio comparison:**

```python
import auraloss

loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[1024, 2048, 8192],
    hop_sizes=[256, 512, 2048],
    win_lengths=[1024, 2048, 8192],
    scale="mel",              # mel-frequency scaling
    n_bins=128,               # mel bins
    sample_rate=44100,
    perceptual_weighting=True,  # A-weighting curve
    w_sc=1.0,                 # spectral convergence weight
    w_log_mag=1.0,            # log-magnitude L1 weight
    w_lin_mag=0.0,            # linear magnitude weight
    w_phs=0.0,               # phase loss weight
)

# Input shape: (batch, channels, seq_len), e.g., (8, 1, 132300)
loss = loss_fn(predicted_audio, target_audio)
```

The three FFT sizes capture **transient detail** (1024), **mid-range spectral
structure** (2048), and **low-frequency resolution** (8192). Mel scaling with
A-weighting aligns the loss with human hearing sensitivity. **Combine with a
time-domain loss** for best results — spectral losses alone can miss fine
temporal alignment:

```python
class SynthLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 8192], hop_sizes=[256, 512, 2048],
            win_lengths=[1024, 2048, 8192], scale="mel", n_bins=128,
            sample_rate=44100, perceptual_weighting=True)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.mrstft(pred, target) + 100.0 * self.l1(pred, target)
```

## Syntheon is a weak baseline predicting only wavetable plus ADSR

**Syntheon** (github.com/gudgud96/syntheon, 159 stars, Apache-2.0) uses a
**WTSv2 model** (~7M parameters) based on Differentiable Wavetable Synthesis
(Shan et al., ICASSP 2022). It is fundamentally a DDSP autoencoder adapted from
`acids-ircam/ddsp_pytorch`. The pipeline in `vital_inferencer.py` preprocesses
audio at **16 kHz** (extracting pitch, loudness, MFCCs), runs a forward pass
producing a single learned wavetable shape plus ADSR envelope parameters, then
converts these to a `.vital` JSON preset.

**What Syntheon does not predict** constitutes the vast majority of Vital's
capability: no multi-oscillator configurations, no filter parameters (Vital has
2 filters with 32+ types), no effects chain (chorus, phaser, delay, reverb,
compressor, EQ, distortion, flanger), no LFO or modulation routing, no
unison/detune, no spectral warping, and no oscillator routing. A Vital forum
user summarized it accurately: _"Just seems like harmonic resynthesis with an
envelope."_ The model is not even trained on Vital-generated audio — it uses the
NSynth dataset of general instrument recordings.

**Beating Syntheon requires**: (1) training on actual Vital preset→audio pairs,
(2) predicting the full parameter space including filters and effects, (3)
handling modulation routing, and (4) using a perceptual encoder (MERT/AST)
rather than hand-crafted MFCCs.

## Fixed modulation matrices are the pragmatic hackathon approach

For predicting Vital's modulation routing (source→destination connections with
amounts), five approaches exist in the literature, but only one is practical for
a hackathon.

**Recommended: predict a dense N_sources × N_destinations matrix** where each
cell is a modulation amount in [-1, 1] and zero means no connection. With ~8
modulation sources and ~20 key destinations, this is just 160 regression outputs
— trivially small.

```python
class ModulationPredictor(nn.Module):
    def __init__(self, input_dim, n_src=8, n_dst=20):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, n_src * n_dst))
        self.n_src, self.n_dst = n_src, n_dst

    def forward(self, x):
        return torch.tanh(self.head(x)).view(-1, self.n_src, self.n_dst)

# Train with MSE + L1 sparsity penalty
mse = F.mse_loss(pred_matrix, target_matrix)
sparsity = torch.mean(torch.abs(pred_matrix))
loss = mse + 0.01 * sparsity

# At inference, threshold to clean routing:
pred_matrix[pred_matrix.abs() < 0.05] = 0
```

The L1 penalty encourages the model to learn sparse connections (most synth
patches use fewer than 10 modulation routings out of hundreds possible). More
sophisticated alternatives — DETR-style Hungarian matching, slot attention,
graph generation — exist but add significant complexity without proportional
benefit for this problem size. The **InverSynth** paper (Barkan et al., 2019)
already demonstrated that flat parameter vector prediction works well for
synthesizer parameter estimation.

## 4,500+ free presets and LHS sampling enable large dataset creation

**Free Vital presets are abundant**: PresetShare hosts **~3,800** community
presets, the HipHopMakers aggregation page catalogs **4,500+** across 20+
sources, and GitHub repos like `atsushieno/open-vital-resources` (CC0/CC-BY
licensed) and `jpriebe/qub1t-vital-presets` add hundreds more. All `.vital`
files are **plain JSON** with base64-encoded wavetable data embedded — fully
self-contained and parseable. At ~1KB–1MB each (depending on embedded wavetable
data), 500 presets total roughly 0.5 GB.

For synthetic dataset generation beyond existing presets, **Latin Hypercube
Sampling** efficiently covers Vital's ~100-dimensional parameter space:

```python
from scipy.stats import qmc
import numpy as np

d = 100   # ~100 Vital parameters
n = 5000  # 50× dimensionality for good coverage

sampler = qmc.LatinHypercube(d=d, optimization="random-cd", seed=42)
samples = sampler.random(n=n)  # (5000, 100) in [0, 1)

# Categorical params: discretize via floor
filter_types = ['analog', 'dirty', 'ladder', 'comb', 'phaser']
filter_idx = np.floor(samples[:, 42] * len(filter_types)).astype(int)
filter_idx = np.clip(filter_idx, 0, len(filter_types) - 1)

# Log-scaled perceptual params (cutoff 20–20000 Hz)
cutoff = 10 ** (np.log10(20) + (np.log10(20000) - np.log10(20)) * samples[:, 10])
```

Use `optimization="random-cd"` for best space-filling via centered discrepancy
minimization. For **100+ dimensions**, **1,000–5,000 samples** provide
reasonable coverage (rule of thumb: n ≥ 10d). For mixed continuous+categorical
spaces, sample everything as continuous in [0,1) then post-hoc discretize
categorical columns with `floor(value × n_categories)` — this preserves LHS
stratification while giving approximately uniform category coverage.

## Filter cutoff dominates perceptual importance, attack time ranks second

Convergent evidence from psychoacoustics and ML inverse synthesis research
establishes a clear parameter importance hierarchy. MDS timbre studies (Grey
1977, McAdams 1995, Caclin 2005) consistently identify **spectral centroid**
(controlled by filter cutoff and wavetable position) as explaining **40–50% of
perceptual variance**, **attack time** as explaining **25–30%**, and **spectral
flux** (filter envelope modulation) as explaining **15–20%**.

The InverSynth paper (Barkan et al., 2019) confirmed this computationally:
**filter parameters had the highest prediction accuracy** (~85–90%) because they
leave the clearest spectral signature, while ADSR envelope parameters were
hardest (~60–70%). The PNP loss paper (Han & Lostanlen, 2024) provides the most
principled approach: weight parameter errors by the **Jacobian of the
synthesis→perception map**, so parameters causing larger perceptual changes get
proportionally higher loss weight.

For a practical weighted parameter loss in wavetable synthesis:

| Parameter group                | Weight       | Rationale                           |
| ------------------------------ | ------------ | ----------------------------------- |
| Filter cutoff                  | **2.0–3.0×** | #1 perceptual dimension universally |
| Wavetable position             | **1.5–2.0×** | Determines base spectral content    |
| Amp attack time                | **1.5–2.0×** | #2 perceptual dimension             |
| Filter resonance               | **1.5×**     | Creates formant-like spectral peaks |
| Filter envelope amount/decay   | **1.5×**     | Drives spectral flux                |
| Envelope decay/sustain/release | **1.0×**     | Important but spectrally subtler    |
| Oscillator detune/unison       | **1.0×**     | Moderate perceptual impact          |
| Effects (reverb, chorus)       | **0.5–0.8×** | Secondary processing                |

Multiple papers (Masuda 2021, Han 2024) demonstrate that **spectral loss
outperforms pure parameter loss** for perceptual sound matching. The recommended
strategy is a **hybrid loss**: weighted parameter loss for initial convergence,
plus multi-resolution spectral loss (via auraloss) for perceptual fidelity.
Pre-training with parameter loss then fine-tuning with spectral loss produces
the best results.

## Conclusion

The implementation path is clear: use **Vita** to render training data from
thousands of existing free presets plus LHS-sampled random configurations,
encode audio with **MERT-95M** (freezing all 12 layers, learning layer-wise
pooling weights), predict the full Vital parameter vector with
**perceptually-weighted** parameter loss plus **auraloss
MultiResolutionSTFTLoss** with mel scaling, handle modulation as a **fixed dense
matrix** with L1 sparsity, and treat wavetable selection as a **categorical
embedding**. This architecture addresses every limitation of Syntheon —
multi-oscillator support, filter/effects prediction, modulation routing,
music-specific audio encoding, and training on actual Vital preset-audio pairs
rather than generic NSynth data. The combination of ~4,500 real presets plus
~5,000 LHS-sampled configurations provides a training set that should be
sufficient for a strong initial model, with the Vita rendering pipeline
generating audio at well above real-time speeds. Pin `transformers==4.38`,
install `nnAudio`, never use BFloat16 with MERT, and keep your FFT sizes at
`[1024, 2048, 8192]`.
