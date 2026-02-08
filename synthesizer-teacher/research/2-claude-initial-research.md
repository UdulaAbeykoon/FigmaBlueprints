# Inverse synthesis: building an ML model to reverse-engineer synthesizer parameters from audio

**Given any audio sample, a well-designed neural network can now predict the
exact synthesizer knob settings needed to recreate that sound** — a problem
known as inverse synthesis or automatic synthesizer programming. The field has
matured rapidly since 2018, progressing from simple CNNs on spectrograms to
transformer encoders, differentiable synthesizers, reinforcement learning
agents, and even diffusion-based parameter generation. For a hackathon targeting
the Vital wavetable synth specifically, the most promising approach combines a
pretrained audio transformer encoder (AudioMAE or BEATs) with a transformer
decoder predicting Vital's **~220–250 parameters**, trained on a massive
synthetic dataset generated via the Vita Python library. This report provides a
complete blueprint: every relevant paper, every useful repo, a cutting-edge
model architecture, and a concrete implementation plan.

---

## The academic landscape: from CNNs to transformers and RL

Research on inverse synthesis spans roughly seven years and four distinct waves
of methodology. Understanding this evolution is essential for designing a
state-of-the-art system.

### Wave 1: CNNs and classical baselines (2017–2019)

The problem was first formalized by **Yee-King, Fedden & d'Inverno (2018)** in
"Automatic Programming of VST Sound Synthesizers Using Deep Networks," which
compared hill climbers, genetic algorithms, MLPs, LSTMs, and CNNs on the Dexed
(DX7) FM synthesizer. Their Bi-LSTM with highway layers performed best.
**InverSynth (Barkan et al., 2019)** then proposed strided 2D CNNs operating on
STFT spectrograms, quantizing each of the DX7's 23 parameters into 16 bins and
framing the task as multi-label classification. This quantization insight —
treating continuous knobs as discrete classes — proved surprisingly effective
and influenced later work.

### Wave 2: VAEs, normalizing flows, and latent spaces (2019–2023)

**FlowSynth (Esling et al., 2020)** introduced normalizing flows to learn an
invertible mapping between a latent audio space and the parameter space of the
u-he Diva synthesizer. This enabled not just parameter inference but also smooth
preset interpolation and macro-control discovery. **PresetGenVAE (Le Vaillant &
Dutoit, 2021)** used a convolutional VAE with multi-channel spectral input,
publishing a **30,000-preset Dexed dataset** that remains a key benchmark. The
same team later developed **SPINVAE (ICASSP 2023)**, the first model to encode
synthesizer presets with transformer blocks and multi-head attention, enabling
smooth latent interpolation. Its successor, **SPINVAE-2 (2024)**, added
timbre-regularized latent dimensions tied to perceptual attributes.

### Wave 3: Differentiable DSP and end-to-end optimization (2020–2024)

**DDSP (Engel et al., ICLR 2020)** was the watershed moment — Google's
differentiable digital signal processing library integrated oscillators,
filters, and reverb as differentiable operations, enabling gradient-based audio
loss optimization. Key derivatives include:

- **DDX7 (Caspe et al., ISMIR 2022)**: differentiable FM synthesis for
  instrument resynthesis
- **Differentiable Wavetable Synthesis (Shan et al., ICASSP 2022)**: learns
  wavetable dictionaries through end-to-end training, achieving high fidelity
  with only 10–20 wavetables
- **Masuda & Saito (ISMIR 2021, extended 2023)**: differentiable subtractive
  synthesizer enabling spectral loss during training, with semi-supervised
  hybrid parameter+audio losses
- **DiffMoog (Uzrad et al., 2024)**: an open-source differentiable modular
  synthesizer with FM/AM, LFOs, filters, envelopes, and a novel signal-chain
  loss. Demonstrated that **Wasserstein loss outperforms L1/L2 for frequency
  estimation**
- **InverSynth II (Barkan et al., ISMIR 2023)**: trains a neural
  synthesizer-proxy (a network approximating the synth's output), then
  backpropagates through it. Introduces inference-time finetuning (ITF) that
  self-supervises predictions at test time. **State-of-the-art on Dexed (144
  parameters) and TAL-NoiseMaker**

### Wave 4: Transformers, RL, and multi-modal approaches (2023–2025)

The latest papers represent the cutting edge. **Bruford et al. (DAFx 2024)**
applied the Audio Spectrogram Transformer (AST) to sound matching on NI Massive,
outperforming MLP and CNN baselines and showing out-of-domain generalization to
vocal imitations. **SynthRL (IJCAI 2025)** is the first RL-based approach — a
CNN+Transformer encoder-decoder uses REINFORCE to optimize synthesis parameters
without ground-truth labels, enabling fine-tuning on out-of-domain sounds using
a reward function combining spectral convergence, spectral centroid, and MFCC
distances. **CTAG (ICML 2024)** demonstrated text-to-audio via synthesizer
programming, using CLAP embeddings and evolutionary strategies to optimize
parameters of a **78-parameter SynthAX modular synth in JAX**. Finally, **Neural
Proxies for Sound Synthesizers (Combes et al., 2025)** trains compact models
mapping presets to audio embeddings from CLAP, OpenL3, and PANNs, creating
differentiable bridges to black-box synthesizers.

### Additional important papers

**Sound2Synth (IJCAI 2022)** introduced multi-modal feature extraction with a
Prime-Dilated Convolution network for FM parameter estimation. **SerumRNN
(EvoMUSART 2021)** tackled step-by-step effect chain programming for Serum.
**SynthScribe (IUI 2024)** built a full multimodal search and creation system.
The **Hayes et al. survey (Frontiers in Signal Processing, 2023)** provides the
most comprehensive overview of all differentiable DSP work. The **PNP Loss paper
(2024)** achieved 100× speedup over DDSP by linearizing the
synthesizer-to-perception Jacobian.

No published paper has specifically targeted Vital. **This is a genuine gap in
the literature**, making this hackathon project potentially publishable.

---

## Every open-source tool that matters

The ecosystem of repos can be organized into four tiers based on relevance to
building a Vital inverse synthesis system.

### Tier 1: Critical infrastructure

**Vita** (github.com/DBraun/Vita) is the single most important tool. It provides
native Python bindings to Vital's C++ synthesis engine — no VST plugin needed.
It supports `synth.render(pitch, velocity, note_dur, render_dur)` returning
NumPy arrays, `synth.get_controls()` for the full parameter dictionary,
`synth.connect_modulation()` for routing, and
`synth.to_json()`/`synth.load_json()` for complete preset I/O. Install with
`pip install vita`. This is what you'll use to generate your training dataset
and to convert model predictions back to playable presets.

**Syntheon** (github.com/gudgud96/syntheon, ~158 stars) is the closest existing
inverse synthesis tool for Vital. It uses a differentiable wavetable synthesis
model internally and has a `vital_inferencer.py` that outputs `.vital` preset
files directly. The API is simple:
`from syntheon import infer_params; output, eval = infer_params("audio.wav", "vital")`.
Use this as your **baseline to beat**.

**DawDreamer** (github.com/DBraun/DawDreamer, ~929 stars) is a Python DAW
framework for hosting VST plugins, Faust DSP, and complex audio routing. It
supports JAX transpilation, making any Faust synthesizer differentiable. Its
multiprocessing example shows how to batch-render presets to audio at scale.
Note: DawDreamer cannot access Vital's modulation matrix via standard VST
parameters (Issue #212) — Vita solves this.

**torchsynth** (github.com/torchsynth/torchsynth) is a GPU-accelerated modular
synthesizer in PyTorch running **16,200× faster than realtime**. It returns
audio with underlying parameters per batch:
`voice = Voice(); audio, params, is_train = voice(batch_id)` yields
`[128, 176400]` audio with `[128, 72]` parameters. Ideal for differentiable
training and augmentation.

### Tier 2: Research references and datasets

**SpiegeLib** (github.com/spiegelib/spiegelib) provides a complete framework for
auto-synth programming with dataset generation, model training (MLP, LSTM, CNN),
genetic algorithms, and evaluation. **Sound2Synth**
(github.com/Sound2Synth/Sound2Synth) implements the IJCAI 2022 FM parameter
estimation pipeline. **preset-gen-vae** (github.com/gwendal-lv/preset-gen-vae)
includes a **30K+ Dexed preset SQLite database** with pre-rendered audio — the
largest published synth preset dataset. **DX7-JAX** (github.com/DBraun/DX7-JAX)
provides 44,884 de-duplicated DX7 presets as a differentiable JAX synthesizer.
**SSSSM-DDSP** (github.com/hyakuchiki/SSSSM-DDSP) bridges semi-supervised
learning with differentiable synthesis. **diff-wave-synth**
(github.com/gudgud96/diff-wave-synth) is the unofficial PyTorch implementation
of Differentiable Wavetable Synthesis used inside Syntheon.

### Tier 3: Pretrained audio models (encoders)

**AudioMAE** (github.com/facebookresearch/AudioMAE) provides ViT-B/16 pretrained
on AudioSet-2M with 768-dim patch embeddings. **BEATs**
(github.com/microsoft/unilm/tree/master/beats) achieves SOTA audio
classification with iterative self-supervised acoustic tokenization.
**LAION-CLAP** (github.com/LAION-AI/CLAP) offers joint audio-text embeddings
enabling text-described sound matching. **AST** (github.com/YuanGongND/ast)
applies Vision Transformer directly to spectrograms. All have HuggingFace
integration and pretrained weights.

### Tier 4: LLM-audio integration

**Qwen-Audio** (github.com/QwenLM/Qwen-Audio) uses a Whisper encoder fused with
Qwen-7B for multi-modal audio understanding. **MU-LLaMA**
(github.com/shansongliu/MU-LLaMA) pairs a MERT music encoder with LLaMA-2.
**SLAM-LLM** (github.com/X-LANCE/SLAM-LLM) provides a general framework for
speech/audio/music LLM integration. These could power the "human-readable
tutorial generation" step of the pipeline.

### Vital-specific preset resources

Existing Vital preset collections on GitHub include **Miserlou/VitalPresets**
(bass, trap, DnB), **jpriebe/qub1t-vital-presets**, and **alanjyu/vital_sounds**
(CC BY-SA). **Vinetics** (github.com/SlavaCat118/Vinetics) uses genetic
algorithms to evolve Vital presets. The commercial tool **MicroMusic**
(micromusic.tech) was trained on **1M+ randomly generated Vital presets**,
demonstrating the feasibility of the inverse problem at scale.

---

## Insights from the 0xdevalias gist

The GitHub gist at `0xdevalias/5a06349b376d01b2a76ad27a86b08c1b` is the most
comprehensive community reference document on AI-driven synth patch generation,
with **58 stars, 8 forks, and 61 revisions** as of late 2025. It serves as a
living index covering people to follow (Dadabots, gudgud96/Hao Hao Tan), AI
tools (MicroMusic, Syntheon, Neutone, Synplant's Genopatch), target synthesizers
(Vital, Serum, Massive), and critically, practical implementation details.

The gist's most valuable insights concern the **rendering bottleneck and preset
format accessibility**. Vital's `.vital` files are plain JSON, making them
trivially parseable and generable — a massive advantage over Serum's
Zlib-compressed binary `.fxp` format. The gist documents that DawDreamer's
`save_state()`/`load_state()` can capture modulation matrix data that
`set_parameter()` cannot access, but recommends **Vita as the superior tool**
since it provides direct `connect_modulation()` API access. It also notes that
SynthAX (JAX, 90,000× realtime) and torchsynth (PyTorch, 16,200× realtime) offer
GPU-native alternatives to VST-based rendering for gradient-based approaches.
The gist tracks GAN-based preset generators including **NeuralDX7** (DX7
cartridge GAN), **Synth1GAN**, and **ThisPatchDoesNotExist**.

---

## Proposed model architecture: VitalInverse

The following architecture combines the strongest ideas from the literature into
a system that advances the state of the art by targeting Vital specifically,
using modern pretrained encoders, and handling the full complexity of Vital's
mixed parameter space including modulation routing.

### Audio encoder: frozen AudioMAE with learned adapter

Use **AudioMAE** (ViT-B/16, pretrained on AudioSet-2M) as a frozen feature
extractor. It processes 128-bin mel spectrograms into 512 patch embeddings of
dimension 768. Add a lightweight **LoRA adapter** (rank 16) to the last 4
transformer layers to adapt general audio features toward synthesizer timbre
without catastrophic forgetting. The CLS token provides a global 768-dim
representation; patch tokens provide frame-level detail. Alternative: **BEATs**
if semantic-level features prove more useful for categorical parameter
prediction (filter type, waveform type).

### Parameter decoder: conditional transformer with split heads

The decoder is a **6-layer transformer decoder** with cross-attention to the
encoder's patch sequence. It uses **N learned query tokens**, one per parameter
group (oscillator 1, oscillator 2, oscillator 3, filter 1, filter 2, envelope
1–3, LFO 1–4, effects chain, global settings), each attending to relevant audio
features. This follows the SynthRL architecture's insight that learned queries +
cross-attention outperform flat regression.

Output heads are split by parameter type:

- **Continuous parameters** (knob positions, ~160 values): Sigmoid-bounded
  regression with MSE loss
- **Discrete/categorical parameters** (filter types, waveform selectors, ~40
  values): **Gumbel-Softmax** with temperature annealing (τ: 1.0→0.1 over
  training) for differentiable discrete sampling
- **Binary toggles** (on/off switches, ~30 values): Binary cross-entropy with
  sigmoid
- **Modulation routing** (variable-length, up to 64 connections): A small **set
  prediction head** using a learned modulation vocabulary — predict K modulation
  slots as (source_id, destination_id, amount) tuples using Gumbel-Softmax for
  source/destination selection and sigmoid for amount

### Loss function: three-component hybrid

**L_total = λ₁·L_param + λ₂·L_spectral + λ₃·L_perceptual**

1. **L_param**: Direct parameter supervision — MSE for continuous, cross-entropy
   for discrete, BCE for binary. Weight λ₁=1.0.
2. **L_spectral**: Multi-scale spectral loss computed by rendering predicted
   parameters through a **neural Vital proxy** (a small WaveNet trained to
   approximate Vital's output, following InverSynth II). Use 6 FFT sizes (64,
   128, 256, 512, 1024, 2048) with both linear and log magnitude distances.
   Weight λ₂=0.5.
3. **L_perceptual**: CLAP embedding cosine similarity between rendered and
   target audio, providing semantic-level guidance. Weight λ₃=0.3.

### Reinforcement learning refinement (optional, for out-of-domain sounds)

After supervised pretraining, fine-tune using **PPO** (not REINFORCE as in
SynthRL — PPO is more stable) with reward = negative multi-scale spectral
distance between target audio and Vital-rendered prediction. This enables the
model to improve on real-world sounds without ground-truth parameters, following
SynthRL's framework. The RL phase uses Vita for rendering in the environment
loop.

### Diffusion-based alternative for multi-modal outputs

Inverse synthesis is fundamentally a **one-to-many problem** — many parameter
configurations can produce perceptually similar sounds. A **conditional
diffusion model** in parameter space (conditioning on AudioMAE features via
cross-attention) can generate diverse valid solutions rather than collapsing to
a mean. This follows the "Diffusion with Forward Models" framework (NeurIPS
2023). During inference, run the diffusion process for 50 steps with
classifier-free guidance, then select the best of K samples based on spectral
distance to the target.

---

## Vital's parameter space: what the model must predict

Vital exposes approximately **220–250 named scalar parameters** in its JSON
preset format, plus variable-length modulation routing and embedded
wavetable/LFO shape data.

| Category               | Count | Type                  | Examples                                                   |
| ---------------------- | ----- | --------------------- | ---------------------------------------------------------- |
| Oscillator params (×3) | ~45   | Continuous + discrete | level, pan, transpose, wavetable_position, distortion_type |
| Sampler                | ~8    | Mixed                 | level, pan, transpose                                      |
| Filters (×2)           | ~20   | Mixed                 | cutoff, resonance, drive, model (32+ types)                |
| Envelopes (×3)         | ~12   | Continuous            | attack, decay, sustain, release                            |
| LFOs (×4)              | ~16   | Mixed                 | frequency, sync_type, smooth                               |
| Effects (9 slots)      | ~80   | Mixed                 | dry_wet, on/off, mode per effect                           |
| Modulation routing     | 0–64  | Structured            | (source, destination, amount, bipolar, stereo)             |
| Macros                 | 4     | Continuous            | macro1–4 values                                            |
| Global                 | ~20   | Mixed                 | polyphony, oversampling, pitch_wheel_range                 |

For a hackathon MVP, **simplify to ~100 core parameters**: 3 oscillators (level,
transpose, wavetable position, unison count, detune), 2 filters (cutoff,
resonance, model), 3 envelopes (ADSR), master volume, and up to 8 modulation
routings. Ignore effects initially. This captures the essential timbral
character while keeping the problem tractable.

---

## Hackathon implementation plan

### Phase 1: Dataset creation (hours 0–6)

**Goal**: Generate 100K–500K audio-parameter pairs from random Vital presets.

```python
import vita, numpy as np, json, soundfile as sf

synth = vita.Synth(sample_rate=44100)
controls = synth.get_controls()  # dict of all ~250 parameters

for i in range(500_000):
    # Randomize parameters within valid ranges
    preset = generate_random_preset(controls)
    synth.load_json(json.dumps(preset))

    # Render 2-second note at middle C
    audio = synth.render(pitch=60, velocity=100,
                         note_duration=1.5, render_duration=2.0)

    # Save audio + parameters
    sf.write(f"data/{i}.wav", audio.T, 44100)
    save_params(f"data/{i}.json", preset["settings"])
```

Use **multiprocessing** (Vita is process-safe) across CPU cores. At ~50ms per
render, 500K samples takes ~7 hours on 8 cores. Alternatively, use **stratified
random sampling** — cluster parameter space into meaningful regions (pads,
leads, basses, plucks) and sample uniformly within each.

**Data augmentation**: Add pitch variation (render at MIDI 48, 60, 72), velocity
variation, and slight audio degradation (noise, EQ) to improve robustness.

### Phase 2: Model training (hours 6–18)

**Tech stack**:

- PyTorch 2.x with `torch.compile` for speed
- HuggingFace `transformers` for AudioMAE/BEATs backbone
- `torchaudio` for mel spectrogram computation
- `wandb` for experiment tracking
- Single CUDA GPU (A100/4090 ideal; 3090/3080 workable)

**Training recipe**:

1. **Stage 1 — Supervised pretraining** (10 epochs, ~8 hours on A100): Freeze
   AudioMAE encoder. Train transformer decoder with parameter loss only.
   Learning rate 3e-4 with cosine decay. Batch size 32.

2. **Stage 2 — Fine-tune with spectral loss** (5 epochs, ~4 hours): Unfreeze
   AudioMAE LoRA adapters. Add multi-scale spectral loss using a pre-trained
   neural Vital proxy. Learning rate 1e-4.

3. **Stage 3 — RL refinement** (optional, 2 hours): PPO with Vita rendering in
   the loop. Reward = negative spectral distance. Small learning rate 1e-5.

For the neural Vital proxy (needed for differentiable spectral loss), train a
lightweight WaveNet or 1D U-Net on 50K preset→audio pairs before the main model.
This proxy approximates Vital's output differentiably.

### Phase 3: Inference pipeline (hours 18–22)

```
Audio Input (.wav)
    → Mel Spectrogram (torchaudio)
    → AudioMAE Encoder (frozen + LoRA)
    → Transformer Decoder (cross-attention)
    → Split Heads → Predicted Parameters
    → Vita: render + compare (optional iterative refinement)
    → Export .vital preset file (JSON)
    → LLM: generate tutorial from parameters
```

**Iterative refinement at inference**: Following InverSynth II, use the
predicted parameters as initialization, then run 50 steps of gradient descent
through the neural proxy to minimize spectral loss against the target audio.
This **inference-time finetuning** consistently improves results by 10–20%.

### Phase 4: LLM tutorial generation (hours 22–24)

Feed the predicted Vital parameters into an LLM (GPT-4o, Claude, or local
Llama-3) with a prompt like:

```
Given these Vital synthesizer parameters: {param_dict}
Generate a step-by-step tutorial explaining how a sound designer
would recreate this sound from an initialized Vital patch.
Explain what each parameter does and why it was set this way.
```

This produces human-readable instructions like: "Start with Oscillator 1 using a
sawtooth wavetable at position 0.35. Set the filter cutoff to 2.4 kHz with
resonance at 0.6. Route Envelope 2 to filter cutoff with an amount of 0.7..."

### Recommended tech stack summary

| Component            | Tool                                 | Why                                       |
| -------------------- | ------------------------------------ | ----------------------------------------- |
| Synth rendering      | **Vita** (`pip install vita`)        | Native Vital bindings, headless, JSON I/O |
| Audio features       | **AudioMAE** via HuggingFace         | Best general audio encoder, pretrained    |
| Deep learning        | **PyTorch 2.x**                      | Industry standard, `torch.compile` speed  |
| Spectrograms         | **torchaudio**                       | GPU-accelerated mel computation           |
| Experiment tracking  | **Weights & Biases**                 | Free for individuals                      |
| RL (optional)        | **Stable-Baselines3**                | Clean PPO implementation                  |
| LLM integration      | **OpenAI API** or **Ollama** (local) | Tutorial generation                       |
| Differentiable synth | **torchsynth** or **SynthAX**        | For proxy training / augmentation         |

---

## Learning resources for the relevant ML techniques

### Differentiable DSP (start here)

The best single resource is the **"Introduction to Differentiable Audio
Synthesiser Programming" web book** (intro2ddsp.github.io) from the ISMIR 2023
tutorial by Ben Hayes et al. It contains 15+ Jupyter notebooks covering PyTorch
fundamentals, differentiable oscillators, additive synthesis, FIR/IIR filters,
and physical modeling. The companion **DDSP review paper** (Hayes et al.,
Frontiers in Signal Processing 2023) provides a comprehensive survey. Google's
original **DDSP Colab notebooks** in the magenta/ddsp repo remain excellent for
hands-on understanding of spectral losses and timbre transfer.

### Audio transformers

The **HuggingFace Audio Course** covers transformer architectures for audio
classification, speech recognition, and generation. For AST specifically, the
**Renumics fine-tuning tutorial** walks through end-to-end AST training on
custom datasets using HuggingFace. The SpeechBrain documentation provides
detailed tutorials for wav2vec 2.0, HuBERT, and BEATs integration.

### Inverse problems and sound matching

Read the **InverSynth II paper** (ISMIR 2023) for the neural proxy +
inference-time finetuning paradigm. Read **SynthRL** (IJCAI 2025) for the RL
formulation. The **DiffMoog paper** (2024) provides the most complete
open-source framework for understanding differentiable sound matching
end-to-end. **SpiegeLib's documentation and examples** (spiegelib.github.io)
offer practical code for dataset generation, model training, and evaluation on
VST synthesizers.

### Reinforcement learning for audio

The **SynthRL paper** is the primary reference. For PPO fundamentals, the
**Spinning Up in Deep RL** guide (spinningup.openai.com) by OpenAI and
**Stable-Baselines3 documentation** provide everything needed. Frame the problem
as: state = mel spectrogram of target audio; action = parameter vector; reward =
negative spectral distance after rendering.

---

## Conclusion: what makes this project novel and feasible

Three factors make Vital inverse synthesis both achievable at a hackathon and
genuinely novel. First, **no published paper targets Vital** — all academic work
uses Dexed, Massive, Diva, or custom synths. Vital's open-source JSON format and
the Vita Python library make it the most ML-accessible commercial-quality
synthesizer available. Second, **the combination of a pretrained audio
transformer encoder with a transformer parameter decoder has only been explored
once** (AST for Massive, DAFx 2024) and never with the AudioMAE/BEATs class of
self-supervised models. Third, **diffusion-based parameter generation for synth
programming is completely unexplored** in the literature — every existing system
produces a single point estimate, ignoring the fundamental one-to-many nature of
the inverse problem.

The minimum viable product for a hackathon is achievable in 24 hours: generate
100K Vital audio-parameter pairs via Vita, train an AudioMAE-encoder →
transformer-decoder model on the simplified 100-parameter space, and demonstrate
audio-in → Vital-preset-out inference with LLM-generated tutorials. The key
insight from MicroMusic's commercial success (trained on 1M+ random presets) is
that brute-force random preset generation works — the synthesizer parameter
space is more structured than it appears, and neural networks learn to navigate
it efficiently.
