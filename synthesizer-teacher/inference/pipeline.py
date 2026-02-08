"""Inference pipeline: load audio, predict parameters, export .vital preset."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

log = logging.getLogger(__name__)


class InferencePipeline:
    """Load a trained model and predict Vital parameters from audio.

    Usage:
        pipeline = InferencePipeline.from_checkpoint("checkpoints/best_model.pt")
        params = pipeline.predict("input_audio.wav")
        pipeline.export_vital_preset(params, "output.vital")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        continuous_names: list[str],
        categorical_names: list[str],
        categorical_n_options: list[int],
        sample_rate: int = 44100,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        device: str = "cuda",
        continuous_ranges: list[tuple[float, float]] | None = None,
        wavetable_catalog: Any | None = None,
        mod_source_names: list[str] | None = None,
        mod_destination_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.continuous_names = continuous_names
        self.categorical_names = categorical_names
        self.categorical_n_options = categorical_n_options
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.continuous_ranges = continuous_ranges  # (min, max) for each continuous param
        self.wavetable_catalog = wavetable_catalog  # WavetableCatalog or None
        self.mod_source_names = mod_source_names or []
        self.mod_destination_names = mod_destination_names or []

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        ).to(self.device)

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, device: str = "cuda"
    ) -> "InferencePipeline":
        """Load pipeline from a training checkpoint."""
        # Import here to avoid circular dependency
        from training.model import VitalInverseModel

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        continuous_names = ckpt["continuous_names"]
        categorical_names = ckpt["categorical_names"]
        categorical_n_options = ckpt["categorical_n_options"]
        n_continuous = ckpt["n_continuous"]
        config = ckpt.get("config", {})

        # Infer head type from state dict keys: MLP heads have nested indices
        # like "categorical_heads.0.0.weight", linear heads have "categorical_heads.0.weight"
        state_dict = ckpt["model_state_dict"]
        has_mlp_heads = any(
            k.startswith("categorical_heads.") and k.count(".") >= 3
            for k in state_dict
        )
        simple = not has_mlp_heads

        # Detect modulation head from checkpoint
        n_mod_sources = ckpt.get("n_mod_sources", 0)
        n_mod_destinations = ckpt.get("n_mod_destinations", 0)
        mod_source_names = ckpt.get("mod_source_names", [])
        mod_destination_names = ckpt.get("mod_destination_names", [])

        model = VitalInverseModel(
            n_continuous=n_continuous,
            categorical_n_options=categorical_n_options,
            mlp_hidden=config.get("mlp_hidden", 512),
            dropout=0.0,  # No dropout during inference
            freeze_early=False,
            simple_categorical_heads=simple,
            n_mod_sources=n_mod_sources,
            n_mod_destinations=n_mod_destinations,
        )
        model.load_state_dict(state_dict)

        if n_mod_sources > 0:
            log.info(
                "Modulation head loaded: %d sources x %d destinations",
                n_mod_sources, n_mod_destinations,
            )

        # Load wavetable catalog if embedded in checkpoint
        wavetable_catalog = None
        wt_json = ckpt.get("wavetable_catalog_json")
        if wt_json:
            try:
                from datagen.wavetables.catalog import WavetableCatalog
                wavetable_catalog = WavetableCatalog.from_json_string(wt_json)
                log.info(
                    "Loaded wavetable catalog from checkpoint: %d wavetables",
                    len(wavetable_catalog),
                )
            except Exception as e:
                log.warning("Failed to load wavetable catalog from checkpoint: %s", e)

        return cls(
            model=model,
            continuous_names=continuous_names,
            categorical_names=categorical_names,
            categorical_n_options=categorical_n_options,
            sample_rate=ckpt.get("sample_rate", config.get("sample_rate", 44100)),
            n_mels=config.get("n_mels", 128),
            n_fft=config.get("n_fft", 2048),
            hop_length=config.get("hop_length", 512),
            device=device,
            continuous_ranges=ckpt.get("continuous_ranges"),
            wavetable_catalog=wavetable_catalog,
            mod_source_names=mod_source_names,
            mod_destination_names=mod_destination_names,
        )

    def _load_and_compute_mel(self, audio_path: str | Path) -> torch.Tensor:
        """Load audio, resample, convert to mono, and compute mel spectrogram.

        Returns:
            Mel tensor of shape (1, 1, n_mels, T) ready for model input.
        """
        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Truncate to 4 seconds max to match training clip length
        max_samples = self.sample_rate * 4
        if waveform.shape[-1] > max_samples:
            log.info(
                "Truncating audio from %.2fs to 4.0s",
                waveform.shape[-1] / self.sample_rate,
            )
            waveform = waveform[..., :max_samples]

        # Minimum length check: mel spectrogram requires at least n_fft samples
        n_fft = self.mel_transform.n_fft
        if waveform.shape[-1] < n_fft:
            min_duration_ms = n_fft / self.sample_rate * 1000
            actual_ms = waveform.shape[-1] / self.sample_rate * 1000
            raise ValueError(
                f"Audio too short ({actual_ms:.0f}ms). "
                f"Minimum duration is {min_duration_ms:.0f}ms "
                f"({n_fft} samples at {self.sample_rate}Hz)."
            )

        waveform = waveform.to(self.device)
        mel = torch.log1p(self.mel_transform(waveform))  # (1, n_mels, T)
        return mel.unsqueeze(0)  # (1, 1, n_mels, T)

    def _extract_modulation(
        self,
        mod_matrix: torch.Tensor,
        threshold: float = 0.05,
        top_k: int = 20,
    ) -> dict:
        """Convert dense modulation matrix to connections dict.

        Args:
            mod_matrix: (4, n_src, n_dst) modulation matrix.
            threshold: Minimum |amount| to consider active.
            top_k: Maximum number of connections to keep.

        Returns:
            Dict with 'connections' list for RenderEngine.
        """
        amount = mod_matrix[0].cpu().numpy()  # (n_src, n_dst)
        connections = []
        for si in range(amount.shape[0]):
            for di in range(amount.shape[1]):
                if abs(amount[si, di]) > threshold:
                    connections.append({
                        "source": self.mod_source_names[si] if si < len(self.mod_source_names) else f"src_{si}",
                        "destination": self.mod_destination_names[di] if di < len(self.mod_destination_names) else f"dst_{di}",
                        "source_idx": si,
                        "dest_idx": di,
                        "amount": float(mod_matrix[0, si, di]),
                        "bipolar": float(mod_matrix[1, si, di]),
                        "power": float(mod_matrix[2, si, di]),
                        "stereo": float(mod_matrix[3, si, di]),
                    })
        # Rank by absolute amount, keep top-K
        connections.sort(key=lambda c: abs(c["amount"]), reverse=True)
        return {
            "connections": connections[:top_k],
            "n_sources": amount.shape[0],
            "n_destinations": amount.shape[1],
        }

    def predict(self, audio_path: str | Path) -> dict[str, Any]:
        """Predict Vital parameters from an audio file.

        Args:
            audio_path: Path to audio file (any format supported by torchaudio).

        Returns:
            Dict with parameter names as keys and predicted values.
            Continuous params are in [0, 1], categorical are integer indices.
            If modulation head exists, includes '_modulation_t3' key.
        """
        mel = self._load_and_compute_mel(audio_path)

        with torch.no_grad():
            cont_pred, cat_logits, mod_pred = self.model(mel)

        result: dict[str, Any] = {}

        cont_values = cont_pred.cpu().squeeze(0).numpy()
        for i, name in enumerate(self.continuous_names):
            result[name] = float(cont_values[i])

        for i, (name, logits) in enumerate(zip(self.categorical_names, cat_logits)):
            result[name] = int(logits.argmax(dim=1).item())

        if mod_pred is not None:
            result["_modulation_t3"] = self._extract_modulation(mod_pred.squeeze(0))

        return result

    def predict_with_confidence(
        self, audio_path: str | Path
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Predict with confidence scores for categorical params.

        Returns:
            (params_dict, confidence_dict) where confidence_dict maps
            categorical param names to their prediction probabilities.
        """
        mel = self._load_and_compute_mel(audio_path)

        with torch.no_grad():
            cont_pred, cat_logits, mod_pred = self.model(mel)

        result: dict[str, Any] = {}
        confidence: dict[str, float] = {}

        cont_values = cont_pred.cpu().squeeze(0).numpy()
        for i, name in enumerate(self.continuous_names):
            result[name] = float(cont_values[i])

        for i, (name, logits) in enumerate(zip(self.categorical_names, cat_logits)):
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            result[name] = int(pred_idx)
            confidence[name] = float(probs[0, pred_idx].item())

        if mod_pred is not None:
            result["_modulation_t3"] = self._extract_modulation(mod_pred.squeeze(0))

        return result, confidence

    def export_vital_preset(
        self,
        params: dict[str, Any],
        output_path: str | Path,
        preset_name: str = "ML Predicted",
    ) -> None:
        """Export predicted parameters as a .vital preset file.

        Args:
            params: Parameter dict from predict().
            output_path: Path for output .vital file.
            preset_name: Name to show in Vital's preset browser.
        """
        # Build Vital preset JSON structure
        preset = {
            "author": "Inverse Synthesis ML",
            "comments": "Preset predicted from audio using machine learning.",
            "macro1": "MACRO 1",
            "macro2": "MACRO 2",
            "macro3": "MACRO 3",
            "macro4": "MACRO 4",
            "preset_name": preset_name,
            "preset_style": "Experimental",
            "synth_version": "1.5.5",
            "settings": {},
        }

        # Add predicted parameters to settings, denormalizing continuous params
        settings = preset["settings"]
        warned_missing_ranges = False
        for name, value in params.items():
            if name.startswith("_"):
                continue
            if name in self.continuous_names:
                if not self.continuous_ranges:
                    if not warned_missing_ranges:
                        log.warning(
                            "continuous_ranges not available in checkpoint; "
                            "exporting normalized [0,1] values (preset may sound wrong)"
                        )
                        warned_missing_ranges = True
                else:
                    idx = self.continuous_names.index(name)
                    if idx < len(self.continuous_ranges):
                        lo, hi = self.continuous_ranges[idx]
                        value = lo + float(value) * (hi - lo)
                    elif not warned_missing_ranges:
                        log.warning(
                            "continuous_ranges incomplete (have %d, need %d); "
                            "some params exported as normalized [0,1]",
                            len(self.continuous_ranges),
                            len(self.continuous_names),
                        )
                        warned_missing_ranges = True
            settings[name] = value

        # Inject wavetable data for each oscillator
        if self.wavetable_catalog is not None:
            wavetables_array: list[dict] = []
            for osc_idx in range(1, 4):  # osc_1, osc_2, osc_3
                wt_name = f"osc_{osc_idx}_wavetable"
                if wt_name in params:
                    wt_index = int(params[wt_name])
                    try:
                        entry = self.wavetable_catalog.get_by_index(wt_index)
                        # base64_data is the JSON-serialized wavetable object
                        wt_obj = json.loads(entry.base64_data)
                        wavetables_array.append(wt_obj)
                    except (IndexError, json.JSONDecodeError) as e:
                        log.warning(
                            "Failed to inject wavetable for %s (index %d): %s",
                            wt_name, wt_index, e,
                        )
                        wavetables_array.append({})
                else:
                    wavetables_array.append({})
            if any(wavetables_array):
                settings["wavetables"] = wavetables_array

        # Inject modulation connections
        mod_data = params.get("_modulation_t3")
        if mod_data and "connections" in mod_data:
            # Build modulations array (64 slots, most empty)
            modulations = [
                {"source": "", "destination": ""}
                for _ in range(64)
            ]
            for slot_idx, conn in enumerate(mod_data["connections"]):
                if slot_idx >= 64:
                    break
                modulations[slot_idx] = {
                    "source": conn["source"],
                    "destination": conn["destination"],
                }
                # Set modulation slot params in settings
                slot_num = slot_idx + 1
                settings[f"modulation_{slot_num}_amount"] = conn["amount"]
                settings[f"modulation_{slot_num}_bipolar"] = round(conn.get("bipolar", 0.0))
                settings[f"modulation_{slot_num}_power"] = conn.get("power", 0.0)
                settings[f"modulation_{slot_num}_stereo"] = round(conn.get("stereo", 0.0))
            settings["modulations"] = modulations

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(preset, f, indent=2)

        log.info("Exported preset to: %s", output_path)

    def predict_with_refinement(
        self,
        audio_path: str | Path,
        midi_note: int = 60,
        sigma0: float = 0.1,
        max_evals: int = 500,
        timeout_sec: float = 60.0,
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
        """Predict parameters then refine with CMA-ES optimization.

        Args:
            audio_path: Path to audio file.
            midi_note: MIDI note for rendering during optimization.
            sigma0: CMA-ES initial step size.
            max_evals: Maximum function evaluations.
            timeout_sec: Timeout in seconds.

        Returns:
            (optimized_params, confidence_dict, refinement_info)
        """
        from inference.cma_optimizer import CMAESOptimizer

        # Step 1: Get initial prediction
        params, confidence = self.predict_with_confidence(audio_path)

        # Step 2: Load raw audio for spectral comparison
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        target_mono = waveform.squeeze(0).numpy()

        # Step 3: Create render engine and optimizer
        try:
            from datagen.config import PipelineConfig
            from datagen.render.engine import RenderEngine
        except ImportError:
            log.warning("Vita not available; skipping CMA-ES refinement")
            return params, confidence, {"skipped": True, "reason": "vita_unavailable"}

        config = PipelineConfig(sample_rate=self.sample_rate)
        engine = RenderEngine(config)
        optimizer = CMAESOptimizer(engine, self.continuous_names, self.sample_rate)

        # Step 4: Extract continuous vector and categorical dict
        cont_vector = np.array(
            [float(params[name]) for name in self.continuous_names],
            dtype=np.float64,
        )
        cat_dict: dict[str, Any] = {
            name: params[name]
            for name in self.categorical_names
            if name in params
        }
        # Attach wavetable catalog so CMA-ES renders use correct wavetables
        if self.wavetable_catalog is not None:
            cat_dict["_wavetable_catalog"] = self.wavetable_catalog
        # Pass modulation data through to CMA-ES renders
        if "_modulation_t3" in params:
            cat_dict["_modulation_t3"] = params["_modulation_t3"]

        # Step 5: Optimize
        refined_vector, info = optimizer.optimize(
            cont_vector, cat_dict, target_mono,
            midi_note=midi_note, sigma0=sigma0,
            max_evals=max_evals, timeout_sec=timeout_sec,
        )

        # Step 6: Rebuild params dict with refined continuous values
        for i, name in enumerate(self.continuous_names):
            params[name] = float(refined_vector[i])

        return params, confidence, info

    def render_comparison(
        self, params: dict[str, Any], midi_note: int = 60
    ) -> np.ndarray | None:
        """Render the predicted parameters through Vita.

        Args:
            params: Predicted parameter dict.
            midi_note: MIDI note to render.

        Returns:
            Stereo audio array (2, n_samples) or None if Vita unavailable.
        """
        try:
            from datagen.config import PipelineConfig
            from datagen.render.engine import RenderEngine

            config = PipelineConfig(sample_rate=self.sample_rate)
            engine = RenderEngine(config)
            # Inject wavetable catalog and keep modulation for RenderEngine
            render_params = dict(params)
            if self.wavetable_catalog is not None:
                render_params["_wavetable_catalog"] = self.wavetable_catalog
            # _modulation_t3 is already in params if model has mod head
            return engine.render_preset(render_params, midi_note=midi_note)
        except ImportError:
            log.warning("Vita not available; cannot render comparison audio")
            return None


def load_pipeline(checkpoint_path: str | Path, device: str = "cuda") -> InferencePipeline:
    """Convenience function to load an inference pipeline."""
    return InferencePipeline.from_checkpoint(checkpoint_path, device)
