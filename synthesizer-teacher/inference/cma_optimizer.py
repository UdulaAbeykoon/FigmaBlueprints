"""CMA-ES inference-time optimization for synthesizer parameter refinement."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


class CMAESOptimizer:
    """Derivative-free optimization of synth params via CMA-ES + spectral loss."""

    def __init__(
        self,
        engine: Any,
        continuous_names: list[str],
        sample_rate: int,
    ) -> None:
        self.engine = engine
        self.continuous_names = continuous_names
        self.sample_rate = sample_rate
        self._mrstft: Any = None  # Lazy-init, cached

    def _get_mrstft(self) -> Any:
        """Lazily create and cache the multi-resolution STFT loss."""
        if self._mrstft is None:
            import auraloss.freq

            self._mrstft = auraloss.freq.MultiResolutionSTFTLoss(
                fft_sizes=[1024, 2048, 8192],
                hop_sizes=[256, 512, 2048],
                win_lengths=[1024, 2048, 8192],
                scale="mel",
                n_bins=128,
                sample_rate=self.sample_rate,
                perceptual_weighting=True,
            )
        return self._mrstft

    def _objective(
        self,
        x: np.ndarray,
        categorical_params: dict[str, Any],
        target_mono: np.ndarray,
        midi_note: int,
    ) -> float:
        """Render candidate parameters and compute spectral distance to target."""
        x_clipped = np.clip(x, 0.0, 1.0)
        preset: dict[str, Any] = {
            name: float(v) for name, v in zip(self.continuous_names, x_clipped)
        }
        preset.update(categorical_params)

        audio = self.engine.render_preset(preset, midi_note=midi_note)
        if audio is None:
            return 1e6

        rendered_mono = audio.mean(axis=0)  # (2, N) -> (N,)
        min_len = min(len(rendered_mono), len(target_mono))
        if min_len < 1024:
            return 1e6

        pred_t = torch.from_numpy(rendered_mono[:min_len]).float().unsqueeze(0).unsqueeze(0)
        tgt_t = torch.from_numpy(target_mono[:min_len]).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            loss = self._get_mrstft()(pred_t, tgt_t)
        return float(loss.item())

    def optimize(
        self,
        initial_continuous: np.ndarray,
        categorical_params: dict[str, Any],
        target_mono: np.ndarray,
        midi_note: int = 60,
        sigma0: float = 0.1,
        max_evals: int = 500,
        timeout_sec: float = 60.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run CMA-ES optimization.

        Args:
            initial_continuous: Initial continuous parameter vector in [0, 1].
            categorical_params: Fixed categorical parameters (not optimized).
                May also contain '_wavetable_catalog' and '_modulation_t3'
                which are passed through to RenderEngine.
            target_mono: Target mono audio as numpy array.
            midi_note: MIDI note for rendering.
            sigma0: Initial step size.
            max_evals: Maximum function evaluations.
            timeout_sec: Timeout in seconds.

        Returns:
            (optimized_continuous, info_dict) where info_dict contains
            initial_loss, final_loss, n_evals, elapsed_sec, improved.
        """
        import cma

        start = time.monotonic()
        initial_loss = self._objective(
            initial_continuous, categorical_params, target_mono, midi_note,
        )
        log.info("CMA-ES initial loss: %.4f", initial_loss)

        opts = cma.CMAOptions()
        opts["bounds"] = [0.0, 1.0]
        opts["maxfevals"] = max_evals
        opts["timeout"] = timeout_sec
        opts["verbose"] = -9  # silent

        es = cma.CMAEvolutionStrategy(initial_continuous.tolist(), sigma0, opts)
        best_loss = initial_loss
        n_evals = 0

        while not es.stop():
            solutions = es.ask()
            losses = [
                self._objective(np.array(s), categorical_params, target_mono, midi_note)
                for s in solutions
            ]
            es.tell(solutions, losses)
            n_evals += len(solutions)
            gen_best = min(losses)
            if gen_best < best_loss:
                best_loss = gen_best

        result = np.clip(np.array(es.result.xbest), 0.0, 1.0)
        elapsed = time.monotonic() - start

        log.info(
            "CMA-ES done: %.4f -> %.4f (%d evals, %.1fs)",
            initial_loss, best_loss, n_evals, elapsed,
        )

        return result, {
            "initial_loss": initial_loss,
            "final_loss": best_loss,
            "n_evals": n_evals,
            "elapsed_sec": elapsed,
            "improved": best_loss < initial_loss,
        }
