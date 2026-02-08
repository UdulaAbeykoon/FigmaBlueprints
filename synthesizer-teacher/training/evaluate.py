"""Render-based evaluation: per-group MSE, categorical accuracy, spectral metrics, W&B audio."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def compute_per_group_mse(
    pred: np.ndarray,
    target: np.ndarray,
    continuous_names: list[str],
) -> dict[str, float]:
    """Compute MSE grouped by module prefix (e.g. osc_1, filter_1, env_1).

    Args:
        pred: (B, n_continuous) predicted values.
        target: (B, n_continuous) target values.
        continuous_names: List of parameter names aligned with columns.

    Returns:
        Dict mapping group name -> mean MSE over that group's params.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(continuous_names):
        # Group by first two underscore-separated tokens (e.g. "osc_1")
        parts = name.split("_")
        if len(parts) >= 2:
            prefix = f"{parts[0]}_{parts[1]}"
        else:
            prefix = parts[0]
        groups[prefix].append(i)

    result: dict[str, float] = {}
    sq_err = (pred - target) ** 2  # (B, n_cont)
    for group_name, indices in sorted(groups.items()):
        result[group_name] = float(sq_err[:, indices].mean())

    result["overall"] = float(sq_err.mean())
    return result


def compute_categorical_accuracy(
    logits_list: list[torch.Tensor],
    target: torch.Tensor,
    categorical_names: list[str],
) -> dict[str, float]:
    """Per-param and overall categorical accuracy.

    Args:
        logits_list: List of (B, n_options_i) tensors.
        target: (B, n_categorical) integer class indices.
        categorical_names: Names aligned with columns.

    Returns:
        Dict mapping param name -> accuracy, plus "overall".
    """
    result: dict[str, float] = {}
    total_correct = 0
    total_count = 0

    for i, logits in enumerate(logits_list):
        preds = logits.argmax(dim=1)
        targets_i = target[:, i]
        correct = (preds == targets_i).sum().item()
        count = len(targets_i)
        total_correct += correct
        total_count += count

        name = categorical_names[i] if i < len(categorical_names) else f"cat_{i}"
        result[name] = correct / max(count, 1)

    result["overall"] = total_correct / max(total_count, 1)
    return result


def compute_modulation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.05,
) -> dict[str, float]:
    """Compute modulation prediction metrics: precision, recall, amount MAE.

    Args:
        pred: (B, 4, n_src, n_dst) predicted modulation matrix.
        target: (B, 4, n_src, n_dst) target modulation matrix.
        threshold: Minimum |amount| to consider a connection active.

    Returns:
        Dict with 'precision', 'recall', 'amount_mae'.
    """
    target_active = (target[:, 0].abs() > 1e-6)  # (B, n_src, n_dst)
    pred_active = (pred[:, 0].abs() > threshold)

    tp = (pred_active & target_active).sum().float()
    fp = (pred_active & ~target_active).sum().float()
    fn = (~pred_active & target_active).sum().float()

    precision = float(tp / (tp + fp).clamp(min=1))
    recall = float(tp / (tp + fn).clamp(min=1))

    # Amount MAE on truly active connections
    if target_active.any():
        amount_mae = float(
            (pred[:, 0] - target[:, 0]).abs()[target_active].mean()
        )
    else:
        amount_mae = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "amount_mae": amount_mae,
    }


def compute_spectral_metrics(
    pred_audio: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: int = 44100,
) -> dict[str, float]:
    """Compute spectral distance metrics between predicted and target audio.

    Uses multi-resolution STFT loss from auraloss library.

    Args:
        pred_audio: (n_samples,) or (batch, n_samples) predicted audio.
        target_audio: Same shape as pred_audio.
        sample_rate: Audio sample rate.

    Returns:
        Dict with 'mrstft_loss', 'spectral_convergence', 'log_stft_magnitude'.
    """
    try:
        import auraloss
    except ImportError:
        log.warning("auraloss not installed; skipping spectral metrics")
        return {}

    # Ensure 3D: (batch, channels, samples)
    if pred_audio.ndim == 1:
        pred_audio = pred_audio[np.newaxis, np.newaxis, :]
        target_audio = target_audio[np.newaxis, np.newaxis, :]
    elif pred_audio.ndim == 2:
        pred_audio = pred_audio[:, np.newaxis, :]
        target_audio = target_audio[:, np.newaxis, :]

    pred_tensor = torch.from_numpy(pred_audio).float()
    target_tensor = torch.from_numpy(target_audio).float()

    # Multi-resolution STFT loss
    mrstft = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        scale="mel",
        n_bins=128,
        sample_rate=sample_rate,
        perceptual_weighting=True,
    )

    with torch.no_grad():
        loss = mrstft(pred_tensor, target_tensor)

    return {
        "mrstft_loss": float(loss.item()),
    }


def compute_batch_spectral_metrics(
    audio_pairs: list[tuple[np.ndarray | None, np.ndarray | None]],
    sample_rate: int = 44100,
) -> dict[str, float]:
    """Compute average spectral metrics over multiple audio pairs.

    Args:
        audio_pairs: List of (target_audio, predicted_audio) pairs.
        sample_rate: Audio sample rate.

    Returns:
        Dict with averaged spectral metrics.
    """
    valid_metrics: list[dict[str, float]] = []

    for target, predicted in audio_pairs:
        if target is None or predicted is None:
            continue
        # Ensure same length
        min_len = min(len(target), len(predicted))
        if min_len < 1024:  # Too short for meaningful spectral analysis
            continue
        metrics = compute_spectral_metrics(
            predicted[:min_len], target[:min_len], sample_rate
        )
        if metrics:
            valid_metrics.append(metrics)

    if not valid_metrics:
        return {}

    # Average over all pairs
    result: dict[str, float] = {}
    all_keys = valid_metrics[0].keys()
    for key in all_keys:
        values = [m[key] for m in valid_metrics if key in m]
        if values:
            result[key] = float(np.mean(values))

    return result


def render_eval_batch(
    model: torch.nn.Module,
    dataset: "Dataset",
    indices: list[int],
    continuous_names: list[str],
    categorical_names: list[str],
    categorical_n_options: list[int],
    device: torch.device,
    sample_rate: int,
) -> list[dict]:
    """Predict on a few samples and return info for rendering.

    Does NOT render via Vita (that requires a live synth). Instead returns
    predicted/target parameter vectors for the caller to render.

    Returns:
        List of dicts with keys:
            - "target_continuous": (n_cont,) float32
            - "pred_continuous": (n_cont,) float32
            - "target_categorical": (n_cat,) int32
            - "pred_categorical": (n_cat,) int32
            - "midi_note": int - the original MIDI note for rendering
            - "target_modulation": (4, n_src, n_dst) float32 or None
            - "pred_modulation": (4, n_src, n_dst) float32 or None
    """
    model.eval()
    results = []

    for idx in indices:
        sample = dataset[idx]
        # Support both 4-tuple (old) and 5-tuple (new with modulation)
        if len(sample) == 5:
            mel_spec, cont_target, cat_target, midi_note, mod_target = sample
        else:
            mel_spec, cont_target, cat_target, midi_note = sample
            mod_target = torch.empty(0)

        mel_batch = mel_spec.unsqueeze(0).to(device)

        with torch.no_grad():
            model_out = model(mel_batch)
            # Support both 2-tuple (old) and 3-tuple (new with modulation)
            if len(model_out) == 3:
                cont_pred, cat_logits, mod_pred = model_out
            else:
                cont_pred, cat_logits = model_out
                mod_pred = None

        pred_cat = np.array(
            [logits.argmax(dim=1).item() for logits in cat_logits], dtype=np.int32,
        )

        result_dict: dict = {
            "target_continuous": cont_target.numpy(),
            "pred_continuous": cont_pred.cpu().squeeze(0).numpy(),
            "target_categorical": cat_target.numpy().astype(np.int32),
            "pred_categorical": pred_cat,
            "midi_note": midi_note,
            "target_modulation": mod_target.numpy() if mod_target.numel() > 0 else None,
            "pred_modulation": mod_pred.cpu().squeeze(0).numpy() if mod_pred is not None else None,
        }

        results.append(result_dict)

    return results


def log_audio_to_wandb(
    audio_pairs: list[tuple[np.ndarray | None, np.ndarray | None]],
    sample_rate: int,
    epoch: int,
    midi_notes: list[int] | None = None,
) -> dict:
    """Create W&B audio log entries from (target, predicted) audio pairs.

    Args:
        audio_pairs: List of (target_audio, predicted_audio). Each is
            mono float32 or None if rendering failed.
        sample_rate: Audio sample rate.
        epoch: Current epoch (for caption).
        midi_notes: Optional list of MIDI notes for each pair.

    Returns:
        Dict suitable for ``wandb.log()``.
    """
    import wandb

    log_dict: dict = {}
    for i, (target, predicted) in enumerate(audio_pairs):
        note_suffix = f"_n{midi_notes[i]}" if midi_notes and i < len(midi_notes) else ""
        if target is not None:
            log_dict[f"audio/target_{i}"] = wandb.Audio(
                target, sample_rate=sample_rate, caption=f"target_{i}{note_suffix}_ep{epoch}",
            )
        if predicted is not None:
            log_dict[f"audio/predicted_{i}"] = wandb.Audio(
                predicted, sample_rate=sample_rate, caption=f"pred_{i}{note_suffix}_ep{epoch}",
            )
    return log_dict
