"""Click CLI for training and evaluation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import click

from training.config import TrainConfig


@click.group()
def main() -> None:
    """Vital inverse synthesis training pipeline."""


# ------------------------------------------------------------------
# precompute-mels
# ------------------------------------------------------------------


@main.command("precompute-mels")
@click.option("-d", "--dataset", "dataset_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--n-mels", type=int, default=128)
@click.option("--n-fft", type=int, default=2048)
@click.option("--hop-length", type=int, default=512)
@click.option("--device", type=str, default="cuda")
@click.option("-v", "--verbose", is_flag=True, default=False)
def precompute_mels_cmd(
    dataset_path: Path,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    device: str,
    verbose: bool,
) -> None:
    """Precompute mel spectrograms and store in HDF5 (uncompressed)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    import torch
    from training.dataset import precompute_mels

    precompute_mels(
        dataset_path,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        device=torch.device(device),
    )


# ------------------------------------------------------------------
# precompute-modulation
# ------------------------------------------------------------------


@main.command("precompute-modulation")
@click.option("-d", "--dataset", "dataset_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("-v", "--verbose", is_flag=True, default=False)
def precompute_modulation_cmd(
    dataset_path: Path,
    verbose: bool,
) -> None:
    """Copy compressed modulation_t3 to uncompressed HDF5 for fast training access."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    from training.dataset import precompute_modulation

    precompute_modulation(dataset_path)


# ------------------------------------------------------------------
# train
# ------------------------------------------------------------------


@main.command()
@click.option("-d", "--dataset", "dataset_path", type=click.Path(exists=True, path_type=Path), default=Path("data/tier1_20k.h5"))
@click.option("--val-fraction", type=float, default=0.15)
@click.option("--epochs", type=int, default=100)
@click.option("-b", "--batch-size", type=int, default=32)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight-decay", type=float, default=1e-4)
@click.option("--grad-clip", type=float, default=1.0)
@click.option("--warmup-epochs", type=int, default=5)
@click.option("--mlp-hidden", type=int, default=512)
@click.option("--dropout", type=float, default=0.1)
@click.option("--no-freeze", is_flag=True, default=False, help="Don't freeze early ResNet layers.")
@click.option("--cont-weight", "continuous_loss_weight", type=float, default=1.0)
@click.option("--cat-weight", "categorical_loss_weight", type=float, default=0.5)
@click.option("--label-smoothing", "categorical_label_smoothing", type=float, default=0.0, help="Label smoothing for categorical loss.")
@click.option("--n-mels", type=int, default=128)
@click.option("--n-fft", type=int, default=2048)
@click.option("--hop-length", type=int, default=512)
@click.option("--device", type=str, default="cuda")
@click.option("--wandb-project", type=str, default="vital-inverse-synthesis")
@click.option("--log-audio-every", type=int, default=5)
@click.option("--n-render-eval", type=int, default=8)
@click.option("--num-workers", type=int, default=4, help="DataLoader workers.")
@click.option("--early-stopping-patience", type=int, default=0, help="Early stopping patience (0=disabled).")
@click.option("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps.")
@click.option("--compute-spectral-metrics/--no-spectral-metrics", default=True, help="Compute spectral distance on validation.")
@click.option("--spec-aug-freq-mask", type=int, default=12, help="SpecAugment max freq bands to mask.")
@click.option("--spec-aug-time-mask", type=int, default=12, help="SpecAugment max time steps to mask.")
@click.option("--spec-aug-n-masks", type=int, default=2, help="SpecAugment number of each mask type.")
@click.option("--simple-cat-heads/--no-simple-cat-heads", default=True, help="Use simple linear categorical heads.")
@click.option("--conditional-loss-mask/--no-conditional-loss-mask", default=True, help="Mask loss for unlearnable params.")
@click.option("--tier", type=int, default=0, help="Training tier (0=auto-detect, 1/2/3 explicit).")
@click.option("--wavetable-catalog", type=str, default="", help="Path to wavetable_catalog.json.")
@click.option("--modulation-loss-weight", type=float, default=0.3, help="Modulation loss weight (tier-3).")
@click.option("--mod-pos-weight", "modulation_pos_weight", type=float, default=20.0, help="BCE pos_weight for modulation sparsity.")
@click.option("--resume", type=click.Path(exists=True, path_type=Path), default=None, help="Resume from checkpoint.")
@click.option("-v", "--verbose", is_flag=True, default=False)
def train(
    dataset_path: Path,
    val_fraction: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    warmup_epochs: int,
    mlp_hidden: int,
    dropout: float,
    no_freeze: bool,
    continuous_loss_weight: float,
    categorical_loss_weight: float,
    categorical_label_smoothing: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    device: str,
    wandb_project: str,
    log_audio_every: int,
    n_render_eval: int,
    num_workers: int,
    early_stopping_patience: int,
    gradient_accumulation_steps: int,
    compute_spectral_metrics: bool,
    spec_aug_freq_mask: int,
    spec_aug_time_mask: int,
    spec_aug_n_masks: int,
    simple_cat_heads: bool,
    conditional_loss_mask: bool,
    tier: int,
    wavetable_catalog: str,
    modulation_loss_weight: float,
    modulation_pos_weight: float,
    resume: Path | None,
    verbose: bool,
) -> None:
    """Train the inverse synthesis model."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Suppress duplicate log output from non-zero DDP ranks
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    config = TrainConfig(
        dataset_path=dataset_path,
        val_fraction=val_fraction,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        warmup_epochs=warmup_epochs,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
        freeze_early=not no_freeze,
        continuous_loss_weight=continuous_loss_weight,
        categorical_loss_weight=categorical_loss_weight,
        categorical_label_smoothing=categorical_label_smoothing,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        device=device,
        wandb_project=wandb_project,
        log_audio_every=log_audio_every,
        n_render_eval=n_render_eval,
        num_workers=num_workers,
        early_stopping_patience=early_stopping_patience,
        gradient_accumulation_steps=gradient_accumulation_steps,
        compute_spectral_metrics=compute_spectral_metrics,
        spec_aug_freq_mask=spec_aug_freq_mask,
        spec_aug_time_mask=spec_aug_time_mask,
        spec_aug_n_masks=spec_aug_n_masks,
        simple_categorical_heads=simple_cat_heads,
        conditional_loss_mask=conditional_loss_mask,
        tier=tier,
        wavetable_catalog=wavetable_catalog,
        modulation_loss_weight=modulation_loss_weight,
        modulation_pos_weight=modulation_pos_weight,
    )

    from training.trainer import Trainer

    trainer = Trainer(config, verbose=verbose)
    if resume is not None:
        trainer.resume_from(resume)
    trainer.train()


# ------------------------------------------------------------------
# eval-tui
# ------------------------------------------------------------------


@main.command("eval-tui")
@click.option("-c", "--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("-d", "--dataset", "dataset_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--device", type=str, default="mps")
def eval_tui(checkpoint: Path, dataset_path: Path, device: str) -> None:
    """Interactive TUI for browsing model predictions vs ground truth."""
    from training.eval_tui import EvalPreviewApp

    app = EvalPreviewApp(checkpoint, dataset_path, device=device)
    app.run()


# ------------------------------------------------------------------
# evaluate
# ------------------------------------------------------------------


@main.command()
@click.option("-c", "--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("-d", "--dataset", "dataset_path", type=click.Path(exists=True, path_type=Path), default=Path("data/tier1_20k.h5"))
@click.option("--n-samples", type=int, default=16)
@click.option("--device", type=str, default="cuda")
@click.option("--compute-spectral/--no-spectral", "compute_spectral", default=True, help="Compute spectral metrics.")
@click.option("-v", "--verbose", is_flag=True, default=False)
def evaluate(
    checkpoint: Path,
    dataset_path: Path,
    n_samples: int,
    device: str,
    compute_spectral: bool,
    verbose: bool,
) -> None:
    """Evaluate a trained checkpoint with render-based metrics."""
    import numpy as np
    import torch

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    continuous_names = ckpt["continuous_names"]
    categorical_names = ckpt["categorical_names"]
    categorical_n_options = ckpt["categorical_n_options"]
    n_continuous = ckpt["n_continuous"]
    n_categorical = ckpt["n_categorical"]
    ckpt_config = ckpt.get("config", {})

    from training.model import VitalInverseModel

    model = VitalInverseModel(
        n_continuous=n_continuous,
        categorical_n_options=categorical_n_options,
        mlp_hidden=ckpt_config.get("mlp_hidden", 512),
        dropout=0.0,
        freeze_early=False,
        simple_categorical_heads=ckpt_config.get("simple_categorical_heads", True),
        n_mod_sources=ckpt.get("n_mod_sources", 0),
        n_mod_destinations=ckpt.get("n_mod_destinations", 0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    from training.dataset import VitalDataset, make_train_val_split

    train_idx, val_idx = make_train_val_split(
        dataset_path, ckpt_config.get("val_fraction", 0.15),
    )
    val_dataset = VitalDataset(dataset_path, val_idx)

    from training.evaluate import (
        compute_batch_spectral_metrics,
        compute_categorical_accuracy,
        compute_per_group_mse,
        render_eval_batch,
    )

    n = min(n_samples, len(val_dataset))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(val_dataset), size=n, replace=False).tolist()

    sample_rate = ckpt_config.get("sample_rate", 44100)
    results = render_eval_batch(
        model, val_dataset, indices,
        continuous_names, categorical_names, categorical_n_options,
        torch.device(device), sample_rate,
    )

    pred_cont = np.stack([r["pred_continuous"] for r in results])
    target_cont = np.stack([r["target_continuous"] for r in results])
    group_mse = compute_per_group_mse(pred_cont, target_cont, continuous_names)

    log.info("Per-group MSE:")
    for group, mse in sorted(group_mse.items()):
        log.info("  %s: %.6f", group, mse)

    cat_logits_all: list[list[torch.Tensor]] = [[] for _ in range(n_categorical)]
    cat_targets_all = []
    dev = torch.device(device)
    for idx in indices:
        sample = val_dataset[idx]
        mel = sample[0]
        cat = sample[2]
        with torch.no_grad():
            model_out = model(mel.unsqueeze(0).to(dev))
            cat_logits = model_out[1]
        for i, logits in enumerate(cat_logits):
            cat_logits_all[i].append(logits.cpu())
        cat_targets_all.append(cat)

    if cat_targets_all:
        merged_logits = [torch.cat(ll) for ll in cat_logits_all]
        merged_targets = torch.stack(cat_targets_all)
        cat_acc = compute_categorical_accuracy(
            merged_logits, merged_targets, categorical_names,
        )
        log.info("Categorical accuracy: overall=%.4f", cat_acc.get("overall", 0.0))

    # Spectral metrics with Vita rendering
    if compute_spectral:
        try:
            from datagen.config import PipelineConfig
            from datagen.render.engine import RenderEngine

            pipe_config = PipelineConfig(sample_rate=sample_rate)
            engine = RenderEngine(pipe_config)

            audio_pairs = []
            for result in results:
                target_preset = {
                    name: float(result["target_continuous"][i])
                    for i, name in enumerate(continuous_names)
                }
                for i, name in enumerate(categorical_names):
                    target_preset[name] = int(result["target_categorical"][i])
                midi_note = int(result["midi_note"])
                target_audio = engine.render_preset(target_preset, midi_note=midi_note)
                target_mono = target_audio.mean(axis=0) if target_audio is not None else None

                pred_preset = {
                    name: float(result["pred_continuous"][i])
                    for i, name in enumerate(continuous_names)
                }
                for i, name in enumerate(categorical_names):
                    pred_preset[name] = int(result["pred_categorical"][i])
                pred_audio = engine.render_preset(pred_preset, midi_note=midi_note)
                pred_mono = pred_audio.mean(axis=0) if pred_audio is not None else None

                audio_pairs.append((target_mono, pred_mono))

            spectral_metrics = compute_batch_spectral_metrics(audio_pairs, sample_rate)
            if spectral_metrics:
                log.info("Spectral metrics:")
                for key, value in spectral_metrics.items():
                    log.info("  %s: %.6f", key, value)
        except ImportError:
            log.warning("Vita not available; skipping spectral metrics")

    log.info("Evaluation complete on %d samples.", n)


if __name__ == "__main__":
    main()
