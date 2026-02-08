"""Training loop with W&B logging, checkpointing, early stopping, and render-based eval."""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from training.config import TrainConfig
from training.dataset import (
    FEATURES_KEY,
    MOD_FEATURES_KEY,
    MOD_PARAMS_KEY,
    VitalDataset,
    make_train_val_split,
)
from training.distributed import (
    barrier,
    cleanup_distributed,
    get_local_rank,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
)
from training.evaluate import (
    compute_batch_spectral_metrics,
    compute_categorical_accuracy,
    compute_per_group_mse,
    log_audio_to_wandb,
    render_eval_batch,
)
from training.loss import VitalLoss
from training.model import VitalInverseModel

log = logging.getLogger(__name__)


def _read_schema(path: Path) -> dict:
    """Read schema attributes from HDF5 file."""
    with h5py.File(path, "r") as f:
        schema = {}
        for key, val in f["schema"].attrs.items():
            if isinstance(val, np.ndarray):
                val = [
                    v.decode("utf-8") if isinstance(v, bytes) else v
                    for v in val
                ]
            elif isinstance(val, bytes):
                val = val.decode("utf-8")
            schema[key] = val
    return schema


def _read_importance_weights(path: Path) -> np.ndarray | None:
    """Read importance weights from HDF5 if present."""
    with h5py.File(path, "r") as f:
        if "importance_weights/weights" in f:
            return f["importance_weights/weights"][:]
    return None


def _build_lr_lambda(warmup_epochs: int, total_epochs: int):
    """Linear warmup then cosine decay to ~0."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.

        Returns True if training should stop.
        """
        if self.patience <= 0:
            return False

        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            return True

        return False


class Trainer:
    """Manages the full training loop."""

    def __init__(self, config: TrainConfig, verbose: bool = False) -> None:
        self.config = config
        self.verbose = verbose

        # --- DDP setup ---
        self.ddp = setup_distributed()
        self.rank = get_rank()
        self.world_size = get_world_size()
        if self.ddp:
            self.device = torch.device(f"cuda:{get_local_rank()}")
        else:
            self.device = torch.device(config.device)

        # --- Validate precomputed mels exist ---
        with h5py.File(config.dataset_path, "r") as f:
            if FEATURES_KEY not in f:
                raise RuntimeError(
                    f"{FEATURES_KEY} not found in {config.dataset_path}. "
                    "Run: python -m training precompute-mels -d <dataset>"
                )

        # --- Read schema ---
        schema = _read_schema(config.dataset_path)
        self.continuous_names: list[str] = schema["continuous_names"]
        self.n_continuous = len(self.continuous_names)

        cat_names = schema.get("categorical_names", [])
        wt_names = schema.get("wavetable_names", [])
        self.categorical_names: list[str] = list(cat_names) + list(wt_names)
        self.n_categorical = len(self.categorical_names)

        raw_n_options = schema.get("categorical_n_options", [])
        self.categorical_n_options: list[int] = [int(x) for x in raw_n_options]

        if not self.categorical_n_options and self.n_categorical > 0:
            log.warning(
                "categorical_n_options not in schema; inferring from data max values"
            )
            with h5py.File(config.dataset_path, "r") as f:
                cat_data = f["params/categorical"][:]
                self.categorical_n_options = [
                    int(cat_data[:, i].max()) + 1 for i in range(self.n_categorical)
                ]

        # Validate n_options against actual data to prevent cross_entropy crashes
        if self.categorical_n_options and self.n_categorical > 0:
            with h5py.File(config.dataset_path, "r") as f:
                cat_data = f["params/categorical"][:]
                data_max = [int(cat_data[:, i].max()) for i in range(self.n_categorical)]
                mismatches = []
                for i in range(self.n_categorical):
                    if data_max[i] >= self.categorical_n_options[i]:
                        mismatches.append(
                            f"  {self.categorical_names[i]}: schema n_options="
                            f"{self.categorical_n_options[i]}, data max={data_max[i]}"
                        )
                        self.categorical_n_options[i] = data_max[i] + 1
                if mismatches and is_main_process():
                    log.warning(
                        "Categorical n_options too small for %d params "
                        "(expanded to fit data):\n%s",
                        len(mismatches), "\n".join(mismatches),
                    )

        self.sample_rate = int(schema["sample_rate"])

        # --- Tier detection ---
        if config.tier > 0:
            self.tier = config.tier
        else:
            self.tier = int(schema.get("tier", 1))
        if is_main_process():
            log.info("Training tier: %d", self.tier)

        # --- Wavetable catalog ---
        self.wavetable_catalog = None
        self.wavetable_catalog_json: str | None = None
        wt_catalog_path = config.wavetable_catalog
        if not wt_catalog_path:
            # Auto-detect from dataset's schema attrs
            wt_catalog_path = schema.get("wavetable_catalog_path", "")
        if wt_catalog_path and Path(wt_catalog_path).exists():
            try:
                from datagen.wavetables.catalog import WavetableCatalog
                self.wavetable_catalog = WavetableCatalog.from_json(Path(wt_catalog_path))
                self.wavetable_catalog_json = self.wavetable_catalog.to_json_string()
                if is_main_process():
                    log.info(
                        "Loaded wavetable catalog: %d wavetables from %s",
                        len(self.wavetable_catalog), wt_catalog_path,
                    )
            except Exception as e:
                if is_main_process():
                    log.warning("Failed to load wavetable catalog: %s", e)

        # --- Modulation dimensions (tier-3) ---
        self.n_mod_sources = 0
        self.n_mod_destinations = 0
        self.mod_source_names: list[str] = []
        self.mod_destination_names: list[str] = []
        with h5py.File(config.dataset_path, "r") as f:
            mod_key = MOD_FEATURES_KEY if MOD_FEATURES_KEY in f else MOD_PARAMS_KEY
            if mod_key in f:
                mod_shape = f[mod_key].shape  # (N, 4, n_src, n_dst)
                self.n_mod_sources = mod_shape[2]
                self.n_mod_destinations = mod_shape[3]
                self.mod_source_names = list(
                    schema.get("mod_source_names", [f"src_{i}" for i in range(self.n_mod_sources)])
                )
                self.mod_destination_names = list(
                    schema.get("mod_destination_names", [f"dst_{i}" for i in range(self.n_mod_destinations)])
                )
                if is_main_process():
                    log.info(
                        "Modulation matrix detected: %d sources x %d destinations",
                        self.n_mod_sources, self.n_mod_destinations,
                    )

        # --- Importance weights ---
        iw = _read_importance_weights(config.dataset_path)
        if iw is not None and iw.shape[0] == self.n_continuous:
            self.importance_weights = torch.from_numpy(iw)
            log.info("Loaded importance weights: shape %s", iw.shape)
        else:
            self.importance_weights = None
            if iw is not None:
                log.warning(
                    "Importance weights shape %s doesn't match n_continuous=%d. "
                    "Recompute: python -m datagen compute-weights -o %s",
                    iw.shape, self.n_continuous, config.dataset_path,
                )
            else:
                log.warning(
                    "No importance weights found in dataset. Using uniform weights. "
                    "For better training, run: python -m datagen compute-weights -o %s",
                    config.dataset_path,
                )

        # --- Split + datasets ---
        train_idx, val_idx = make_train_val_split(
            config.dataset_path, config.val_fraction, seed=42,
        )
        self.train_dataset = VitalDataset(
            config.dataset_path, train_idx,
            training=True,
            spec_aug_freq_mask=config.spec_aug_freq_mask,
            spec_aug_time_mask=config.spec_aug_time_mask,
            spec_aug_n_masks=config.spec_aug_n_masks,
        )
        self.val_dataset = VitalDataset(config.dataset_path, val_idx)

        use_pin = self.device.type == "cuda"

        if self.ddp:
            self.train_sampler = DistributedSampler(
                self.train_dataset, shuffle=True, drop_last=True,
            )
            self.val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False, drop_last=True,
            )
        else:
            self.train_sampler = None
            self.val_sampler = None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
            pin_memory=use_pin,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
            pin_memory=use_pin,
        )

        # --- Model ---
        self.model = VitalInverseModel(
            n_continuous=self.n_continuous,
            categorical_n_options=self.categorical_n_options,
            mlp_hidden=config.mlp_hidden,
            dropout=config.dropout,
            freeze_early=config.freeze_early,
            simple_categorical_heads=config.simple_categorical_heads,
            n_mod_sources=self.n_mod_sources,
            n_mod_destinations=self.n_mod_destinations,
        ).to(self.device)

        if self.ddp:
            local_rank = get_local_rank()
            self.model = DDP(self.model, device_ids=[local_rank])

        # --- Loss ---
        mod_loss_weight = config.modulation_loss_weight if self.n_mod_sources > 0 else 0.0
        self.criterion = VitalLoss(
            importance_weights=self.importance_weights,
            n_continuous=self.n_continuous,
            continuous_weight=config.continuous_loss_weight,
            categorical_weight=config.categorical_loss_weight,
            label_smoothing=config.categorical_label_smoothing,
            continuous_names=self.continuous_names,
            categorical_names=self.categorical_names,
            conditional_mask=config.conditional_loss_mask,
            tier=self.tier,
            modulation_loss_weight=mod_loss_weight,
            modulation_pos_weight=config.modulation_pos_weight,
        ).to(self.device)

        # --- Optimizer (only trainable params) ---
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=config.lr, weight_decay=config.weight_decay,
        )

        # --- LR scheduler ---
        # Use last_epoch=-1 (default) and step once so epoch 0 gets the
        # correct warmed-up LR instead of training at full LR.
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, _build_lr_lambda(config.warmup_epochs, config.epochs),
        )
        self.scheduler.step()  # Apply warmup multiplier for epoch 0

        # --- Early stopping ---
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )

        self.best_val_loss = float("inf")
        self.start_epoch = 0
        self.checkpoint_dir = Path("checkpoints")
        if is_main_process():
            self.checkpoint_dir.mkdir(exist_ok=True)

        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in self.model.parameters())
        if is_main_process():
            log.info(
                "Model: %d total params, %d trainable (%.1f%%)",
                n_total, n_trainable, 100 * n_trainable / n_total,
            )
            log.info(
                "Data: %d train, %d val | Cont: %d, Cat: %d",
                len(self.train_dataset), len(self.val_dataset),
                self.n_continuous, self.n_categorical,
            )
            if self.ddp:
                log.info("DDP enabled: %d GPUs", self.world_size)
            if config.early_stopping_patience > 0:
                log.info("Early stopping enabled with patience=%d", config.early_stopping_patience)
            if config.gradient_accumulation_steps > 1:
                log.info("Gradient accumulation: %d steps", config.gradient_accumulation_steps)

    @property
    def _unwrapped_model(self) -> VitalInverseModel:
        """Return the underlying model, unwrapping DDP if necessary."""
        return self.model.module if self.ddp else self.model

    def resume_from(self, checkpoint_path: Path) -> None:
        """Load model, optimizer, scheduler, and epoch from a checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self._unwrapped_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.start_epoch = ckpt["epoch"] + 1
        if is_main_process():
            log.info(
                "Resumed from %s (epoch %d, best_val_loss=%.4f)",
                checkpoint_path, ckpt["epoch"], self.best_val_loss,
            )

    def train(self) -> None:
        """Run the full training loop."""
        import wandb

        if is_main_process():
            wandb.init(
                project=self.config.wandb_project,
                config={
                    "n_continuous": self.n_continuous,
                    "n_categorical": self.n_categorical,
                    "categorical_n_options": self.categorical_n_options,
                    "sample_rate": self.sample_rate,
                    **{
                        k: v for k, v in self.config.__dict__.items()
                        if not isinstance(v, Path)
                    },
                    "dataset_path": str(self.config.dataset_path),
                    "world_size": self.world_size,
                },
            )

        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                # Set epoch on distributed samplers for proper shuffling
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)
                if self.val_sampler is not None:
                    self.val_sampler.set_epoch(epoch)

                train_metrics = self._train_epoch(epoch)
                val_metrics = self._validate_epoch(epoch)

                lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()

                if is_main_process():
                    log_dict = {
                        "epoch": epoch,
                        "lr": lr,
                        "train/loss": train_metrics["loss"],
                        "train/cont_loss": train_metrics["cont_loss"],
                        "train/cat_loss": train_metrics["cat_loss"],
                        "val/loss": val_metrics["loss"],
                        "val/cont_loss": val_metrics["cont_loss"],
                        "val/cat_loss": val_metrics["cat_loss"],
                    }
                    if "mod_loss" in train_metrics:
                        log_dict["train/mod_loss"] = train_metrics["mod_loss"]
                    if "mod_loss" in val_metrics:
                        log_dict["val/mod_loss"] = val_metrics["mod_loss"]

                    for group, mse in val_metrics.get("group_mse", {}).items():
                        log_dict[f"val/mse_{group}"] = mse
                    for name, acc in val_metrics.get("cat_accuracy", {}).items():
                        log_dict[f"val/acc_{name}"] = acc

                    # Modulation metrics
                    for key, value in val_metrics.get("mod_metrics", {}).items():
                        log_dict[f"val/mod_{key}"] = value

                    # Spectral metrics
                    for key, value in val_metrics.get("spectral_metrics", {}).items():
                        log_dict[f"val/{key}"] = value

                    is_best = val_metrics["loss"] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics["loss"]
                        self._save_checkpoint(epoch, is_best=True)
                        log_dict["best_val_loss"] = self.best_val_loss

                    if (epoch + 1) % self.config.log_audio_every == 0:
                        audio_log = self._audio_eval(epoch)
                        log_dict.update(audio_log)

                    wandb.log(log_dict)

                    if self.verbose:
                        log.info(
                            "Epoch %d: train=%.4f val=%.4f (cont=%.4f cat=%.4f) lr=%.2e",
                            epoch,
                            train_metrics["loss"],
                            val_metrics["loss"],
                            val_metrics["cont_loss"],
                            val_metrics["cat_loss"],
                            lr,
                        )

                # Early stopping: rank 0 decides, broadcast to all ranks
                should_stop = False
                if is_main_process():
                    should_stop = self.early_stopping(val_metrics["loss"])
                    if should_stop:
                        log.info(
                            "Early stopping triggered at epoch %d (patience=%d)",
                            epoch, self.config.early_stopping_patience,
                        )

                if self.ddp:
                    stop_tensor = torch.tensor(
                        int(should_stop), device=self.device,
                    )
                    dist.broadcast(stop_tensor, src=0)
                    should_stop = bool(stop_tensor.item())

                if should_stop:
                    break
        finally:
            if is_main_process():
                wandb.finish()
            cleanup_distributed()

        if is_main_process():
            log.info("Training complete. Best val loss: %.4f", self.best_val_loss)

    def _train_epoch(self, epoch: int = 0) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_cont = 0.0
        total_cat = 0.0
        total_mod = 0.0
        n_batches = 0

        accum_steps = self.config.gradient_accumulation_steps
        self.optimizer.zero_grad()

        n_total_batches = len(self.train_loader)
        residual = n_total_batches % accum_steps
        residual_start = n_total_batches - residual if residual > 0 else n_total_batches

        for batch_idx, (mel_spec, cont_target, cat_target, _midi_note, mod_target) in enumerate(self.train_loader):
            mel_spec = mel_spec.to(self.device)
            cont_target = cont_target.to(self.device)
            cat_target = cat_target.to(self.device)
            mod_target = mod_target.to(self.device) if mod_target.numel() > 0 else None

            # Suppress all-reduce during accumulation steps (DDP only)
            is_sync_step = (batch_idx + 1) % accum_steps == 0
            is_last_batch = (batch_idx + 1) == n_total_batches
            sync_ctx = (
                nullcontext()
                if (is_sync_step or is_last_batch or not self.ddp)
                else self.model.no_sync()
            )

            # Scale by actual window size (handles residual batches at end)
            if residual > 0 and batch_idx >= residual_start:
                effective_accum = residual
            else:
                effective_accum = accum_steps

            with sync_ctx:
                cont_pred, cat_logits, mod_pred = self.model(mel_spec)
                loss, cont_loss, cat_loss, mod_loss = self.criterion(
                    cont_pred, cont_target, cat_logits, cat_target,
                    modulation_pred=mod_pred,
                    modulation_target=mod_target,
                    current_epoch=epoch,
                )

                # Scale loss for gradient accumulation
                scaled_loss = loss / effective_accum
                scaled_loss.backward()

            # Step optimizer every accum_steps batches or at end
            if is_sync_step or is_last_batch:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Track unscaled loss.item() — same magnitude as validation batches
            # since both use the same batch_size. scaled_loss is only for gradients.
            total_loss += loss.item()
            total_cont += cont_loss.item()
            total_cat += cat_loss.item()
            total_mod += mod_loss.item()
            n_batches += 1

        metrics = {
            "loss": total_loss / max(n_batches, 1),
            "cont_loss": total_cont / max(n_batches, 1),
            "cat_loss": total_cat / max(n_batches, 1),
        }
        if self.n_mod_sources > 0:
            metrics["mod_loss"] = total_mod / max(n_batches, 1)
        return metrics

    @torch.no_grad()
    def _validate_epoch(self, epoch: int = 0) -> dict:
        # Use unwrapped model for inference to skip DDP overhead
        eval_model = self._unwrapped_model
        eval_model.eval()

        total_loss = 0.0
        total_cont = 0.0
        total_cat = 0.0
        total_mod = 0.0
        n_batches = 0

        all_cont_pred = []
        all_cont_target = []
        all_cat_logits: list[list[torch.Tensor]] = [
            [] for _ in range(self.n_categorical)
        ]
        all_cat_target = []
        all_mod_pred = []
        all_mod_target = []

        for mel_spec, cont_target, cat_target, _midi_note, mod_target in self.val_loader:
            mel_spec = mel_spec.to(self.device)
            cont_target = cont_target.to(self.device)
            cat_target = cat_target.to(self.device)
            mod_target_dev = mod_target.to(self.device) if mod_target.numel() > 0 else None

            cont_pred, cat_logits, mod_pred = eval_model(mel_spec)
            loss, cont_loss, cat_loss, mod_loss = self.criterion(
                cont_pred, cont_target, cat_logits, cat_target,
                modulation_pred=mod_pred,
                modulation_target=mod_target_dev,
                current_epoch=epoch,
            )

            total_loss += loss.item()
            total_cont += cont_loss.item()
            total_cat += cat_loss.item()
            total_mod += mod_loss.item()
            n_batches += 1

            all_cont_pred.append(cont_pred.cpu())
            all_cont_target.append(cont_target.cpu())
            for i, logits in enumerate(cat_logits):
                all_cat_logits[i].append(logits.cpu())
            all_cat_target.append(cat_target.cpu())
            if mod_pred is not None:
                all_mod_pred.append(mod_pred.cpu())
            if mod_target.numel() > 0:
                all_mod_target.append(mod_target)

        # All-reduce scalar losses across ranks for globally accurate metrics
        if self.ddp:
            stats = torch.tensor(
                [total_loss, total_cont, total_cat, total_mod, n_batches],
                device=self.device, dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss, total_cont, total_cat, total_mod = (
                stats[0].item(), stats[1].item(), stats[2].item(), stats[3].item(),
            )
            n_batches = int(stats[4].item())

        metrics: dict = {
            "loss": total_loss / max(n_batches, 1),
            "cont_loss": total_cont / max(n_batches, 1),
            "cat_loss": total_cat / max(n_batches, 1),
        }
        if self.n_mod_sources > 0:
            metrics["mod_loss"] = total_mod / max(n_batches, 1)

        # Per-group MSE: all ranks compute on their local shard, then all-reduce
        if all_cont_pred:
            pred_np = torch.cat(all_cont_pred).numpy()
            target_np = torch.cat(all_cont_target).numpy()
            local_group_mse = compute_per_group_mse(
                pred_np, target_np, self.continuous_names,
            )

            if self.ddp:
                # Pack into tensor, all-reduce (average), unpack
                group_keys = sorted(local_group_mse.keys())
                mse_tensor = torch.tensor(
                    [local_group_mse[k] for k in group_keys],
                    device=self.device, dtype=torch.float64,
                )
                dist.all_reduce(mse_tensor, op=dist.ReduceOp.SUM)
                mse_tensor /= self.world_size
                if is_main_process():
                    metrics["group_mse"] = {
                        k: float(mse_tensor[i])
                        for i, k in enumerate(group_keys)
                    }
            else:
                metrics["group_mse"] = local_group_mse

        # Categorical accuracy: all ranks compute correct/total, then all-reduce
        if all_cat_target and self.n_categorical > 0:
            merged_logits = [torch.cat(ll) for ll in all_cat_logits]
            merged_targets = torch.cat(all_cat_target)

            if self.ddp:
                # Compute per-param correct counts and totals
                n_params = len(merged_logits)
                correct_counts = torch.zeros(n_params + 1, device=self.device, dtype=torch.float64)
                total_counts = torch.zeros(n_params + 1, device=self.device, dtype=torch.float64)
                overall_correct = 0
                overall_total = 0

                for i, logits in enumerate(merged_logits):
                    preds = logits.argmax(dim=1)
                    targets_i = merged_targets[:, i]
                    correct = (preds == targets_i).sum().item()
                    count = len(targets_i)
                    correct_counts[i] = correct
                    total_counts[i] = count
                    overall_correct += correct
                    overall_total += count

                correct_counts[-1] = overall_correct
                total_counts[-1] = overall_total

                dist.all_reduce(correct_counts, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_counts, op=dist.ReduceOp.SUM)

                if is_main_process():
                    cat_acc: dict[str, float] = {}
                    for i in range(n_params):
                        name = (
                            self.categorical_names[i]
                            if i < len(self.categorical_names)
                            else f"cat_{i}"
                        )
                        cat_acc[name] = float(
                            correct_counts[i] / max(total_counts[i], 1)
                        )
                    cat_acc["overall"] = float(
                        correct_counts[-1] / max(total_counts[-1], 1)
                    )
                    metrics["cat_accuracy"] = cat_acc
            else:
                metrics["cat_accuracy"] = compute_categorical_accuracy(
                    merged_logits, merged_targets, self.categorical_names,
                )

        # Modulation metrics: precision/recall/amount MAE
        if all_mod_pred and all_mod_target:
            from training.evaluate import compute_modulation_metrics
            mod_pred_cat = torch.cat(all_mod_pred)
            mod_tgt_cat = torch.cat(all_mod_target)
            mod_metrics = compute_modulation_metrics(mod_pred_cat, mod_tgt_cat)
            metrics["mod_metrics"] = mod_metrics

        return metrics

    def _audio_eval(self, epoch: int) -> dict:
        """Run render-based eval on a few validation samples with spectral metrics."""
        n = min(self.config.n_render_eval, len(self.val_dataset))
        if n == 0:
            return {}

        rng = np.random.RandomState(epoch)
        indices = rng.choice(len(self.val_dataset), size=n, replace=False).tolist()

        eval_results = render_eval_batch(
            self._unwrapped_model,
            self.val_dataset,
            indices,
            self.continuous_names,
            self.categorical_names,
            self.categorical_n_options,
            self.device,
            self.sample_rate,
        )

        try:
            audio_pairs, spectral_log = self._render_and_log_audio(eval_results, epoch)

            # Compute spectral metrics if enabled
            if self.config.compute_spectral_metrics:
                spectral_metrics = compute_batch_spectral_metrics(
                    audio_pairs, self.sample_rate
                )
                for key, value in spectral_metrics.items():
                    spectral_log[f"spectral/{key}"] = value

            return spectral_log
        except ImportError:
            log.debug("Vita not available; skipping audio rendering")
            return {}

    def _render_and_log_audio(
        self, eval_results: list[dict], epoch: int,
    ) -> tuple[list[tuple[np.ndarray | None, np.ndarray | None]], dict]:
        """Render predicted params via Vita and log audio to W&B.

        Continuous values are passed as [0,1] normalized — RenderEngine
        denormalizes internally via ``_apply_params``.

        Returns:
            Tuple of (audio_pairs, log_dict)
        """
        from datagen.config import PipelineConfig
        from datagen.render.engine import RenderEngine

        pipe_config = PipelineConfig(sample_rate=self.sample_rate)
        engine = RenderEngine(pipe_config)

        audio_pairs: list[tuple[np.ndarray | None, np.ndarray | None]] = []
        for result in eval_results:
            midi_note = result.get("midi_note", 60)  # Fallback to 60 for compatibility
            
            target_preset: dict = {
                name: float(result["target_continuous"][i])
                for i, name in enumerate(self.continuous_names)
            }
            for i, name in enumerate(self.categorical_names):
                target_preset[name] = int(result["target_categorical"][i])
            if self.wavetable_catalog is not None:
                target_preset["_wavetable_catalog"] = self.wavetable_catalog
            if "target_modulation" in result and result["target_modulation"] is not None:
                target_preset["_modulation_t3"] = self._dense_to_connections(
                    result["target_modulation"],
                )
            target_audio = engine.render_preset(target_preset, midi_note=midi_note)
            target_mono = (
                target_audio.mean(axis=0) if target_audio is not None else None
            )

            pred_preset: dict = {
                name: float(result["pred_continuous"][i])
                for i, name in enumerate(self.continuous_names)
            }
            for i, name in enumerate(self.categorical_names):
                pred_preset[name] = int(result["pred_categorical"][i])
            if self.wavetable_catalog is not None:
                pred_preset["_wavetable_catalog"] = self.wavetable_catalog
            if "pred_modulation" in result and result["pred_modulation"] is not None:
                pred_preset["_modulation_t3"] = self._dense_to_connections(
                    result["pred_modulation"],
                )
            # Use same MIDI note as target for fair comparison
            pred_audio = engine.render_preset(pred_preset, midi_note=midi_note)
            pred_mono = pred_audio.mean(axis=0) if pred_audio is not None else None

            audio_pairs.append((target_mono, pred_mono))

        # Collect MIDI notes for captions
        midi_notes = [result.get("midi_note", 60) for result in eval_results]
        log_dict = log_audio_to_wandb(audio_pairs, self.sample_rate, epoch, midi_notes)
        return audio_pairs, log_dict

    def _dense_to_connections(
        self, mod_matrix: np.ndarray, threshold: float = 0.05,
    ) -> dict:
        """Convert dense (4, n_src, n_dst) matrix to connections dict for RenderEngine."""
        connections = []
        amount = mod_matrix[0]  # (n_src, n_dst)
        for si in range(amount.shape[0]):
            for di in range(amount.shape[1]):
                if abs(amount[si, di]) > threshold:
                    conn = {
                        "source": self.mod_source_names[si] if si < len(self.mod_source_names) else f"src_{si}",
                        "destination": self.mod_destination_names[di] if di < len(self.mod_destination_names) else f"dst_{di}",
                        "amount": float(mod_matrix[0, si, di]),
                        "bipolar": float(mod_matrix[1, si, di]),
                        "power": float(mod_matrix[2, si, di]),
                        "stereo": float(mod_matrix[3, si, di]),
                    }
                    connections.append(conn)
        return {"connections": connections}

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        # Build continuous param ranges for preset export denormalization
        continuous_ranges: list[tuple[float, float]] = []
        try:
            with h5py.File(self.config.dataset_path, "r") as f:
                if "schema" in f:
                    mins = f["schema"].attrs.get("continuous_min", [])
                    maxs = f["schema"].attrs.get("continuous_max", [])
                    if len(mins) == self.n_continuous and len(maxs) == self.n_continuous:
                        continuous_ranges = [
                            (float(lo), float(hi)) for lo, hi in zip(mins, maxs)
                        ]
        except Exception:
            pass  # Ranges are optional; inference falls back gracefully

        state = {
            "epoch": epoch,
            "model_state_dict": self._unwrapped_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
            "continuous_names": self.continuous_names,
            "categorical_names": self.categorical_names,
            "categorical_n_options": self.categorical_n_options,
            "n_continuous": self.n_continuous,
            "n_categorical": self.n_categorical,
            "sample_rate": self.sample_rate,
            "continuous_ranges": continuous_ranges,
            "tier": self.tier,
            "wavetable_catalog_json": self.wavetable_catalog_json,
            "n_mod_sources": self.n_mod_sources,
            "n_mod_destinations": self.n_mod_destinations,
            "mod_source_names": self.mod_source_names,
            "mod_destination_names": self.mod_destination_names,
        }
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, path)
            log.info("Saved best checkpoint (epoch %d, val_loss=%.4f)", epoch, self.best_val_loss)

        # Always save latest checkpoint for crash recovery
        latest_path = self.checkpoint_dir / "latest_model.pt"
        torch.save(state, latest_path)
