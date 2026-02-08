"""TUI for evaluating model predictions against ground truth with audio playback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

try:
    import sounddevice as sd

    _HAS_AUDIO = True
except (ImportError, OSError):
    _HAS_AUDIO = False

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.timer import Timer
from textual.widgets import DataTable, Footer, Header, Static

from training.dataset import FEATURES_KEY, make_train_val_split

_BAR_W = 20


def _bar(v: float, w: int = _BAR_W) -> str:
    n = max(0, min(w, round(v * w)))
    return "\u2588" * n + "\u2591" * (w - n)


def _dual_bar(pred: float, target: float, w: int = _BAR_W) -> str:
    """Render pred (green) vs target (red) as overlapping bar."""
    p = max(0, min(w, round(pred * w)))
    t = max(0, min(w, round(target * w)))
    chars = []
    for i in range(w):
        if i < min(p, t):
            chars.append("\u2588")  # both — full block
        elif i < t:
            chars.append("\u2593")  # target only — dark shade
        elif i < p:
            chars.append("\u2592")  # pred only — medium shade
        else:
            chars.append("\u2591")  # neither — light shade
    return "".join(chars)


class EvalPreviewApp(App):
    """Browse validation samples and compare model predictions."""

    TITLE = "Model Eval Preview"

    CSS = """
    #main { height: 1fr; }
    #left { width: 42; }
    #right {
        width: 1fr;
        border-left: solid $surface-lighten-2;
        padding: 1 2;
    }
    #samples {
        height: 1fr;
        margin: 0 1;
    }
    #info { }
    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: $primary-background-darken-1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("t", "play_target", "Target"),
        Binding("p", "play_predicted", "Predicted"),
        Binding("space", "replay", "Replay"),
    ]

    def __init__(
        self,
        checkpoint_path: Path,
        dataset_path: Path,
        device: str = "mps",
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.device_str = device
        self._dev = torch.device(device)

        # Will be initialized in on_mount
        self.model: torch.nn.Module | None = None
        self.continuous_names: list[str] = []
        self.categorical_names: list[str] = []
        self.categorical_n_options: list[int] = []
        self.n_continuous = 0
        self.n_categorical = 0
        self.sample_rate = 44100

        self._h5: h5py.File | None = None
        self._val_indices: np.ndarray = np.array([])
        self._engine: Any = None
        self._tier: int = 1
        self._mod_source_names: list[str] = []
        self._mod_dest_names: list[str] = []

        # Current sample state
        self._sel_idx = -1
        self._target_audio: np.ndarray | None = None
        self._pred_audio: np.ndarray | None = None
        self._last_played: str = "target"  # "target" or "predicted"
        self._playing = False
        self._play_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            yield DataTable(id="samples", cursor_type="row")
            with VerticalScroll(id="right"):
                yield Static("Loading model...", id="info")
        status = "  Loading..."
        if not _HAS_AUDIO:
            status += "  |  pip install sounddevice for playback"
        yield Static(status, id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._load_model()
        self._load_dataset()

    def _load_model(self) -> None:
        from inference.pipeline import InferencePipeline

        pipeline = InferencePipeline.from_checkpoint(
            self.checkpoint_path, device=self.device_str
        )
        self.model = pipeline.model
        self.continuous_names = pipeline.continuous_names
        self.categorical_names = pipeline.categorical_names
        self.categorical_n_options = pipeline.categorical_n_options
        self.n_continuous = len(self.continuous_names)
        self.n_categorical = len(self.categorical_names)
        self.sample_rate = pipeline.sample_rate

    def _load_dataset(self) -> None:
        _, val_idx = make_train_val_split(self.dataset_path, 0.15)
        self._val_indices = val_idx
        self._h5 = h5py.File(self.dataset_path, "r")

        # Read tier and mod names from schema, then init render engine
        if "schema" in self._h5:
            self._tier = int(self._h5["schema"].attrs.get("tier", 1))
            raw = self._h5["schema"].attrs.get("mod_source_names", [])
            self._mod_source_names = [
                v.decode() if isinstance(v, bytes) else str(v) for v in raw
            ]
        self._init_engine()

        # Read metadata for table
        tbl = self.query_one("#samples", DataTable)
        tbl.add_columns("#", "Name", "Note")

        hashes = self._h5["metadata/preset_hash"][:]
        notes = self._h5["metadata/midi_note"][:]
        has_names = "metadata/preset_name" in self._h5

        for row_i, ds_idx in enumerate(val_idx):
            h = hashes[ds_idx]
            if isinstance(h, bytes):
                h = h.decode()
            label = h[:10] if h else str(ds_idx)
            if has_names:
                nm = self._h5["metadata/preset_name"][ds_idx]
                if isinstance(nm, bytes):
                    nm = nm.decode()
                if nm:
                    label = nm[:16]
            note = int(notes[ds_idx])
            tbl.add_row(str(row_i), label, str(note), key=str(row_i))

        self.sub_title = f"{self.dataset_path.name} — {len(val_idx)} val samples"
        self._status(f"Ready — {len(val_idx)} validation samples")
        tbl.focus()

    def _init_engine(self) -> None:
        try:
            from datagen.config import MOD_DEST_BLOCKLIST, PipelineConfig
            from datagen.render.engine import RenderEngine

            config = PipelineConfig(sample_rate=self.sample_rate, tier=self._tier)
            self._engine = RenderEngine(config)

            if self._tier >= 3:
                import vita

                all_dests = vita.get_modulation_destinations()
                self._mod_dest_names = sorted(
                    d for d in all_dests if d not in MOD_DEST_BLOCKLIST
                )
        except ImportError:
            self._engine = None

    def _matrix_to_connections(self, matrix: np.ndarray) -> list[dict]:
        """Reconstruct connections list from dense (4, n_src, n_dst) matrix."""
        amounts = matrix[0]
        src_indices, dst_indices = np.nonzero(amounts)
        connections = []
        for si, di in zip(src_indices, dst_indices):
            if si < len(self._mod_source_names) and di < len(self._mod_dest_names):
                connections.append({
                    "source": self._mod_source_names[int(si)],
                    "destination": self._mod_dest_names[int(di)],
                    "amount": float(matrix[0, si, di]),
                    "bipolar": float(matrix[1, si, di]),
                    "power": float(matrix[2, si, di]),
                    "stereo": float(matrix[3, si, di]),
                })
        return connections

    # ── events ──────────────────────────────────────────────

    def on_data_table_row_highlighted(
        self, ev: DataTable.RowHighlighted
    ) -> None:
        if ev.cursor_row is not None:
            self._schedule_select(ev.cursor_row)

    def on_data_table_row_selected(self, ev: DataTable.RowSelected) -> None:
        if ev.row_key:
            idx = int(ev.row_key.value)
            self._cancel_timer()
            self._select_sample(idx)

    def _schedule_select(self, idx: int) -> None:
        self._cancel_timer()
        self._stop()
        self._play_timer = self.set_timer(0.2, lambda: self._select_sample(idx))

    def _cancel_timer(self) -> None:
        if self._play_timer is not None:
            self._play_timer.stop()
            self._play_timer = None

    # ── core: select sample, predict, render, display ───────

    def _select_sample(self, row_idx: int) -> None:
        self._play_timer = None
        if self._h5 is None or self.model is None:
            return
        if row_idx < 0 or row_idx >= len(self._val_indices):
            return

        self._sel_idx = row_idx
        ds_idx = int(self._val_indices[row_idx])

        # Read data
        mel = torch.from_numpy(self._h5[FEATURES_KEY][ds_idx]).unsqueeze(0).to(self._dev)
        cont_target = self._h5["params/continuous"][ds_idx].astype(np.float32)
        cat_target = self._h5["params/categorical"][ds_idx].astype(np.int64)
        midi_note = int(self._h5["metadata/midi_note"][ds_idx])

        # Predict
        with torch.no_grad():
            cont_pred_t, cat_logits, _mod_pred = self.model(mel)
        cont_pred = cont_pred_t.cpu().squeeze(0).numpy()
        cat_pred = np.array(
            [l.argmax(dim=1).item() for l in cat_logits], dtype=np.int64
        )

        # Metrics
        sq_err = (cont_pred - cont_target) ** 2
        cont_mse = float(sq_err.mean())
        cat_correct = int(np.sum(cat_pred == cat_target))

        # Get preset metadata
        preset_hash = self._h5["metadata/preset_hash"][ds_idx]
        if isinstance(preset_hash, bytes):
            preset_hash = preset_hash.decode()
        preset_name = ""
        if "metadata/preset_name" in self._h5:
            preset_name = self._h5["metadata/preset_name"][ds_idx]
            if isinstance(preset_name, bytes):
                preset_name = preset_name.decode()

        # Render audio from params (stereo (2, n_samples), transposed at playback)
        self._target_audio = None
        self._pred_audio = None
        spectral_dist = None

        if self._engine is not None:
            # Build target preset with all available data including modulation
            target_preset: dict[str, Any] = {
                name: float(cont_target[i])
                for i, name in enumerate(self.continuous_names)
            }
            for i, name in enumerate(self.categorical_names):
                target_preset[name] = int(cat_target[i])

            if self._tier >= 3 and "params/modulation_t3" in self._h5:
                matrix = self._h5["params/modulation_t3"][ds_idx]
                connections = self._matrix_to_connections(matrix)
                if connections:
                    target_preset["_modulation_t3"] = {"connections": connections}

            # Build predicted preset (scalar params only — model doesn't predict modulation)
            pred_preset: dict[str, Any] = {
                name: float(cont_pred[i])
                for i, name in enumerate(self.continuous_names)
            }
            for i, name in enumerate(self.categorical_names):
                pred_preset[name] = int(cat_pred[i])

            ta = self._engine.render_preset(target_preset, midi_note=midi_note)
            pa = self._engine.render_preset(pred_preset, midi_note=midi_note)

            if ta is not None:
                self._target_audio = ta
            if pa is not None:
                self._pred_audio = pa

            if self._target_audio is not None and self._pred_audio is not None:
                from datagen.params.importance import _spectral_distance

                spectral_dist = _spectral_distance(
                    self._target_audio, self._pred_audio
                )

        # Build display
        display_name = preset_name or (preset_hash[:12] if preset_hash else "")
        L: list[str] = [
            f"[bold]Sample #{row_idx}[/bold]"
            + (f"  —  {display_name}" if display_name else ""),
            f"  MIDI note   {midi_note}",
            "",
            "[bold]Metrics[/bold]",
            f"  Cont MSE    {cont_mse:.4f}",
            f"  Cat Acc     {cat_correct}/{self.n_categorical}"
            f" ({100*cat_correct/max(self.n_categorical,1):.1f}%)",
        ]

        if spectral_dist is not None:
            L.append(f"  Spectral    {spectral_dist:.4f}")

        if self._target_audio is not None:
            t_mono = self._target_audio.mean(axis=0) if self._target_audio.ndim == 2 else self._target_audio
            t_rms = float(np.sqrt(np.mean(t_mono**2)))
            L.append(f"  Target RMS  {t_rms:.4f}")
        if self._pred_audio is not None:
            p_mono = self._pred_audio.mean(axis=0) if self._pred_audio.ndim == 2 else self._pred_audio
            p_rms = float(np.sqrt(np.mean(p_mono**2)))
            L.append(f"  Pred RMS    {p_rms:.4f}")

        # Worst continuous predictions
        L.append("")
        L.append("[bold]Continuous — worst 15[/bold]")
        L.append(f"  {'param':<32s} {'target':>6s} {'pred':>6s}  {'err':>8s}")
        worst = np.argsort(sq_err)[-15:][::-1]
        for j in worst:
            tv = float(cont_target[j])
            pv = float(cont_pred[j])
            bar = _dual_bar(pv, tv)
            L.append(
                f"  {self.continuous_names[j]:<32s} {tv:6.3f} {pv:6.3f}  {sq_err[j]:8.4f}"
            )
            L.append(f"    {bar}")

        # Categorical errors
        errors = []
        for i in range(self.n_categorical):
            if cat_pred[i] != cat_target[i]:
                errors.append((self.categorical_names[i], int(cat_pred[i]), int(cat_target[i])))

        if errors:
            L.append("")
            L.append(f"[bold]Categorical — {len(errors)} errors[/bold]")
            for name, pv, tv in errors:
                L.append(f"  {name:<32s} pred={pv}  target={tv}")
        else:
            L.append("")
            L.append("[bold]Categorical — all correct[/bold]")

        self.query_one("#info", Static).update("\n".join(L))

        # Auto-play target
        self._play_target()

    # ── playback ────────────────────────────────────────────

    def action_play_target(self) -> None:
        self._play_target()

    def action_play_predicted(self) -> None:
        self._play_predicted()

    def action_replay(self) -> None:
        if self._last_played == "predicted":
            self._play_predicted()
        else:
            self._play_target()

    def _play_target(self) -> None:
        if not _HAS_AUDIO or self._target_audio is None:
            self._status("No target audio")
            return
        self._stop_audio()
        sd.play(self._target_audio.T, samplerate=self.sample_rate)
        self._playing = True
        self._last_played = "target"
        self._status(f"Playing TARGET #{self._sel_idx}")

    def _play_predicted(self) -> None:
        if not _HAS_AUDIO or self._pred_audio is None:
            self._status("No predicted audio")
            return
        self._stop_audio()
        sd.play(self._pred_audio.T, samplerate=self.sample_rate)
        self._playing = True
        self._last_played = "predicted"
        self._status(f"Playing PREDICTED #{self._sel_idx}")

    def _stop(self) -> None:
        self._cancel_timer()
        self._stop_audio()

    def _stop_audio(self) -> None:
        if _HAS_AUDIO and self._playing:
            sd.stop()
        self._playing = False

    def action_quit(self) -> None:
        self._stop()
        if self._h5 is not None:
            self._h5.close()
        self.exit()

    def _status(self, msg: str) -> None:
        try:
            self.query_one("#status-bar", Static).update(
                f"  {msg}  |  [t]arget [p]redicted [space]replay [q]uit"
            )
        except Exception:
            pass
