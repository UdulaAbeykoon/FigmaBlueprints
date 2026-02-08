"""TUI preview tool for browsing and listening to generated datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import sounddevice as sd

    _HAS_AUDIO = True
except (ImportError, OSError):
    _HAS_AUDIO = False

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import DataTable, Footer, Header, Label, OptionList, Static

from datagen.storage.reader import HDF5Reader

_BAR_W = 20


def _bar(v: float, w: int = _BAR_W) -> str:
    """Render a [0,1] value as a Unicode bar."""
    n = max(0, min(w, round(v * w)))
    return "\u2588" * n + "\u2591" * (w - n)


class PreviewApp(App):
    """Browse generated datasets and listen to audio samples."""

    TITLE = "Vital Dataset Preview"

    CSS = """
    #main { height: 1fr; }
    #left { width: 58; }
    #right {
        width: 1fr;
        border-left: solid $surface-lighten-2;
    }
    #ds-label, #samp-label {
        text-style: bold;
        color: $accent;
        padding: 0 1;
    }
    #ds-list {
        height: auto;
        max-height: 7;
        margin: 0 1 1 1;
    }
    #samples {
        height: 1fr;
        margin: 0 1;
    }
    #info {
        padding: 1 2;
    }
    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: $primary-background-darken-1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("space", "toggle_play", "Play/Stop"),
    ]

    def __init__(self, data_dir: Path) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.h5_files = sorted(data_dir.glob("*.h5"))
        self._reader: HDF5Reader | None = None
        self._cont_names: list[str] = []
        self._cat_names: list[str] = []
        self._wt_names: list[str] = []
        self._mod_src_names: list[str] = []
        self._n: int = 0
        self._audio: np.ndarray | None = None
        self._sample_rate: int = 44100
        self._sel_idx: int = -1
        self._playing = False
        self._play_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Label("Datasets", id="ds-label")
                opts = [f.name for f in self.h5_files] or ["(no .h5 files)"]
                yield OptionList(*opts, id="ds-list")
                yield Label("Samples", id="samp-label")
                yield DataTable(id="samples", cursor_type="row")
            with VerticalScroll(id="right"):
                yield Static(self._welcome(), id="info")
        status = "  Ready"
        if not _HAS_AUDIO:
            status += "  |  pip install sounddevice for playback"
        yield Static(status, id="status-bar")
        yield Footer()

    def _welcome(self) -> str:
        if not self.h5_files:
            return (
                "[bold]No datasets found[/bold]\n\n"
                f"Looking in: {self.data_dir}\n\n"
                "Generate one with:\n"
                "  python -m datagen generate --tier 1 -n 100 -o data/test.h5"
            )
        return "Select a dataset to begin."

    def on_mount(self) -> None:
        tbl = self.query_one("#samples", DataTable)
        tbl.add_columns("#", "Name", "Source", "Note")
        if self.h5_files:
            self._open(self.h5_files[0])
        tbl.focus()

    # ── events ──────────────────────────────────────────────

    def on_option_list_option_selected(
        self, ev: OptionList.OptionSelected
    ) -> None:
        idx = ev.option_index
        if 0 <= idx < len(self.h5_files):
            self._stop()
            self._open(self.h5_files[idx])

    def on_data_table_row_highlighted(
        self, ev: DataTable.RowHighlighted
    ) -> None:
        if ev.cursor_row is not None:
            self._show(ev.cursor_row)
            self._schedule_play(ev.cursor_row)

    def on_data_table_row_selected(
        self, ev: DataTable.RowSelected
    ) -> None:
        if ev.row_key:
            idx = int(ev.row_key.value)
            self._show(idx)
            self._cancel_play_timer()
            self._play(idx)

    def action_toggle_play(self) -> None:
        if self._playing:
            self._stop()
        else:
            tbl = self.query_one("#samples", DataTable)
            if tbl.cursor_row is not None and self._reader:
                self._play(tbl.cursor_row)

    def action_quit(self) -> None:
        self._stop()
        if self._reader:
            self._reader.close()
        self.exit()

    # ── dataset loading ─────────────────────────────────────

    def _open(self, path: Path) -> None:
        if self._reader:
            self._reader.close()
        self._reader = HDF5Reader(path)
        self._reader.open()

        self._cont_names = self._reader.get_continuous_names()
        self._cat_names = self._reader.get_categorical_names()
        self._n = self._reader.n_samples

        # Read sample rate and extra schema attrs for display
        f = self._reader._file
        schema = f.get("schema")
        if schema is not None:
            self._sample_rate = int(schema.attrs.get("sample_rate", 44100))
            raw_wt = schema.attrs.get("wavetable_names", [])
            self._wt_names = [
                v.decode() if isinstance(v, bytes) else str(v) for v in raw_wt
            ]
            raw_ms = schema.attrs.get("mod_source_names", [])
            self._mod_src_names = [
                v.decode() if isinstance(v, bytes) else str(v) for v in raw_ms
            ]
        else:
            self._wt_names = []
            self._mod_src_names = []

        # Populate table from metadata only (fast)
        tbl = self.query_one("#samples", DataTable)
        tbl.clear()

        srcs = f["metadata/source"][:]
        notes = f["metadata/midi_note"][:]
        has_names = "metadata/preset_name" in f
        names = f["metadata/preset_name"][:] if has_names else None
        has_hashes = "metadata/preset_hash" in f
        hashes = f["metadata/preset_hash"][:] if has_hashes else None
        for i in range(self._n):
            s = srcs[i]
            if isinstance(s, bytes):
                s = s.decode()
            nm = ""
            if names is not None:
                nm = names[i]
                if isinstance(nm, bytes):
                    nm = nm.decode()
            if not nm and hashes is not None:
                h = hashes[i]
                if isinstance(h, bytes):
                    h = h.decode()
                if h:
                    nm = h[:8]
            tbl.add_row(str(i), nm[:18], s[:9], str(int(notes[i])), key=str(i))

        self.sub_title = f"{path.name} \u2014 {self._n} samples"
        if self._n > 0:
            self._show(0)

    # ── sample display ──────────────────────────────────────

    def _show(self, idx: int) -> None:
        if not self._reader or not (0 <= idx < self._n):
            return
        self._sel_idx = idx
        sample = self._reader.get_sample(idx)
        self._audio = sample["audio"]
        audio = self._audio

        rms = float(np.sqrt(np.mean(audio**2)))
        peak = float(np.abs(audio).max())

        preset_name = sample.get("preset_name", "")
        preset_hash = sample.get("preset_hash", "")
        display_name = preset_name or (preset_hash[:12] if preset_hash else "")
        L: list[str] = [
            f"[bold]Sample #{idx}[/bold]"
            + (f"  —  {display_name}" if display_name else "")
            + "\n",
            f"  Name        {preset_name or '[dim](none)[/dim]'}",
            f"  Hash        {preset_hash[:16]}…" if preset_hash else "",
            f"  Source      {sample['source']}",
            f"  MIDI note   {sample['midi_note']}",
            f"  Tier        {sample['tier']}",
            f"  RMS         {rms:.4f}  {_bar(min(rms * 5, 1.0))}",
            f"  Peak        {peak:.4f}  {_bar(peak)}",
            f"  Shape       {audio.shape}",
            "",
        ]

        # Continuous parameters
        cont = sample["continuous"]
        if self._cont_names:
            L.append(f"[bold]Parameters ({len(cont)})[/bold]\n")
            for i, nm in enumerate(self._cont_names):
                if i < len(cont):
                    v = float(cont[i])
                    L.append(f"  {nm:<30s}{_bar(v)} {v:.3f}")
            L.append("")

        # Categorical parameters (categorical + wavetable)
        cat = sample["categorical"]
        all_cat_names = list(self._cat_names) + list(self._wt_names)
        if all_cat_names or len(cat) > 0:
            L.append(f"[bold]Categorical ({len(cat)})[/bold]\n")
            for i in range(len(cat)):
                nm = all_cat_names[i] if i < len(all_cat_names) else f"cat_{i}"
                L.append(f"  {nm:<30s}{int(cat[i])}")
            L.append("")

        # Tier 3 modulation — shape is (4, n_src, n_dst) or (n_src, n_dst)
        mod3 = sample.get("modulation_t3")
        if mod3 is not None and np.count_nonzero(mod3):
            # Handle both old (n_src, n_dst) and new (4, n_src, n_dst) formats
            if mod3.ndim == 3:
                amounts = mod3[0]  # channel 0 = amount
            else:
                amounts = mod3

            nc = int(np.count_nonzero(amounts))
            L.append("[bold]Modulation \u2014 Tier 3[/bold]\n")
            L.append(f"  Active connections  {nc}")
            L.append(f"  Matrix              {mod3.shape}")
            # Show top connections by absolute amount
            src_idx, dst_idx = np.nonzero(amounts)
            abs_vals = np.abs(amounts[src_idx, dst_idx])
            order = np.argsort(-abs_vals)[:10]
            if len(order) > 0:
                L.append("")
                L.append("  [dim]Top connections:[/dim]")
                for rank in order:
                    si, di = int(src_idx[rank]), int(dst_idx[rank])
                    val = float(amounts[si, di])
                    src_nm = (
                        self._mod_src_names[si]
                        if si < len(self._mod_src_names)
                        else f"src_{si}"
                    )
                    dst_nm = (
                        self._cont_names[di]
                        if di < len(self._cont_names)
                        else f"dst_{di}"
                    )
                    flags = ""
                    if mod3.ndim == 3:
                        if mod3[1, si, di] > 0.5:
                            flags += " B"
                        if mod3[3, si, di] > 0.5:
                            flags += " S"
                    L.append(f"    {src_nm} -> {dst_nm:<24s}{val:+.3f}{flags}")
            L.append("")

        self.query_one("#info", Static).update("\n".join(L))

    # ── playback ────────────────────────────────────────────

    def _schedule_play(self, idx: int) -> None:
        """Debounced auto-play: waits 150ms so rapid scrolling doesn't stack."""
        self._cancel_play_timer()
        self._stop()
        self._play_timer = self.set_timer(0.15, lambda: self._play(idx))

    def _cancel_play_timer(self) -> None:
        if self._play_timer is not None:
            self._play_timer.stop()
            self._play_timer = None

    def _play(self, idx: int) -> None:
        self._play_timer = None
        if not _HAS_AUDIO:
            self._status("pip install sounddevice for audio playback")
            return
        if self._sel_idx != idx:
            self._show(idx)
        if self._audio is None:
            return
        if self._playing:
            sd.stop()
        sd.play(self._audio.T, samplerate=self._sample_rate)
        self._playing = True
        self._status(f"Playing #{idx}")

    def _stop(self) -> None:
        self._cancel_play_timer()
        if _HAS_AUDIO and self._playing:
            sd.stop()
        self._playing = False

    def _status(self, msg: str) -> None:
        try:
            self.query_one("#status-bar", Static).update(f"  {msg}")
        except Exception:
            pass
