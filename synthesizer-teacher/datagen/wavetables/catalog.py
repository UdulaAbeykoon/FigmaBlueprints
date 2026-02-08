"""Wavetable catalog: name → index → base64 data, persisted to JSON."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datagen.wavetables.discovery import WavetableEntry, discover_wavetables

log = logging.getLogger(__name__)


class WavetableCatalog:
    """Manages a catalog of Vital wavetables for preset generation.

    Each wavetable is identified by index and can be looked up by name
    or content hash. The catalog persists to JSON for reuse across runs.
    """

    def __init__(self) -> None:
        self._entries: list[WavetableEntry] = []
        self._by_name: dict[str, int] = {}
        self._by_hash: dict[str, int] = {}

    @classmethod
    def from_discovery(
        cls,
        extra_dirs: list[Path] | None = None,
        extra_files: list[Path] | None = None,
    ) -> WavetableCatalog:
        """Build catalog by scanning Vital install directories."""
        catalog = cls()
        entries = discover_wavetables(extra_dirs=extra_dirs, extra_files=extra_files)
        for entry in entries:
            catalog.add(entry)
        return catalog

    @classmethod
    def from_json(cls, path: Path) -> WavetableCatalog:
        """Load a previously saved catalog from JSON."""
        catalog = cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data["wavetables"]:
            entry = WavetableEntry(
                name=item["name"],
                content_hash=item["content_hash"],
                base64_data=item["base64_data"],
                source_file=item.get("source_file", ""),
            )
            catalog.add(entry)
        log.info("Loaded %d wavetables from %s", len(catalog), path)
        return catalog

    def add(self, entry: WavetableEntry) -> int:
        """Add a wavetable entry. Returns its index."""
        if entry.content_hash in self._by_hash:
            return self._by_hash[entry.content_hash]

        idx = len(self._entries)
        self._entries.append(entry)
        self._by_hash[entry.content_hash] = idx
        # Names may collide; first one wins for name→index
        if entry.name not in self._by_name:
            self._by_name[entry.name] = idx
        return idx

    def to_json_string(self) -> str:
        """Serialize catalog to a JSON string for embedding in checkpoints."""
        data = {
            "version": "1.0.0",
            "count": len(self._entries),
            "wavetables": [
                {
                    "index": i,
                    "name": e.name,
                    "content_hash": e.content_hash,
                    "base64_data": e.base64_data,
                    "source_file": e.source_file,
                }
                for i, e in enumerate(self._entries)
            ],
        }
        return json.dumps(data)

    @classmethod
    def from_json_string(cls, s: str) -> WavetableCatalog:
        """Reconstruct catalog from a JSON string (e.g. from checkpoint)."""
        catalog = cls()
        data = json.loads(s)
        for item in data["wavetables"]:
            entry = WavetableEntry(
                name=item["name"],
                content_hash=item["content_hash"],
                base64_data=item["base64_data"],
                source_file=item.get("source_file", ""),
            )
            catalog.add(entry)
        return catalog

    def save_json(self, path: Path) -> None:
        """Persist the catalog to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0.0",
            "count": len(self._entries),
            "wavetables": [
                {
                    "index": i,
                    "name": e.name,
                    "content_hash": e.content_hash,
                    "base64_data": e.base64_data,
                    "source_file": e.source_file,
                }
                for i, e in enumerate(self._entries)
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("Saved %d wavetables to %s", len(self._entries), path)

    def get_by_index(self, idx: int) -> WavetableEntry:
        return self._entries[idx]

    def get_by_name(self, name: str) -> WavetableEntry | None:
        idx = self._by_name.get(name)
        return self._entries[idx] if idx is not None else None

    def get_by_hash(self, content_hash: str) -> WavetableEntry | None:
        idx = self._by_hash.get(content_hash)
        return self._entries[idx] if idx is not None else None

    def index_of_hash(self, content_hash: str) -> int | None:
        return self._by_hash.get(content_hash)

    def names(self) -> list[str]:
        return [e.name for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, content_hash: str) -> bool:
        return content_hash in self._by_hash
