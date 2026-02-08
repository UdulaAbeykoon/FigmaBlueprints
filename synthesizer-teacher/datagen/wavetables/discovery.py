"""Scan Vital install directories for factory wavetable data."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from datagen.config import VITAL_PRESET_DIRS, VITAL_WAVETABLE_DIRS

log = logging.getLogger(__name__)


@dataclass
class WavetableEntry:
    """A single discovered wavetable."""

    name: str
    content_hash: str  # SHA256 of base64 data
    base64_data: str
    source_file: str  # path to the .vital file it was found in


def _hash_wavetable(base64_data: str) -> str:
    """SHA256 hash of wavetable base64 content for deduplication."""
    return hashlib.sha256(base64_data.encode("utf-8")).hexdigest()


def _extract_wavetables_from_preset(
    preset_path: Path,
) -> list[WavetableEntry]:
    """Extract wavetable data from a .vital preset JSON file.

    Vital presets store wavetable data in ``settings.wavetables`` as a list
    of objects, each containing ``name`` and ``groups`` with base64-encoded
    audio data.
    """
    entries: list[WavetableEntry] = []
    try:
        data = json.loads(preset_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        log.debug("Could not parse %s: %s", preset_path, e)
        return entries

    settings = data.get("settings", data)
    wavetables = settings.get("wavetables", [])

    for wt in wavetables:
        name = wt.get("name", "unknown")
        # The wavetable data is stored in groups[].components[].audio_file
        # We serialize the entire wavetable object as our "data" for injection
        wt_json = json.dumps(wt, separators=(",", ":"), sort_keys=True)
        content_hash = _hash_wavetable(wt_json)

        entries.append(WavetableEntry(
            name=name,
            content_hash=content_hash,
            base64_data=wt_json,
            source_file=str(preset_path),
        ))

    return entries


def discover_wavetables(
    extra_dirs: list[Path] | None = None,
    extra_files: list[Path] | None = None,
) -> list[WavetableEntry]:
    """Scan known Vital install paths for factory wavetables.

    Searches preset directories for .vital files, extracts embedded
    wavetable data, and deduplicates by content hash.

    Args:
        extra_dirs: Additional directories to scan beyond defaults.
        extra_files: Specific .vital files to scan.

    Returns:
        Deduplicated list of WavetableEntry objects.
    """
    search_dirs = list(VITAL_PRESET_DIRS)
    if extra_dirs:
        search_dirs.extend(extra_dirs)

    seen_hashes: set[str] = set()
    unique_entries: list[WavetableEntry] = []

    # Scan preset directories for .vital files
    preset_files: list[Path] = []
    for d in search_dirs:
        if d.exists() and d.is_dir():
            found = list(d.rglob("*.vital"))
            preset_files.extend(found)
            log.info("Scanning %s (%d .vital files)", d, len(found))
        else:
            log.debug("Directory not found: %s", d)

    if extra_files:
        preset_files.extend(extra_files)

    log.info("Scanning %d preset files for wavetables", len(preset_files))

    for preset_path in preset_files:
        for entry in _extract_wavetables_from_preset(preset_path):
            if entry.content_hash not in seen_hashes:
                seen_hashes.add(entry.content_hash)
                unique_entries.append(entry)

    log.info(
        "Discovered %d unique wavetables from %d preset files",
        len(unique_entries), len(preset_files),
    )
    return unique_entries


def discover_wavetable_files(
    extra_dirs: list[Path] | None = None,
) -> list[Path]:
    """Find standalone .vitaltable or .wav wavetable files.

    Vital also stores wavetables as separate files in Wavetable directories.
    """
    search_dirs = list(VITAL_WAVETABLE_DIRS)
    if extra_dirs:
        search_dirs.extend(extra_dirs)

    files: list[Path] = []
    for d in search_dirs:
        if d.exists() and d.is_dir():
            files.extend(d.rglob("*.vitaltable"))
            files.extend(d.rglob("*.wav"))

    log.info("Found %d standalone wavetable files", len(files))
    return files
