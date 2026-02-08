"""Click CLI for dataset generation pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np

log = logging.getLogger("datagen")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
def main() -> None:
    """Vital inverse synthesis dataset generation pipeline."""
    pass


@main.command()
@click.option("--tier", type=click.IntRange(1, 3), default=1, help="Parameter tier (1-3).")
@click.option("--n-samples", "-n", type=int, default=1000, help="Number of samples to generate.")
@click.option("--output", "-o", type=click.Path(), default="data/dataset.h5", help="Output HDF5 path.")
@click.option("--workers", "-w", type=int, default=1, help="Number of render workers.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option("--sample-rate", type=int, default=44100, help="Audio sample rate.")
@click.option("--duration", type=float, default=2.0, help="Render duration in seconds.")
@click.option("--midi-notes", type=str, default="48,60,72", help="Comma-separated MIDI notes.")
@click.option("--community-dir", type=click.Path(exists=False), multiple=True, help="Community preset directory (repeatable).")
@click.option("--include-factory/--no-factory", default=False, help="Include factory presets.")
@click.option("--wavetable-catalog", type=click.Path(exists=False), default=None, help="Wavetable catalog JSON.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging.")
def generate(
    tier: int,
    n_samples: int,
    output: str,
    workers: int,
    seed: int,
    sample_rate: int,
    duration: float,
    midi_notes: str,
    community_dir: tuple[str, ...],
    include_factory: bool,
    wavetable_catalog: str | None,
    verbose: bool,
) -> None:
    """Generate a dataset of audio-parameter pairs."""
    _setup_logging(verbose)

    import vita

    from datagen.config import PipelineConfig
    from datagen.params.registry import ParamRegistry
    from datagen.pipeline import Pipeline
    from datagen.wavetables.catalog import WavetableCatalog

    notes = [int(n.strip()) for n in midi_notes.split(",")]

    config = PipelineConfig(
        tier=tier,
        n_samples=n_samples,
        output_path=Path(output),
        seed=seed,
        workers=workers,
        sample_rate=sample_rate,
        render_duration=duration,
        midi_notes=notes,
    )

    # Build registry from live Vita synth (source of truth for all params)
    click.echo("Discovering parameters from Vita synth...")
    synth = vita.Synth()
    registry = ParamRegistry.from_synth(synth)
    del synth  # free the synth instance

    # Load or create wavetable catalog
    catalog = None
    if wavetable_catalog and Path(wavetable_catalog).exists():
        catalog = WavetableCatalog.from_json(Path(wavetable_catalog))
        click.echo(f"Loaded wavetable catalog: {len(catalog)} wavetables")
        # Add virtual wavetable params to registry
        registry.add_wavetable_params(len(catalog))
    else:
        click.echo("No wavetable catalog provided; using default indices.")

    n_cont = len(registry.continuous_names(tier))
    n_cat = len(registry.categorical_names(tier))
    n_wt = len(registry.wavetable_names(tier))
    click.echo(f"Tier {tier}: {n_cont} continuous, {n_cat} categorical, {n_wt} wavetable params")

    # Collect extra presets (community + factory)
    extra_presets: list = []

    if community_dir:
        from datagen.presets.ingest import PresetIngester
        ingester = PresetIngester(config, registry, catalog)
        for cd in community_dir:
            community = ingester.scan(Path(cd))
            extra_presets.extend(community)
            click.echo(f"Ingested {len(community)} community presets from {cd}")

    if include_factory:
        from datagen.presets.factory import ingest_factory_presets
        factory = ingest_factory_presets(config, registry, catalog)
        extra_presets.extend(factory)
        click.echo(f"Loaded {len(factory)} factory presets")

    pipeline = Pipeline(config, registry=registry, catalog=catalog)
    stats = pipeline.run(extra_presets=extra_presets if extra_presets else None)

    click.echo(f"\nGeneration complete:")
    click.echo(f"  Accepted:  {stats.total_accepted}")
    click.echo(f"  Attempted: {stats.total_attempted}")
    click.echo(f"  Rejected:  {stats.total_rejected} ({stats.rejection_rate:.1%})")
    click.echo(f"  Time:      {stats.elapsed_seconds:.1f}s")
    click.echo(f"  Speed:     {stats.renders_per_second:.1f} renders/s")
    if stats.rejection_reasons:
        click.echo(f"  Rejection reasons:")
        for reason, count in sorted(stats.rejection_reasons.items()):
            click.echo(f"    {reason}: {count}")
    click.echo(f"  Output:    {config.output_path}")


@main.command("discover-wt")
@click.option("--output", "-o", type=click.Path(), default="data/wavetable_catalog.json", help="Output JSON path.")
@click.option("--extra-dir", type=click.Path(exists=True), multiple=True, help="Additional dirs to scan.")
@click.option("--extra-file", type=click.Path(exists=True), multiple=True, help="Specific .vital files to scan.")
@click.option("--verbose", "-v", is_flag=True)
def discover_wt(
    output: str,
    extra_dir: tuple[str, ...],
    extra_file: tuple[str, ...],
    verbose: bool,
) -> None:
    """Discover factory wavetables and save catalog to JSON."""
    _setup_logging(verbose)

    from datagen.wavetables.catalog import WavetableCatalog

    extra_dirs = [Path(d) for d in extra_dir] if extra_dir else None
    extra_files = [Path(f) for f in extra_file] if extra_file else None

    catalog = WavetableCatalog.from_discovery(
        extra_dirs=extra_dirs, extra_files=extra_files
    )

    if len(catalog) == 0:
        click.echo("No wavetables found. Try providing --extra-dir or --extra-file.")
        click.echo("Expected locations (macOS):")
        click.echo("  ~/Library/Application Support/Vital/Factory/Presets/")
        sys.exit(1)

    out_path = Path(output)
    catalog.save_json(out_path)
    click.echo(f"Discovered {len(catalog)} wavetables -> {out_path}")

    # Show first few names
    names = catalog.names()
    for name in names[:10]:
        click.echo(f"  - {name}")
    if len(names) > 10:
        click.echo(f"  ... and {len(names) - 10} more")


@main.command("download-presets")
@click.option("--output", "-o", type=click.Path(), default="presets/community", help="Output directory.")
@click.option("--verbose", "-v", is_flag=True)
def download_presets(output: str, verbose: bool) -> None:
    """Download community presets from known GitHub repositories."""
    _setup_logging(verbose)

    from datagen.presets.download import download_presets as _download

    out_path = Path(output)
    files = _download(out_path)
    click.echo(f"Downloaded {len(files)} .vital files to {out_path}")


@main.command("compute-weights")
@click.option("--n-base", type=int, default=500, help="Number of base presets for perturbation.")
@click.option("--output", "-o", type=click.Path(), required=True, help="HDF5 file to write weights to.")
@click.option("--seed", type=int, default=42)
@click.option("--workers", "-w", type=int, default=0, help="Parallel workers (0=auto, 1=serial).")
@click.option("--verbose", "-v", is_flag=True)
def compute_weights(n_base: int, output: str, seed: int, workers: int, verbose: bool) -> None:
    """Compute perturbation-based parameter importance weights."""
    _setup_logging(verbose)

    import vita

    from datagen.config import PipelineConfig
    from datagen.params.importance import compute_importance_weights
    from datagen.params.registry import ParamRegistry
    from datagen.render.engine import RenderEngine
    from datagen.storage.writer import HDF5Writer
    from datagen.storage.schema import HDF5Schema

    config = PipelineConfig()
    synth = vita.Synth()
    registry = ParamRegistry.from_synth(synth)
    del synth
    engine = RenderEngine(config)

    # Read continuous param names from the HDF5 file so weights match the dataset
    out_path = Path(output)
    continuous_names_override = None
    if out_path.exists():
        import h5py

        with h5py.File(out_path, "r") as f:
            if "schema" in f and "continuous_names" in f["schema"].attrs:
                raw = f["schema"].attrs["continuous_names"]
                continuous_names_override = [
                    n.decode("utf-8") if isinstance(n, bytes) else n
                    for n in raw
                ]
                log.info(
                    "Read %d continuous param names from dataset schema",
                    len(continuous_names_override),
                )

    weights = compute_importance_weights(
        engine=engine,
        config=config,
        registry=registry,
        n_base_presets=n_base,
        seed=seed,
        workers=workers,
        continuous_names_override=continuous_names_override,
    )

    # Write weights to the HDF5 file
    schema = HDF5Schema.from_config(config, registry)
    with HDF5Writer(out_path, schema) as writer:
        writer.write_importance_weights(weights)

    click.echo(f"Wrote importance weights ({len(weights)} params) to {out_path}")
    click.echo(f"  Min: {weights.min():.4f}, Max: {weights.max():.4f}, Mean: {weights.mean():.4f}")


@main.command()
@click.option("--data-dir", "-d", type=click.Path(exists=True), default="data", help="Directory containing HDF5 files.")
def preview(data_dir: str) -> None:
    """Browse and listen to dataset samples in a TUI."""
    try:
        from datagen.preview import PreviewApp
    except ImportError:
        click.echo("Preview requires extra deps: pip install 'vital-inverse-synthesis[preview]'")
        click.echo("Or directly: pip install textual sounddevice")
        raise SystemExit(1)

    PreviewApp(Path(data_dir)).run()


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--sample", "-s", type=int, default=None, help="Show details for a specific sample index.")
def inspect(path: str, sample: int | None) -> None:
    """Inspect an HDF5 dataset file."""
    from datagen.storage.reader import HDF5Reader

    with HDF5Reader(Path(path)) as reader:
        info = reader.info()

        click.echo(f"Dataset: {path}")
        click.echo(f"  Samples:   {info['n_samples']}")
        click.echo(f"  File size: {info['file_size_mb']:.1f} MB")

        click.echo(f"\n  Shapes:")
        for name, shape in sorted(info.get("shapes", {}).items()):
            click.echo(f"    {name}: {shape}")

        if "schema" in info:
            click.echo(f"\n  Schema:")
            for key, val in sorted(info["schema"].items()):
                if isinstance(val, list) and len(val) > 5:
                    click.echo(f"    {key}: [{len(val)} items]")
                else:
                    click.echo(f"    {key}: {val}")

        if "source_distribution" in info:
            click.echo(f"\n  Source distribution:")
            for src, count in sorted(info["source_distribution"].items()):
                click.echo(f"    {src}: {count}")

        if "tier_distribution" in info:
            click.echo(f"\n  Tier distribution:")
            for tier, count in sorted(info["tier_distribution"].items()):
                click.echo(f"    Tier {tier}: {count}")

        if "midi_note_distribution" in info:
            click.echo(f"\n  MIDI note distribution:")
            for note, count in sorted(info["midi_note_distribution"].items()):
                click.echo(f"    Note {note}: {count}")

        # Show importance weights if present
        weights = reader.get_importance_weights()
        if weights is not None:
            click.echo(f"\n  Importance weights: {len(weights)} params")
            click.echo(f"    Min: {weights.min():.4f}, Max: {weights.max():.4f}")

        # Show specific sample
        if sample is not None:
            if sample >= info["n_samples"]:
                click.echo(f"\n  Error: sample index {sample} out of range (max {info['n_samples'] - 1})")
                return

            s = reader.get_sample(sample)
            click.echo(f"\n  Sample {sample}:")
            click.echo(f"    MIDI note:  {s['midi_note']}")
            click.echo(f"    Source:     {s['source']}")
            click.echo(f"    Tier:       {s['tier']}")
            click.echo(f"    Audio:      shape={s['audio'].shape}, "
                       f"RMS={float(np.sqrt(np.mean(s['audio'].astype(np.float64)**2))):.4f}, "
                       f"peak={float(abs(s['audio']).max()):.4f}")
            click.echo(f"    Continuous: shape={s['continuous'].shape}, "
                       f"range=[{s['continuous'].min():.3f}, {s['continuous'].max():.3f}]")
            click.echo(f"    Categorical: {s['categorical']}")


if __name__ == "__main__":
    main()
