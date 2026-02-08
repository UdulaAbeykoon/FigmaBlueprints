"""Click CLI for inference and demo."""

from __future__ import annotations

import logging
from pathlib import Path

import click


@click.group()
def main() -> None:
    """Vital inverse synthesis inference and demo."""


# ------------------------------------------------------------------
# infer
# ------------------------------------------------------------------


@main.command()
@click.option("-c", "--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("-i", "--input", "input_audio", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("-o", "--output", "output_preset", type=click.Path(path_type=Path), default=None)
@click.option("--render/--no-render", default=False, help="Render predicted audio.")
@click.option("--tutorial/--no-tutorial", default=False, help="Generate LLM tutorial.")
@click.option("--refine/--no-refine", default=False, help="Refine with CMA-ES optimization.")
@click.option("--refine-evals", type=int, default=500, help="CMA-ES max function evaluations.")
@click.option("--refine-timeout", type=float, default=60.0, help="CMA-ES timeout in seconds.")
@click.option("--refine-sigma", type=float, default=0.1, help="CMA-ES initial step size.")
@click.option("--device", type=str, default="cuda")
@click.option("-v", "--verbose", is_flag=True, default=False)
def infer(
    checkpoint: Path,
    input_audio: Path,
    output_preset: Path | None,
    render: bool,
    tutorial: bool,
    refine: bool,
    refine_evals: int,
    refine_timeout: float,
    refine_sigma: float,
    device: str,
    verbose: bool,
) -> None:
    """Run inference on an audio file to predict Vital parameters."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    from inference.pipeline import InferencePipeline

    pipeline = InferencePipeline.from_checkpoint(checkpoint, device)

    refinement_info = None
    if refine:
        params, confidence, refinement_info = pipeline.predict_with_refinement(
            input_audio,
            sigma0=refine_sigma,
            max_evals=refine_evals,
            timeout_sec=refine_timeout,
        )
        if refinement_info.get("improved"):
            log.info(
                "CMA-ES improved loss: %.4f -> %.4f (%d evals, %.1fs)",
                refinement_info["initial_loss"],
                refinement_info["final_loss"],
                refinement_info["n_evals"],
                refinement_info["elapsed_sec"],
            )
        elif not refinement_info.get("skipped"):
            log.info("CMA-ES did not improve (initial loss was already optimal)")
    else:
        params, confidence = pipeline.predict_with_confidence(input_audio)

    log.info("Predicted parameters:")
    for name, value in sorted(params.items()):
        if isinstance(value, float):
            log.info("  %s: %.4f", name, value)
        else:
            conf = confidence.get(name, 0)
            log.info("  %s: %d (conf: %.0f%%)", name, value, conf * 100)

    # Export .vital preset
    if output_preset is None:
        output_preset = Path(input_audio).with_suffix(".vital")
    pipeline.export_vital_preset(params, output_preset)

    # Render predicted audio
    if render:
        pred_audio = pipeline.render_comparison(params)
        if pred_audio is not None:
            import soundfile as sf
            audio_path = output_preset.with_suffix(".wav")
            sf.write(str(audio_path), pred_audio.T, pipeline.sample_rate)
            log.info("Rendered audio to: %s", audio_path)

    # Generate tutorial
    if tutorial:
        try:
            from inference.tutorial import TutorialGenerator
            gen = TutorialGenerator()
            tutorial_text = gen.generate(params, confidence=confidence)
            tutorial_path = output_preset.with_suffix(".md")
            tutorial_path.write_text(tutorial_text)
            log.info("Tutorial saved to: %s", tutorial_path)
        except Exception as e:
            log.warning("Tutorial generation failed: %s", e)


# ------------------------------------------------------------------
# demo
# ------------------------------------------------------------------


@main.command()
@click.option("-c", "--checkpoint", type=click.Path(exists=True, path_type=Path), default=Path("checkpoints/best_model.pt"))
@click.option("--device", type=str, default="cuda")
@click.option("--share", is_flag=True, default=False, help="Create public Gradio link.")
@click.option("--port", type=int, default=7860)
def demo(
    checkpoint: Path,
    device: str,
    share: bool,
    port: int,
) -> None:
    """Launch the Gradio demo interface."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    from inference.demo import create_demo

    app = create_demo(checkpoint, device)
    app.launch(share=share, server_port=port)


if __name__ == "__main__":
    main()
