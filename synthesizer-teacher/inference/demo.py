"""Gradio demo app for Vital inverse synthesis."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_temp_files: list[str] = []


def create_demo(checkpoint_path: str | Path, device: str = "cuda"):
    """Create the Gradio demo interface.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        device: Device to run inference on.

    Returns:
        Gradio Blocks interface.
    """
    import gradio as gr

    from inference.pipeline import InferencePipeline
    from inference.tutorial import TutorialGenerator, generate_offline_tutorial

    # Load the inference pipeline
    log.info("Loading inference pipeline from %s", checkpoint_path)
    pipeline = InferencePipeline.from_checkpoint(checkpoint_path, device)
    log.info("Pipeline loaded successfully")

    # Try to initialize tutorial generator
    tutorial_gen = None
    try:
        tutorial_gen = TutorialGenerator()
        log.info("Tutorial generator initialized with Anthropic API")
    except (ImportError, ValueError) as e:
        log.warning("Tutorial generator unavailable: %s", e)
        log.info("Will use offline template-based tutorials")

    def predict_and_analyze(
        audio_file: str,
        sound_description: str,
        use_cmaes: bool = False,
    ) -> tuple[str, str, str | None, str | None]:
        """Main prediction function.

        Returns:
            (params_text, tutorial_text, predicted_audio_path, vital_preset_path)
        """
        # Clean up previous temp files
        import os
        for f in _temp_files:
            try:
                os.unlink(f)
            except OSError:
                pass
        _temp_files.clear()

        if audio_file is None:
            return "Please upload an audio file.", "", None, None

        # Predict parameters (with optional CMA-ES refinement)
        refinement_info = None
        if use_cmaes:
            try:
                params, confidence, refinement_info = pipeline.predict_with_refinement(
                    audio_file,
                )
            except Exception as e:
                log.warning("CMA-ES refinement failed: %s", e)
                params, confidence = pipeline.predict_with_confidence(audio_file)
        else:
            params, confidence = pipeline.predict_with_confidence(audio_file)

        # Format parameters for display
        params_lines = ["## Predicted Parameters\n"]
        if refinement_info and not refinement_info.get("skipped"):
            params_lines.append("### CMA-ES Refinement\n")
            params_lines.append(
                f"- **Initial loss**: {refinement_info['initial_loss']:.4f}"
            )
            params_lines.append(
                f"- **Final loss**: {refinement_info['final_loss']:.4f}"
            )
            params_lines.append(
                f"- **Evaluations**: {refinement_info['n_evals']}"
            )
            params_lines.append(
                f"- **Time**: {refinement_info['elapsed_sec']:.1f}s"
            )
            improved = refinement_info.get("improved", False)
            params_lines.append(
                f"- **Improved**: {'Yes' if improved else 'No'}"
            )
            params_lines.append("")
        for name, value in sorted(params.items()):
            if isinstance(value, float):
                params_lines.append(f"- **{name}**: {value:.4f}")
            else:
                conf = confidence.get(name, 0)
                params_lines.append(f"- **{name}**: {value} (conf: {conf:.0%})")
        params_text = "\n".join(params_lines)

        # Generate tutorial
        if tutorial_gen:
            try:
                tutorial = tutorial_gen.generate(
                    params,
                    confidence=confidence,
                    sound_description=sound_description if sound_description else None,
                )
            except Exception as e:
                log.warning("LLM tutorial generation failed: %s", e)
                tutorial = generate_offline_tutorial(params)
        else:
            tutorial = generate_offline_tutorial(params)

        # Render predicted audio
        predicted_audio_path = None
        try:
            pred_audio = pipeline.render_comparison(params)
            if pred_audio is not None:
                import soundfile as sf

                # Convert to mono for playback
                mono = pred_audio.mean(axis=0)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, mono, pipeline.sample_rate)
                    predicted_audio_path = f.name
                    _temp_files.append(f.name)
        except Exception as e:
            log.warning("Audio rendering failed: %s", e)

        # Export .vital preset
        vital_preset_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".vital", delete=False) as f:
                pipeline.export_vital_preset(params, f.name, preset_name="ML Predicted")
                vital_preset_path = f.name
                _temp_files.append(f.name)
        except Exception as e:
            log.warning("Preset export failed: %s", e)

        return params_text, tutorial, predicted_audio_path, vital_preset_path

    # Build the Gradio interface
    with gr.Blocks(
        title="Vital Inverse Synthesis",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
        ),
    ) as demo:
        gr.Markdown(
            """
            # 🎹 Vital Inverse Synthesis
            
            Upload an audio sample and our ML model will predict the Vital synthesizer 
            parameters to recreate the sound, plus generate a step-by-step tutorial.
            
            **How it works:**
            1. Upload an audio file (WAV, MP3, etc.)
            2. Optionally describe the sound you're trying to recreate
            3. Click "Analyze" to predict parameters
            4. Download the .vital preset and follow the tutorial!
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Input Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                sound_desc = gr.Textbox(
                    label="Sound Description (optional)",
                    placeholder="e.g., 'warm pad with subtle movement'",
                    lines=2,
                )
                cmaes_toggle = gr.Checkbox(
                    label="CMA-ES Refinement (slower, more accurate)",
                    value=False,
                )
                analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")

            with gr.Column(scale=1):
                predicted_audio = gr.Audio(
                    label="Predicted Sound",
                    type="filepath",
                    interactive=False,
                )
                vital_file = gr.File(
                    label="Download .vital Preset",
                    file_types=[".vital"],
                )

        with gr.Row():
            with gr.Column(scale=1):
                params_output = gr.Markdown(label="Parameters")
            with gr.Column(scale=1):
                tutorial_output = gr.Markdown(label="Tutorial")

        # Connect the prediction function
        analyze_btn.click(
            fn=predict_and_analyze,
            inputs=[audio_input, sound_desc, cmaes_toggle],
            outputs=[params_output, tutorial_output, predicted_audio, vital_file],
        )

        gr.Markdown(
            """
            ---
            **Tips:**
            - For best results, use short audio clips (2-4 seconds)
            - Single sustained notes work better than complex sequences
            - The model predicts ~448 Vital parameters from audio
            
            *Built for QHacks 2026*
            """
        )

    return demo


def main():
    """Launch the Gradio demo."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Launch Vital Inverse Synthesis demo")
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda, mps, cpu)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on",
    )
    args = parser.parse_args()

    demo = create_demo(args.checkpoint, args.device)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
