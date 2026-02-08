"""LLM-powered tutorial generation using Anthropic Claude API."""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# Parameter groupings for structured tutorial generation
PARAM_GROUPS = {
    "oscillators": {
        "title": "Oscillators",
        "description": "Sound sources that generate the raw waveform",
        "prefixes": ["osc_1", "osc_2", "osc_3"],
    },
    "filters": {
        "title": "Filters",
        "description": "Shape the harmonic content of the sound",
        "prefixes": ["filter_1", "filter_2", "filter_fx"],
    },
    "envelopes": {
        "title": "Envelopes",
        "description": "Control how parameters change over time",
        "prefixes": ["env_1", "env_2", "env_3", "env_4", "env_5", "env_6"],
    },
    "lfos": {
        "title": "LFOs",
        "description": "Low-frequency oscillators for modulation",
        "prefixes": ["lfo_1", "lfo_2", "lfo_3", "lfo_4", "lfo_5", "lfo_6", "lfo_7", "lfo_8"],
    },
    "effects": {
        "title": "Effects",
        "description": "Post-processing effects",
        "prefixes": ["chorus", "compressor", "delay", "distortion", "eq", "flanger", "phaser", "reverb"],
    },
    "global": {
        "title": "Global Settings",
        "description": "Overall synth settings",
        "prefixes": ["volume", "polyphony", "portamento", "pitch_bend", "voice", "macro_control", "beats_per_minute", "stereo"],
    },
    "random_generators": {
        "title": "Random Generators",
        "description": "Random modulation sources",
        "prefixes": ["random_1", "random_2", "random_3", "random_4"],
    },
}

# Human-readable names for common parameters
PARAM_DISPLAY_NAMES = {
    "osc_1_level": "Oscillator 1 Level",
    "osc_1_pan": "Oscillator 1 Pan",
    "osc_1_tune": "Oscillator 1 Tune (semitones)",
    "osc_1_unison_detune": "Oscillator 1 Unison Detune",
    "osc_1_unison_voices": "Oscillator 1 Unison Voices",
    "osc_1_wavetable_position": "Oscillator 1 Wavetable Position",
    "filter_1_cutoff": "Filter 1 Cutoff",
    "filter_1_resonance": "Filter 1 Resonance",
    "filter_1_drive": "Filter 1 Drive",
    "filter_1_model": "Filter 1 Model",
    "env_1_attack": "Envelope 1 Attack",
    "env_1_decay": "Envelope 1 Decay",
    "env_1_sustain": "Envelope 1 Sustain",
    "env_1_release": "Envelope 1 Release",
    "reverb_dry_wet": "Reverb Dry/Wet",
    "reverb_size": "Reverb Size",
    "delay_dry_wet": "Delay Dry/Wet",
    "delay_tempo": "Delay Tempo",
    "chorus_dry_wet": "Chorus Dry/Wet",
    "distortion_drive": "Distortion Drive",
    "volume": "Master Volume",
}


def _get_display_name(param_name: str) -> str:
    """Get human-readable display name for a parameter."""
    if param_name in PARAM_DISPLAY_NAMES:
        return PARAM_DISPLAY_NAMES[param_name]
    # Convert snake_case to Title Case
    return param_name.replace("_", " ").title()


def _group_parameters(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Group parameters by their module prefix."""
    grouped: dict[str, dict[str, Any]] = {group: {} for group in PARAM_GROUPS}
    grouped["other"] = {}

    for name, value in params.items():
        if name.startswith("_"):
            continue

        matched = False
        for group_key, group_info in PARAM_GROUPS.items():
            for prefix in group_info["prefixes"]:
                if name.startswith(prefix):
                    grouped[group_key][name] = value
                    matched = True
                    break
            if matched:
                break

        if not matched:
            grouped["other"][name] = value

    return grouped


def _format_value(name: str, value: Any) -> str:
    """Format a parameter value for display."""
    if isinstance(value, float):
        # For normalized values, convert to percentage
        if 0 <= value <= 1:
            return f"{value * 100:.0f}%"
        return f"{value:.2f}"
    return str(value)


def generate_tutorial_prompt(
    params: dict[str, Any],
    confidence: dict[str, float] | None = None,
    sound_description: str | None = None,
) -> str:
    """Generate the prompt for Claude to create a tutorial.

    Args:
        params: Predicted parameter dict.
        confidence: Optional confidence scores for categorical params.
        sound_description: Optional user description of the target sound.

    Returns:
        Formatted prompt string for the LLM.
    """
    grouped = _group_parameters(params)

    param_sections = []
    for group_key, group_params in grouped.items():
        if not group_params:
            continue

        if group_key in PARAM_GROUPS:
            title = PARAM_GROUPS[group_key]["title"]
        else:
            title = "Other Parameters"

        lines = [f"### {title}"]
        for name, value in sorted(group_params.items()):
            display_name = _get_display_name(name)
            formatted_value = _format_value(name, value)
            line = f"- **{display_name}**: {formatted_value}"
            if confidence and name in confidence:
                line += f" (confidence: {confidence[name]*100:.0f}%)"
            lines.append(line)

        param_sections.append("\n".join(lines))

    params_text = "\n\n".join(param_sections)

    sound_context = ""
    if sound_description:
        sound_context = f"\n\nThe user described the sound they want to recreate as: \"{sound_description}\"\n"

    prompt = f"""You are an expert sound designer and Vital synthesizer specialist. 
A machine learning model has analyzed an audio sample and predicted the Vital synthesizer 
parameters that would recreate this sound.{sound_context}

## Predicted Parameters

{params_text}

## Your Task

Create a step-by-step tutorial for recreating this sound in Vital. The tutorial should:

1. **Start with the foundation**: Explain which oscillator settings to configure first
2. **Add filtering**: Describe the filter settings and their effect on the sound
3. **Shape the dynamics**: Explain envelope settings for amplitude and timbral changes
4. **Add modulation**: If LFOs are active, explain what they're modulating and why
5. **Apply effects**: Describe any effects and their contribution to the final sound
6. **Fine-tuning tips**: Suggest adjustments the user might want to try

Format the tutorial with clear headings and numbered steps. Use plain language that 
a beginner to intermediate Vital user would understand. Explain *why* each setting 
contributes to the overall sound, not just what to set it to.

Keep the tutorial concise but informative - aim for 400-600 words.
"""
    return prompt


class TutorialGenerator:
    """Generate step-by-step tutorials from predicted Vital parameters."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the tutorial generator.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

    def generate(
        self,
        params: dict[str, Any],
        confidence: dict[str, float] | None = None,
        sound_description: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> str:
        """Generate a tutorial from predicted parameters.

        Args:
            params: Predicted parameter dict from inference pipeline.
            confidence: Optional confidence scores for categorical params.
            sound_description: Optional description of the target sound.
            model: Claude model to use.

        Returns:
            Generated tutorial text.
        """
        prompt = generate_tutorial_prompt(params, confidence, sound_description)

        message = self.client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        return message.content[0].text

    def generate_quick_summary(
        self,
        params: dict[str, Any],
        model: str = "claude-sonnet-4-20250514",
    ) -> str:
        """Generate a short 2-3 sentence summary of the sound.

        Args:
            params: Predicted parameter dict.
            model: Claude model to use.

        Returns:
            Short summary text.
        """
        grouped = _group_parameters(params)

        # Create a condensed param summary
        key_params = []
        for group_key in ["oscillators", "filters", "effects"]:
            for name, value in list(grouped.get(group_key, {}).items())[:3]:
                key_params.append(f"{_get_display_name(name)}: {_format_value(name, value)}")

        params_text = ", ".join(key_params[:10])

        prompt = f"""Based on these Vital synthesizer parameters, write a 2-3 sentence 
description of what this sound would sound like. Be specific about the character and 
texture of the sound.

Key parameters: {params_text}

Keep your response under 75 words and focus on describing the sonic character."""

        message = self.client.messages.create(
            model=model,
            max_tokens=200,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        return message.content[0].text


def generate_offline_tutorial(params: dict[str, Any]) -> str:
    """Generate a basic tutorial without LLM API (template-based).

    Falls back to this when Anthropic API is not available.

    Args:
        params: Predicted parameter dict.

    Returns:
        Template-based tutorial text.
    """
    grouped = _group_parameters(params)

    sections = ["# How to Recreate This Sound in Vital\n"]

    # Oscillators section
    if grouped.get("oscillators"):
        sections.append("## Step 1: Set Up the Oscillators\n")
        for name, value in sorted(grouped["oscillators"].items()):
            display_name = _get_display_name(name)
            formatted_value = _format_value(name, value)
            sections.append(f"- Set **{display_name}** to {formatted_value}")
        sections.append("")

    # Filters section
    if grouped.get("filters"):
        sections.append("## Step 2: Configure the Filter\n")
        for name, value in sorted(grouped["filters"].items()):
            display_name = _get_display_name(name)
            formatted_value = _format_value(name, value)
            sections.append(f"- Set **{display_name}** to {formatted_value}")
        sections.append("")

    # Envelopes section
    if grouped.get("envelopes"):
        sections.append("## Step 3: Shape the Envelopes\n")
        for name, value in sorted(grouped["envelopes"].items()):
            display_name = _get_display_name(name)
            formatted_value = _format_value(name, value)
            sections.append(f"- Set **{display_name}** to {formatted_value}")
        sections.append("")

    # Effects section
    if grouped.get("effects"):
        sections.append("## Step 4: Add Effects\n")
        for name, value in sorted(grouped["effects"].items()):
            display_name = _get_display_name(name)
            formatted_value = _format_value(name, value)
            sections.append(f"- Set **{display_name}** to {formatted_value}")
        sections.append("")

    # LFOs section
    if grouped.get("lfos"):
        sections.append("## Step 5: Configure LFOs\n")
        for name, value in sorted(grouped["lfos"].items()):
            display_name = _get_display_name(name)
            formatted_value = _format_value(name, value)
            sections.append(f"- Set **{display_name}** to {formatted_value}")
        sections.append("")

    # Global section
    if grouped.get("global"):
        sections.append("## Step 6: Global Settings\n")
        for name, value in sorted(grouped["global"].items()):
            display_name = _get_display_name(name)
            formatted_value = _format_value(name, value)
            sections.append(f"- Set **{display_name}** to {formatted_value}")
        sections.append("")

    sections.append("---\n")
    sections.append("*This is an auto-generated tutorial. Adjust parameters to taste.*")

    return "\n".join(sections)
