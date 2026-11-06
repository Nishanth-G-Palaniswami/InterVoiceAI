"""Model presets for the voice pipeline.

Every model here runs through LiveKit Inference, so your LiveKit Cloud keys are
the only credentials you need. No Deepgram, Google or Cartesia accounts.

Pick a preset with MODEL_PRESET in .env.local, or edit one below.
Full model lists: https://docs.livekit.io/agents/models/
Prices:           https://livekit.com/pricing/inference

Rough cost per minute of conversation (our estimate, not LiveKit's):
  budget    ~ $0.015 / min   -> the free $2.50 credit lasts ~150 min
  quality   ~ $0.03  / min   -> the free $2.50 credit lasts ~80 min
Text-to-speech is usually the biggest line item.
"""

import os
from dataclasses import dataclass

from livekit.agents import inference


@dataclass
class Models:
    stt: inference.STT
    llm: inference.LLM
    tts: inference.TTS
    # Expressive mode lets the LLM add delivery tags (pacing, emotion) that the
    # TTS renders. Only turn it on for TTS models that support markup.
    expressive: bool


def budget() -> Models:
    """LiveKit's starter defaults. Cheapest, and good enough to win with."""
    return Models(
        stt=inference.STT(model="assemblyai/universal-3-6-pro", language="en"),
        llm=inference.LLM(model="google/gemma-4-31b-it"),
        tts=inference.TTS(
            model="fishaudio/s2.1-pro", voice="fa4c9eb3dccc4806b382b40d61c6b10a"
        ),
        expressive=True,
    )


def quality() -> Models:
    """Deepgram + Gemini + Cartesia. About twice the cost of `budget`."""
    return Models(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        llm=inference.LLM(model="google/gemini-3.1-flash-lite"),
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        expressive=False,
    )


def interview() -> Models:
    """Deepgram Nova-3 for per-word timings and filler words, budget LLM and TTS.

    The InterviewAgent reads word timings off the final transcripts to build its
    delivery signals (pauses, pace, fillers). AssemblyAI offers no filler-word
    switch, so this preset swaps only the STT and keeps the cheap defaults.
    """
    return Models(
        stt=inference.STT(
            model="deepgram/nova-3",
            language="en",
            extra_kwargs={
                "interim_results": True,
                "smart_format": False,  # keeps "um" / "uh" as plain tokens
                "filler_words": True,
                "punctuate": True,
            },
        ),
        llm=inference.LLM(model="google/gemma-4-31b-it"),
        tts=inference.TTS(
            model="fishaudio/s2.1-pro", voice="fa4c9eb3dccc4806b382b40d61c6b10a"
        ),
        expressive=True,
    )


PRESETS = {"budget": budget, "quality": quality, "interview": interview}


def load_models() -> Models:
    name = os.getenv("MODEL_PRESET", "budget").strip().lower()
    if name not in PRESETS:
        raise ValueError(
            f"Unknown MODEL_PRESET '{name}'. Choose one of: {', '.join(PRESETS)}"
        )
    return PRESETS[name]()
