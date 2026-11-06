"""Session options that change what the LLM is told or allowed to do. Shared by the live worker (agent.py) and the
persona harness so evals see the same instructions as a real call: on call dfa187cd a prompt passed every eval but
broke live because only the live session had `expressive=True` (TTS markup instructions)."""

from __future__ import annotations

import os

from livekit.agents import inference
from livekit.plugins import google

LLM_BEHAVIOUR: dict = {
    "expressive": True,  # injects the TTS markup guide into the LLM context (needs a markup-aware TTS)
    "max_tool_steps": 3,
}


def make_llm(model: str, provider: str | None = None) -> google.LLM | inference.LLM:
    """Vertex (default, Cloud credits) or LiveKit Inference (`LLM_PROVIDER=inference`: no Google credentials, spends the
    LiveKit account's Inference credit). Same Gemini model and minimal thinking either way (BUILD_PLAN §2)."""
    if (provider or os.environ.get("LLM_PROVIDER", "vertex")) == "inference":
        return inference.LLM(model if "/" in model else f"google/{model}", extra_kwargs={"reasoning_effort": "minimal"})
    return google.LLM(model=model, thinking_config={"thinking_level": "minimal"})


def make_tts(model: str, voice: str = "") -> inference.TTS:
    return inference.TTS(model, voice=voice) if voice else inference.TTS(model)
