"""Environment configuration for the worker. Secrets come from env (LiveKit Cloud secrets or .env.local)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    events_base_url: str
    events_secret: str
    gemini_live_model: str
    cartesia_voice_id: str
    tts_model: str
    stt_model: str
    kill_switch: bool
    endpointing_max_delay: float = 3.0  # latency experiment knob (BUILD_PLAN §5.1); one variable per run
    standalone_role: str = ""  # pivot-only: interview every caller with this role JSON, no web (BUILD_PLAN §4.4)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            events_base_url=os.environ.get("APP_BASE_URL", "http://localhost:3000"),
            events_secret=os.environ.get("AGENT_EVENTS_SECRET", ""),
            gemini_live_model=os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.5-flash-lite"),
            cartesia_voice_id=os.environ.get("CARTESIA_VOICE_ID", ""),
            tts_model=os.environ.get("TTS_MODEL", "cartesia/sonic-3.6"),
            stt_model=os.environ.get("STT_MODEL", "deepgram/nova-3"),
            kill_switch=os.environ.get("KILL_SWITCH", "0") == "1",
            endpointing_max_delay=float(os.environ.get("ENDPOINTING_MAX_DELAY", "3.0")),
            standalone_role=os.environ.get("STANDALONE_ROLE", ""),
        )
