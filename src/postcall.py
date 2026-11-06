"""Post-call hook for ANY agent in this repo: records the events the cheat-detection analysis needs.

Two pieces, both optional and independent of which agent answers the call:

  1. `WordTimingsMixin`: add it to your Agent class so each candidate turn carries Deepgram word timings
     (needed for the voice-side signals). It overrides `stt_node` only to copy the final words out.

        class MyAgent(WordTimingsMixin, Agent): ...

  2. `attach_postcall(session, agent, call_id, answer_tools={"record_answer"})`: writes
     `calls/<call_id>.events.jsonl` with `turn` events (text, timings, latency metrics, words) and a
     `tool_record_answer` event each time one of your answer-recording tools runs, plus `call_started` /
     `call_ended`. Then: `uv run python evals/ai_detect/analyze_call.py calls/<call_id>.events.jsonl`
     or `uv run python evals/ai_detect/dashboard.py --serve 8765 --open` for the live screen.

Nothing here touches what the agent says. The analysis is for the recruiter after (or during) the call.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import pathlib
from collections.abc import AsyncIterable
from typing import Any

from livekit import rtc
from livekit.agents import Agent, AgentSession, ModelSettings, stt

log = logging.getLogger("postcall")


class WordTimingsMixin:
    """Keep each final transcript's word timings so the turn record can carry them."""

    _pc_words: list[tuple[str, float, float]]

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ):  # type: ignore[override]
        if not hasattr(self, "_pc_words"):
            self._pc_words = []
        async for ev in Agent.default.stt_node(self, audio, model_settings):  # type: ignore[arg-type]
            if (
                isinstance(ev, stt.SpeechEvent)
                and ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT
                and ev.alternatives
            ):
                for w in ev.alternatives[0].words or []:
                    a, b = getattr(w, "start_time", None), getattr(w, "end_time", None)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        self._pc_words.append((str(w), float(a), float(b)))
            yield ev

    def take_words(self) -> list[list[Any]]:
        words, self._pc_words = getattr(self, "_pc_words", []), []
        if not words:
            return []
        t0 = words[0][1]
        return [[w, round(a - t0, 2), round(b - t0, 2)] for w, a, b in words]


class _Dump:
    def __init__(self, call_id: str, dump_dir: pathlib.Path) -> None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        self.path = dump_dir / f"{call_id}.events.jsonl"
        self.call_id = call_id
        self.seq = 0

    def emit(self, type_: str, payload: dict[str, Any]) -> None:
        self.seq += 1
        ev = {
            "schema_version": 1,
            "call_id": self.call_id,
            "seq": self.seq,
            "type": type_,
            "occurred_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "payload": payload,
        }
        try:
            with self.path.open("a") as f:
                f.write(json.dumps(ev, separators=(",", ":"), default=str) + "\n")
        except OSError as e:
            log.warning("postcall dump failed: %s", e)


def _turn_record(item: Any) -> dict[str, Any] | None:
    if getattr(item, "type", None) != "message" or item.role not in (
        "user",
        "assistant",
    ):
        return None
    m = item.metrics or {}
    ms = {
        k: (round(v * 1000) if isinstance(v, (int, float)) else None)
        for k, v in {
            "transcription_delay_ms": m.get("transcription_delay"),
            "end_of_turn_delay_ms": m.get("end_of_turn_delay"),
            "llm_ttft_ms": m.get("llm_node_ttft"),
            "tts_ttfb_ms": m.get("tts_node_ttfb"),
            "e2e_ms": m.get("e2e_latency"),
        }.items()
    }
    return {
        "turn_id": item.id,
        "role": "agent" if item.role == "assistant" else "candidate",
        "text": item.text_content or "",
        "interrupted": bool(item.interrupted),
        "started_at": m.get("started_speaking_at"),
        "ended_at": m.get("stopped_speaking_at"),
        "metrics": ms,
    }


def attach_postcall(
    session: AgentSession,
    agent: Agent,
    call_id: str,
    *,
    answer_tools: set[str] = frozenset({"record_answer"}),
    knockout_tools: set[str] = frozenset({"mark_knockout"}),
    dump_dir: str | os.PathLike | None = None,
    meta: dict[str, Any] | None = None,
) -> _Dump:
    """Register the listeners. Call before `session.start`. Returns the dump (emit `call_ended` yourself on shutdown
    if you want outcome fields; a bare `call_ended` is written when the session closes)."""
    dump = _Dump(
        call_id, pathlib.Path(dump_dir or os.environ.get("EVENTS_DUMP_DIR", "calls"))
    )
    dump.emit("call_started", {"mode": "inbound", **(meta or {})})
    count = {"turns": 0, "interruptions": 0}

    @session.on("conversation_item_added")
    def _item(ev: Any) -> None:
        rec = _turn_record(ev.item)
        if rec is None:
            return
        count["turns"] += 1
        rec["seq"] = count["turns"]
        if rec["role"] == "candidate" and hasattr(agent, "take_words"):
            rec["words"] = agent.take_words()  # type: ignore[attr-defined]
        if rec["interrupted"]:
            count["interruptions"] += 1
        dump.emit("turn", rec)

    @session.on("function_tools_executed")
    def _tools(ev: Any) -> None:
        for call in ev.function_calls:
            try:
                args = (
                    json.loads(call.arguments)
                    if isinstance(call.arguments, str)
                    else dict(call.arguments or {})
                )
            except (TypeError, ValueError):
                args = {"raw": str(call.arguments)}
            if call.name in answer_tools:
                dump.emit(
                    "tool_record_answer",
                    {
                        "question_id": str(
                            args.get("question_id")
                            or args.get("question_number")
                            or call.name
                        ),
                        **{k: str(v)[:800] for k, v in args.items()},
                    },
                )
            elif call.name in knockout_tools:
                dump.emit(
                    "tool_mark_knockout", {k: str(v)[:500] for k, v in args.items()}
                )

    @session.on("close")
    def _close(ev: Any) -> None:
        dump.emit(
            "call_ended",
            {
                "outcome": "completed",
                "reason": getattr(ev, "reason", None),
                "interruptions": count["interruptions"],
                "transcript_rev": count["turns"],
            },
        )

    return dump
