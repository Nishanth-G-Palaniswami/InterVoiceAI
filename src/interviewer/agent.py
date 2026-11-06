"""InterviewAgent: one Agent driven by InterviewState, with the pre-TTS guardrail in llm_node.

Phase 1 uses a single Agent whose instructions are re-rendered after each tool call (simpler and
more predictable than an AgentTask graph); see BUILD_PLAN §5.5.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any, Literal

from livekit import rtc
from livekit.agents import Agent, ModelSettings, RunContext, function_tool, llm, stt
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.types import FlushSentinel

from .events import EventsClient
from .guardrail import CheckResult, SentenceGate
from .metadata import DispatchMetadata
from .outcomes import Outcome
from .prompt import render_instructions
from .state import InterviewState

log = logging.getLogger("interviewer.agent")

Publish = Callable[[str, dict[str, Any]], Awaitable[None]]


async def _noop_publish(topic: str, data: dict[str, Any]) -> None:
    return None


# EndCallTool returns this to the model, which then speaks the goodbye; the close line lives here, not before the tool.
END_INSTRUCTIONS = ("Say one short goodbye turn. If the interview finished, thank the candidate by first name and say "
                    "the recruiting team reviews every screen within a few business days and will reach out by email. "
                    "Otherwise, say a brief polite goodbye that fits the situation. Ask nothing else.")


_WORD = re.compile(r"[a-z]{4,}")
_STOP = {"this", "that", "your", "have", "with", "does", "what", "about", "from", "they", "there", "right", "work"}


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower())) - _STOP


def _overlap(said: set[str], text: str) -> float:
    words = _words(text)
    return len(words & said) / len(words) if words else 0.0


def was_asked(last_agent_text: str, question: str, clarify: str | None = None,
              others: tuple[str, ...] = ()) -> bool:
    """True if the agent's last turn asked this question (or its clarify line), allowing paraphrase: a quarter of
    its key words appear, and it matches at least as well as any other knockout question. Guards against logging
    a knockout from an answer to the opening or to a different question."""
    said = _words(last_agent_text)
    score = max(_overlap(said, question), _overlap(said, clarify or ""))
    return score >= 0.25 and all(score >= _overlap(said, o) for o in others)


def asked_recently(agent_texts: list[str], question: str, clarify: str | None = None,
                   others: tuple[str, ...] = (), lookback: int = 4) -> bool:
    """`agent_texts` newest first. Walks back past turns that ask no knockout (a clarification such as "Seven p.m.
    to seven a.m. Does that work for you?") to the latest knockout question asked, and checks it is this one."""
    for text in agent_texts[:lookback]:
        if was_asked(text, question, clarify, others):
            return True
        if any(was_asked(text, o) for o in others):
            return False
    return False


_ANSWER_START = re.compile(r"^\W*(yes|yeah|yep|yup|no|nope|nah|sure|correct|i (do|don't|have|haven't|can|can't|am|'m|will|won't))\b", re.I)
_QUESTION_START = re.compile(r"^\W*(what|which|when|where|who|whom|why|how|does|do|is|are|can|could|will|would|should|may)\b", re.I)


def is_question_only(text: str) -> bool:
    """The candidate asked something instead of answering (call dfa187cd, readiness persona knockout_question_back).
    "Yes. What are the hours?" still counts as an answer."""
    t = text.strip()
    if not t or _ANSWER_START.match(t):
        return False
    return t.endswith("?") or bool(_QUESTION_START.match(t))


class InterviewAgent(Agent):
    def __init__(self, meta: DispatchMetadata, state: InterviewState, events: EventsClient | None,
                 publish: Publish = _noop_publish, *, tts_sink: list[str] | None = None) -> None:
        self.meta = meta
        self.istate = state
        self.events = events
        self._publish = publish
        self.tts_sink = tts_sink  # tests: records exactly what reaches synthesis
        self._words: list[tuple[str, float, float]] = []  # final STT words since the last candidate turn (§5.7)
        super().__init__(
            instructions=render_instructions(meta, state),
            tools=[EndCallTool(
                delete_room=True,
                end_instructions=END_INSTRUCTIONS,
                on_tool_called=self._on_end_call,
            )],
        )

    # ---------- delivery signals (BUILD_PLAN §5.7) ----------
    async def stt_node(self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings):
        """Pass STT events through unchanged; keep each final transcript's word timings for the turn record."""
        async for ev in Agent.default.stt_node(self, audio, model_settings):
            if isinstance(ev, stt.SpeechEvent) and ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives:
                self.collect_words(ev.alternatives[0].words or [])
            yield ev

    def collect_words(self, words: list[Any]) -> None:
        for w in words:
            start, end = getattr(w, "start_time", None), getattr(w, "end_time", None)
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                self._words.append((str(w), float(start), float(end)))

    def take_words(self) -> list[list[Any]]:
        """Words of the candidate turn just committed, times relative to its first word (seconds, 2 dp)."""
        words, self._words = self._words, []
        if not words:
            return []
        t0 = words[0][1]
        return [[w, round(a - t0, 2), round(b - t0, 2)] for w, a, b in words]

    # ---------- guardrail ----------
    def _guard(self, r: CheckResult, source: str) -> str:
        if not r.ok:
            ev = {"rule": r.rule, "source": source, "stage": self.istate.stage}
            self.istate.guardrail_events.append(ev)
            if self.events:
                self.events.emit("guardrail_event", ev)
            log.info("guardrail replaced a sentence rule=%s source=%s", r.rule, source)
        return r.text

    async def llm_node(self, chat_ctx: llm.ChatContext, tools: list[llm.Tool],
                       model_settings: ModelSettings) -> AsyncIterable[llm.ChatChunk | str | FlushSentinel]:
        gate = SentenceGate()
        first_delta_at: float | None = None
        try:
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                if isinstance(chunk, llm.ChatChunk):
                    d = chunk.delta
                    if d is None:
                        if chunk.usage is not None:
                            yield chunk
                        continue
                    if d.tool_calls or d.extra:
                        # Tool calls and provider extras (e.g. thought signatures) pass through untouched;
                        # only text is withheld for checking.
                        yield llm.ChatChunk(id=chunk.id, usage=chunk.usage,
                                            delta=llm.ChoiceDelta(role=d.role, content=None,
                                                                  tool_calls=d.tool_calls, extra=d.extra))
                    elif chunk.usage is not None and not d.content:
                        yield chunk
                    text = d.content or ""
                elif isinstance(chunk, str):
                    text = chunk
                else:  # FlushSentinel
                    for r in gate.flush():
                        yield self._guard(r, "llm")
                    yield chunk
                    continue
                if text:
                    if first_delta_at is None:
                        first_delta_at = time.monotonic()
                    for r in gate.feed(text):
                        yield self._guard(r, "llm")
            for r in gate.flush():
                yield self._guard(r, "llm")
        except (asyncio.CancelledError, GeneratorExit):
            gate.drop()  # interrupted: whatever was buffered is never spoken
            raise
        finally:
            if first_delta_at is not None and gate.first_release_at is not None:
                self._last_guardrail_ms = round((gate.first_release_at - first_delta_at) * 1000)

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        """Final check on every speech path (session.say included). Already-checked LLM text passes
        through at each sentence boundary, so this adds no audible delay."""
        gate = SentenceGate()

        async def checked() -> AsyncIterable[str]:
            try:
                async for t in text:
                    for r in gate.feed(t):
                        out = self._guard(r, "tts") if not r.ok else r.text
                        if self.tts_sink is not None:
                            self.tts_sink.append(out)
                        yield out
                for r in gate.flush():
                    out = self._guard(r, "tts") if not r.ok else r.text
                    if self.tts_sink is not None:
                        self.tts_sink.append(out)
                    yield out
            except (asyncio.CancelledError, GeneratorExit):
                gate.drop()
                raise

        return Agent.default.tts_node(self, checked(), model_settings)

    # ---------- state plumbing ----------
    def _last_user_text(self) -> str:
        for item in reversed(self.chat_ctx.items):
            if getattr(item, "type", "") == "message" and item.role == "user":
                return item.text_content or ""
        return ""

    def _agent_texts(self) -> list[str]:
        """Agent turns, newest first."""
        return [item.text_content or "" for item in reversed(self.chat_ctx.items)
                if getattr(item, "type", "") == "message" and item.role == "assistant"]

    async def _refresh(self) -> None:
        self.istate.advance_stage()
        await self.update_instructions(render_instructions(self.meta, self.istate))
        await self._publish("interview.state", self.istate.snapshot())

    def _emit(self, type_: str, payload: dict[str, Any], key: str | None = None) -> None:
        if self.events:
            self.events.emit(type_, payload, key=key)

    async def _on_end_call(self, ev: Any) -> None:
        if self.istate.explicit_outcome is None:
            self.istate.closed_normally = True
        self.istate.stage = "close"
        await self._publish("interview.state", self.istate.snapshot())

    # ---------- tools ----------
    @function_tool
    async def mark_knockout(self, context: RunContext, knockout_id: str,
                            result: Literal["pass", "fail", "unclear"], evidence: str) -> str:
        """Record the outcome of a knockout question right after the candidate answers it.

        Args:
            knockout_id: The knockout id from the plan, for example ko_license.
            result: pass, fail, or unclear.
            evidence: A short quote or paraphrase of what the candidate said.
        """
        if knockout_id not in self.meta.plan.knockout_ids():
            return f"Unknown knockout_id {knockout_id!r}. Valid ids: {', '.join(self.meta.plan.knockout_ids())}."
        k = next(k for k in self.meta.plan.knockouts if k.id == knockout_id)
        others = tuple(o.question for o in self.meta.plan.knockouts if o.id != knockout_id)
        if not asked_recently(self._agent_texts(), k.question, k.clarify, others):
            return (f"Not recorded: you have not asked [{knockout_id}] yet. Ask it now, \"{k.question}\", and call "
                    "mark_knockout only after the candidate answers it.")
        if is_question_only(self._last_user_text()):
            return (f"Not recorded: the candidate asked a question instead of answering [{knockout_id}]. Answer it from "
                    "the approved facts in one sentence, then ask the knockout again and wait for their answer.")
        attempts = self.istate.knockouts.get(knockout_id, {}).get("attempts", 0) + 1
        self.istate.knockouts[knockout_id] = {"result": result, "evidence": evidence[:500], "attempts": attempts}
        self._emit("tool_mark_knockout", {"knockout_id": knockout_id, "result": result, "evidence": evidence[:500],
                                          "attempts": attempts}, key=f"{knockout_id}:{attempts}")
        await self._refresh()
        return "Recorded. Do not tell the candidate the result. Continue with the next step in CURRENT STATE."

    @function_tool
    async def record_answer(self, context: RunContext, question_id: str, summary: str,
                            quality_hint: Literal["strong", "adequate", "thin", "no_answer"]) -> str:
        """Record the candidate's answer to a core question after they finish (and after the probe, if used).

        Args:
            question_id: The core question id from the plan, for example q_judgment.
            summary: One factual sentence summarizing what they said. No judgement words.
            quality_hint: strong, adequate, thin, or no_answer.
        """
        if question_id not in self.meta.plan.question_ids():
            return f"Unknown question_id {question_id!r}. Valid ids: {', '.join(self.meta.plan.question_ids())}."
        q = next(q for q in self.meta.plan.questions if q.id == question_id)
        texts = self._agent_texts()
        if question_id not in self.istate.probed and self.istate.time_flag() == "ok":
            if quality_hint in ("thin", "no_answer") and q.probe and not any(was_asked(t, q.probe) for t in texts[:2]):
                # Call e17a0efa: the model recorded two thin answers without the probe the prompt asks for. A dodge
                # ("I don't remember") gets the same one follow-up (playbook: one redirect, then move on).
                self.istate.probed.add(question_id)
                return (f"Not recorded yet: the answer to [{question_id}] is thin or missing. Ask the probe once now, \"{q.probe}\", "
                        "wait for their answer, then call record_answer again. Do not probe a second time.")
            if texts and (was_asked(texts[0], q.question) or (q.short and was_asked(texts[0], q.short))):
                # BUILD_PLAN §5.7: a read answer is fluent; one follow-up about a detail they said is what a script
                # cannot anticipate. Strong answers get it too, once.
                self.istate.probed.add(question_id)
                return (f"Not recorded yet: before recording [{question_id}], ask one short follow-up about a specific "
                        "detail they just mentioned (what exactly they did or said, or what happened next). Do not repeat "
                        "the question. Wait for their answer, then call record_answer again. If instead they said they "
                        "want to stop, do not follow up: ask once whether they want to stop the interview here.")
        self.istate.answers[question_id] = {"summary": summary[:800], "quality_hint": quality_hint}
        self._emit("tool_record_answer", {"question_id": question_id, "summary": summary[:800],
                                          "quality_hint": quality_hint}, key=question_id)
        await self._refresh()
        return "Recorded. Acknowledge briefly and continue with the next step in CURRENT STATE."

    @function_tool
    async def create_followup(self, context: RunContext,
                              kind: Literal["visa", "human_callback", "accessibility", "question", "language", "other"],
                              note: str) -> str:
        """Note something the recruiting team must follow up on (an unanswered question, a callback, a request).

        Args:
            kind: The type of follow-up.
            note: What the team needs to know, in one sentence.
        """
        item = {"kind": kind, "note": note[:500]}
        self.istate.followups.append(item)
        self._emit("tool_followup", item)
        await self._refresh()
        return "Noted for the team. Tell the candidate the team will follow up, then continue."

    @function_tool
    async def request_reschedule(self, context: RunContext) -> str:
        """The candidate says now is not a good time or asks to reschedule."""
        self.istate.explicit_outcome = Outcome.RESCHEDULE_REQUESTED
        self._emit("tool_reschedule", {})
        await self._refresh()
        return "Tell them they'll get an email to pick a new time, thank them, then call end_call."

    @function_tool
    async def recording_refused(self, context: RunContext) -> str:
        """The candidate objects to the call being recorded."""
        self.istate.explicit_outcome = Outcome.RECORDING_REFUSED
        self._emit("recording_refused", {})
        await self._refresh()
        return ("Recording is being stopped. Say you can't continue the screen without recording but the team "
                "will reach out directly, and ask what time of day is best for a call back. When they answer, call "
                "create_followup with kind human_callback and their preferred time, then call end_call.")

    @function_tool
    async def wrong_person(self, context: RunContext) -> str:
        """The person on the line is not the candidate and the candidate is not available."""
        self.istate.explicit_outcome = Outcome.WRONG_PERSON
        self._emit("tool_followup", {"kind": "other", "note": "Wrong person answered."})
        await self._refresh()
        return "Thank them politely and call end_call. Do not share any details about the application."

    @function_tool
    async def escalate_to_human(self, context: RunContext, reason: Literal["hostile", "distress", "unsafe", "other"],
                                note: str = "") -> str:
        """Hand the call to the human team: repeated abuse, distress, or anything unsafe.

        Args:
            reason: hostile, distress, unsafe, or other.
            note: One sentence of context.
        """
        if reason == "hostile":
            self.istate.explicit_outcome = Outcome.TERMINATED_HOSTILE
        self._emit("escalate", {"reason": reason, "note": note[:500]})
        await self._refresh()
        return "Say one short polite closing line, then call end_call."


def turn_record(item: Any, *, guardrail_ms: int | None = None) -> dict[str, Any] | None:
    """ChatMessage → `turn` event payload (BUILD_PLAN §4.7, §5.1)."""
    if getattr(item, "type", None) != "message" or item.role not in ("user", "assistant"):
        return None
    text = item.text_content or ""
    m = item.metrics or {}
    metrics = {k: (round(v * 1000) if isinstance(v, (int, float)) else None) for k, v in {
        "transcription_delay_ms": m.get("transcription_delay"),
        "end_of_turn_delay_ms": m.get("end_of_turn_delay"),
        "llm_ttft_ms": m.get("llm_node_ttft"),
        "tts_ttfb_ms": m.get("tts_node_ttfb"),
        "e2e_ms": m.get("e2e_latency"),
        "playback_latency_ms": m.get("playback_latency"),
    }.items()}
    metrics["guardrail_ms"] = guardrail_ms if item.role == "assistant" else None
    return {
        "turn_id": item.id,
        "role": "agent" if item.role == "assistant" else "candidate",
        "text": text,
        "heard_text": text if item.role == "assistant" else None,  # SDK truncates interrupted text to what played
        "final": True,
        "interrupted": bool(item.interrupted),
        "confidence": item.transcript_confidence,
        "started_at": m.get("started_speaking_at"),
        "ended_at": m.get("stopped_speaking_at"),
        "metrics": metrics,
    }


def dumps(d: dict[str, Any]) -> str:
    return json.dumps(d, separators=(",", ":"), default=str)
