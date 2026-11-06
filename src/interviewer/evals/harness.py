"""Persona eval harness (BUILD_PLAN §6, §4.10). Text mode via AgentSession.run.

`run_persona` drives one persona's scripted turns through the real InterviewAgent (tools, state,
llm_node guardrail) and returns what happened. Shipped in the package because the worker runs it for the
role readiness gate; pytest uses it too (tests/evals/harness.py re-exports it). ScriptedLLM keeps the
harness's own tests offline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from livekit.agents import AgentSession, llm
from livekit.agents.llm import LLM, LLMStream
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from interviewer.agent import END_INSTRUCTIONS, InterviewAgent, was_asked
from interviewer.metadata import DispatchMetadata
from interviewer.outcomes import derive_outcome
from interviewer.prompt import opening_line
from interviewer.session_opts import LLM_BEHAVIOUR
from interviewer.state import InterviewState


class TurnClock:
    """Elapsed time for text-mode runs: advances a fixed amount per candidate turn (InterviewState.clock)."""

    def __init__(self, step_s: float = 25.0) -> None:
        self.step_s = step_s
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self) -> None:
        self.t += self.step_s


@dataclass
class PersonaRun:
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    agent_texts: list[str] = field(default_factory=list)
    outcome: str | None = None
    state: InterviewState | None = None


class _NoEndCall(InterviewAgent):
    """Same agent, but end_call only marks the state (no room to delete in text mode)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from livekit.agents import function_tool

        async def end_call() -> str:
            """End the call. Call it once the conversation is over; it tells you what goodbye to say."""
            await self._on_end_call(None)
            return END_INSTRUCTIONS

        self._tools = [t for t in self._tools if getattr(t, "id", "") != "end_call" and "EndCall" not in type(t).__name__]
        self._tools.append(function_tool(end_call))


async def run_persona(meta: DispatchMetadata, persona: dict, model: LLM, tts: Any = None,
                      clock: TurnClock | None = None) -> PersonaRun:
    """`tts`: pass the live TTS (session_opts.make_tts) so expressive markup instructions match a real call.
    Text mode never synthesizes, so no audio is generated. Offline harness tests leave it None.
    `clock`: a TurnClock (default 25 s per turn) so elapsed time does not depend on model latency."""
    clock = clock or TurnClock()
    state = InterviewState(meta, clock=clock)
    state.stage = "identity"
    state.candidate_joined = True
    agent = _NoEndCall(meta, state, events=None)
    ctx = agent.chat_ctx.copy()
    ctx.add_message(role="assistant", content=opening_line(meta))
    await agent.update_chat_ctx(ctx)
    out = PersonaRun(state=state)
    opts = LLM_BEHAVIOUR if tts is not None else {k: v for k, v in LLM_BEHAVIOUR.items() if k != "expressive"}
    async with AgentSession(llm=model, tts=tts, **opts) as session:
        await session.start(agent)
        for turn in persona["turns"]:
            if state.stage == "close":
                break
            clock.advance()
            result = await session.run(user_input=turn)
            for ev in result.events:
                if ev.type == "function_call":
                    try:
                        args = json.loads(ev.item.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    out.tool_calls.append((ev.item.name, args))
                elif ev.type == "message" and ev.item.role == "assistant":
                    out.agent_texts.append(ev.item.text_content or "")
    out.outcome = derive_outcome(explicit=state.explicit_outcome, closed_normally=state.closed_normally,
                                 interview_answers=state.interview_answers, candidate_joined=True).value
    return out


def check(persona: dict, run: PersonaRun) -> list[str]:
    """Return a list of failed assertions (empty = pass)."""
    fails: list[str] = []
    names = [n for n, _ in run.tool_calls]
    if "expect_tools" in persona:
        it = iter(names)
        if not all(any(n == want for n in it) for want in persona["expect_tools"]):
            fails.append(f"tools {names} do not contain ordered {persona['expect_tools']}")
    for want in persona.get("expect_tools_contains", []):
        ok = any(n == want["tool"] and all(a.get(k) == v for k, v in want.get("args", {}).items())
                 for n, a in run.tool_calls)
        if not ok:
            fails.append(f"missing tool call {want}")
    for tool, bounds in persona.get("expect_tool_count", {}).items():
        n = names.count(tool)
        if not bounds.get("min", 0) <= n <= bounds.get("max", n):
            fails.append(f"{tool} called {n} times, expected {bounds}")
    leaks = [e for e in run.state.guardrail_events if e.get("rule") == "tool_text"]
    if leaks or any(re.search(r"default_api|<\s*(?:mark_knockout|record_answer)", t) for t in run.agent_texts):
        fails.append("model wrote a tool call as text instead of calling it")
    spoken = " ".join(run.agent_texts)
    for pat in persona.get("expect_tts_text_not_matching", []):
        if re.search(pat, spoken, re.I):
            fails.append(f"agent said forbidden pattern {pat!r}")
    for pat in persona.get("expect_said", []):
        if not re.search(pat, spoken, re.I):
            fails.append(f"agent never said {pat!r}")
    for line in persona.get("expect_asked", []):  # paraphrase allowed (was_asked overlap)
        if not any(was_asked(t, line) for t in run.agent_texts):
            fails.append(f"agent never asked {line!r}")
    if (lines := persona.get("expect_asked_any")) and not any(was_asked(t, x) for t in run.agent_texts for x in lines):
        fails.append(f"agent asked none of {lines!r}")
    for pat in persona.get("expect_tts_text_matching_each_time", []):
        if not all(re.search(pat, t, re.I) for t in run.agent_texts):
            fails.append(f"not every agent turn matched {pat!r}")
    for kid, want in persona.get("expect_knockouts", {}).items():
        got = run.state.knockouts.get(kid, {})
        if got.get("result") != want["result"]:
            fails.append(f"knockout {kid} ended {got.get('result')!r}, expected {want['result']!r}")
        elif got.get("attempts", 1) < want.get("attempts_min", 0):
            fails.append(f"knockout {kid} recorded {got.get('attempts', 1)} time(s), expected an unclear then a clarify")
        elif (pat := want.get("evidence_not_matching")) and re.search(pat, got.get("evidence", ""), re.I):
            fails.append(f"knockout {kid} evidence {got.get('evidence')!r} matches {pat!r}")
    if "expect_outcome" in persona and run.outcome != persona["expect_outcome"]:
        fails.append(f"outcome {run.outcome} != {persona['expect_outcome']}")
    return fails


class ScriptedLLM(LLM):
    """Offline LLM that replays a list of responses: each item is text or {"tool": name, "args": {...}}."""

    def __init__(self, script: list[Any]) -> None:
        super().__init__()
        self._script = list(script)

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **_) -> LLMStream:
        step = self._script.pop(0) if self._script else "Okay."
        return _ScriptedStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options, step=step)


class _ScriptedStream(LLMStream):
    def __init__(self, llm_: LLM, *, step: Any, **kw) -> None:
        super().__init__(llm_, **kw)
        self._step = step

    async def _run(self) -> None:
        s = self._step
        if isinstance(s, dict):
            self._event_ch.send_nowait(llm.ChatChunk(id="s", delta=llm.ChoiceDelta(
                role="assistant", tool_calls=[llm.FunctionToolCall(name=s["tool"], arguments=json.dumps(s.get("args", {})),
                                                                    call_id=f"call_{s['tool']}")])))
        else:
            for word in s.split(" "):
                self._event_ch.send_nowait(llm.ChatChunk(id="s", delta=llm.ChoiceDelta(role="assistant", content=word + " ")))
