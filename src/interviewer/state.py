"""InterviewState: what has happened, in code, not in the model's memory (BUILD_PLAN §5.5)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from .metadata import DispatchMetadata
from .outcomes import Outcome

Stage = Literal["open", "identity", "knockouts", "core", "candidate_q", "close", "ended"]


@dataclass
class InterviewState:
    meta: DispatchMetadata
    started_at: float | None = None  # set from `clock` on creation
    stage: Stage = "open"
    notice_played: bool = False
    knockouts: dict[str, dict] = field(default_factory=dict)
    answers: dict[str, dict] = field(default_factory=dict)
    followups: list[dict] = field(default_factory=list)
    explicit_outcome: Outcome | None = None
    closed_normally: bool = False
    candidate_joined: bool = False
    interruptions: int = 0
    silence_events: list[str] = field(default_factory=list)
    probed: set[str] = field(default_factory=set)  # questions whose thin answer was sent back for the probe
    guardrail_events: list[dict] = field(default_factory=list)
    # Live calls use wall-clock time; text-mode evals pass a turn clock so a slow model never trips SOFT CLOSE.
    clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.started_at is None:
            self.started_at = self.clock()

    @property
    def elapsed_s(self) -> float:
        return self.clock() - self.started_at

    @property
    def knockout_failed(self) -> bool:
        return any(k["result"] == "fail" for k in self.knockouts.values())

    @property
    def interview_answers(self) -> int:
        return len(self.answers)

    def next_knockout(self) -> str | None:
        """First knockout not yet answered. An `unclear` answer gets exactly one clarify, then moves on."""
        for k in self.meta.plan.knockouts:
            rec = self.knockouts.get(k.id)
            if rec is None or (rec["result"] == "unclear" and rec.get("attempts", 1) < 2):
                return k.id
        return None

    def next_question(self) -> str | None:
        for q in self.meta.plan.questions:
            if q.id not in self.answers:
                return q.id
        return None

    def time_flag(self) -> str:
        tb = self.meta.plan.time_budget_s
        if self.elapsed_s >= tb.get("hard_close", 540):
            return "HARD CLOSE"
        if self.elapsed_s >= tb.get("soft_close", 420):
            return "SOFT CLOSE"
        return "ok"

    def advance_stage(self) -> None:
        """Recompute the stage from facts after every tool call."""
        if self.stage in ("close", "ended"):
            return
        if self.stage in ("open", "identity") and not self.knockouts and not self.answers:
            return
        if self.next_knockout() is not None:
            self.stage = "knockouts"
            return
        core_limit = 2 if self.knockout_failed else len(self.meta.plan.questions)
        if self.interview_answers < core_limit and self.next_question() is not None and self.time_flag() == "ok":
            self.stage = "core"
            return
        self.stage = "candidate_q" if self.time_flag() != "HARD CLOSE" else "close"

    def next_step(self) -> str:
        """The one thing the model should do next, spelled out (the model follows this over its memory)."""
        ko_first = self.next_knockout()
        if self.stage in ("open", "identity"):
            first = next(k for k in self.meta.plan.knockouts if k.id == ko_first) if ko_first else None
            then = f'ask knockout [{first.id}]: "{first.question}", then call mark_knockout' if first else "start the core questions"
            return (f"the opening already confirmed who they are and the role; do not ask again. "
                    f"Once they say now is a good time, {then}")
        if self.stage == "knockouts" and ko_first:
            k = next(k for k in self.meta.plan.knockouts if k.id == ko_first)
            if self.knockouts.get(k.id, {}).get("result") == "unclear":
                return f'clarify knockout [{k.id}] once: "{k.clarify or k.question}", then call mark_knockout again'
            return f'ask knockout [{k.id}]: "{k.question}", then call mark_knockout'
        if self.stage == "core":
            q = next(q for q in self.meta.plan.questions if q.id == self.next_question())
            step = f'ask core question [{q.id}]: "{q.question}", then call record_answer'
            if self.knockout_failed:
                step += " (a knockout failed: at most 2 core questions in total, then candidate questions)"
            return step
        if self.stage == "candidate_q":
            return ("ask what questions they have (at most two). When they have none left, call end_call "
                    "right away; it gives you the goodbye to say, so do not say goodbye before calling it")
        return "call end_call now"

    def render(self) -> str:
        """Layer D: a few lines regenerated after every tool call."""
        ko = ", ".join(f"{k}={v['result']}" for k, v in self.knockouts.items()) or "none yet"
        core_total = len(self.meta.plan.questions)
        nxt = self.next_step()
        return (
            "# CURRENT STATE\n"
            f"STAGE: {self.stage} · ELAPSED: {int(self.elapsed_s // 60)}m{int(self.elapsed_s % 60):02d}s · TIME: {self.time_flag()}\n"
            f"KNOCKOUTS: {ko}\n"
            f"CORE ANSWERED: {self.interview_answers}/{core_total} · NEXT: {nxt}\n"
            f"FOLLOWUPS: {len(self.followups)}"
        )

    def snapshot(self) -> dict:
        return {
            "stage": self.stage, "elapsed_s": round(self.elapsed_s, 1), "knockouts": self.knockouts,
            "answered": list(self.answers), "next": self.next_knockout() or self.next_question(),
            "time": self.time_flag(), "interruptions": self.interruptions,
            "guardrail_events": len(self.guardrail_events),
        }
