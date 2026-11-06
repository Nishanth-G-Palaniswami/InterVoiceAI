"""Delivery signals for "reading a prepared script" (reference repo BUILD_PLAN §5.7), ported from apps/web/src/lib/delivery.ts.

Pure functions over the call's agent events so they run offline in the eval harness and in the standalone
post-call report. Observations for the recruiter only: never scored, never an auto-reject, never raised live.
Thresholds are UNCALIBRATED until a calibration session (10 read answers, 10 spontaneous); evals/ai_detect measures them.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

Word = list  # [text, start_s, end_s], times relative to the turn's first word

THRESHOLDS = {
    "min_words": 40,
    "max_fillers_per100": 1.0,
    "max_pauses_per100": 1.0,
    "min_wpm": 150,
    "follow_up_drop_ratio": 3.0,
    "pause_s": 0.4,
    "min_signals": 3,
}

FILLERS = {"um", "uh", "erm", "er", "hmm", "mm", "uhm", "umm", "uhh", "ah"}
_PUNCT_END = re.compile(r"[.?!,;:]$")


def _norm(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower())


@dataclass
class Features:
    words: int = 0
    seconds: float = 0.0
    wpm: int | None = None
    fillers_per100: float = 0.0
    pauses_per100: float = 0.0
    restarts_per100: float = 0.0
    disfluency_per100: float = 0.0


@dataclass
class CandTurn:
    turn_id: str
    text: str
    words: list[Word] = field(default_factory=list)
    started_at: float | None = None


@dataclass
class AnswerDelivery:
    question_id: str
    onset_ms: int | None
    main: Features
    follow_up: Features | None
    signals: list[str]
    sounds_read: bool
    eligible: bool
    turn_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def features(turns: list[CandTurn]) -> Features:
    """Fillers, mid-sentence pauses (>= pause_s and not after . ? ! , ; :), restarts (immediate repeats, "I mean")."""
    T = THRESHOLDS
    words = 0
    seconds = 0.0
    fillers = pauses = restarts = 0
    for t in turns:
        ws = t.words
        if not ws:
            continue
        # Deepgram word tokens carry no punctuation; borrow it from the punctuated text when the counts line up.
        tokens = t.text.strip().split()
        punct = [bool(_PUNCT_END.search(x)) for x in tokens] if len(tokens) == len(ws) else None
        words += len(ws)
        seconds += max(0.0, float(ws[-1][2]) - float(ws[0][1]))
        for i, w_ in enumerate(ws):
            w = _norm(str(w_[0]))
            if w in FILLERS:
                fillers += 1
            if i > 0:
                prev = _norm(str(ws[i - 1][0]))
                if w and w == prev and w not in FILLERS:
                    restarts += 1
                if prev == "i" and w == "mean":
                    restarts += 1
                gap = float(w_[1]) - float(ws[i - 1][2])
                if gap >= T["pause_s"] and not (punct and punct[i - 1]):
                    pauses += 1

    def per100(n: int) -> float:
        return round(n / words * 1000) / 10 if words else 0.0

    return Features(
        words=words,
        seconds=round(seconds * 10) / 10,
        wpm=round(words / seconds * 60) if seconds > 0 else None,
        fillers_per100=per100(fillers),
        pauses_per100=per100(pauses),
        restarts_per100=per100(restarts),
        disfluency_per100=per100(fillers + pauses + restarts),
    )


def is_question(text: str) -> bool:
    """An agent turn that opens an answer: any question or prompt. Short acks and nudges ("Got it.", "Take your time.")
    do not count. Not "ends with ?": an interrupted question has no "?" and "Tell me about a time ..." is a prompt."""
    return "?" in text or len(text.strip().split()) >= 6


def judge(m: Features, f: Features | None) -> tuple[list[str], bool, bool]:
    """(signals, sounds_read, eligible)."""
    T = THRESHOLDS
    signals: list[str] = []
    if m.words == 0:
        return signals, False, False
    if m.fillers_per100 < T["max_fillers_per100"]:
        signals.append("almost no fillers")
    if m.pauses_per100 < T["max_pauses_per100"]:
        signals.append("no mid-sentence pauses")
    if m.restarts_per100 == 0:
        signals.append("no restarts")
    if m.wpm is not None and m.wpm >= T["min_wpm"]:
        signals.append(f"steady fast pace ({m.wpm} wpm)")
    if f and f.words >= 8 and f.disfluency_per100 >= T["follow_up_drop_ratio"] * max(m.disfluency_per100, 1):
        signals.append(f"follow-up much less fluent ({f.disfluency_per100} vs {m.disfluency_per100} per 100 words)")
    eligible = m.words >= T["min_words"]
    return signals, eligible and len(signals) >= T["min_signals"], eligible


def answer_delivery(events: list[dict[str, Any]]) -> list[AnswerDelivery]:
    """Split a call's events into answers: candidate turns between an agent question and the `tool_record_answer`
    that closes it. Turns after a second agent question inside the same answer are the follow-up.

    `events` are the agent's event dicts (`type`, `payload`, optional `seq`), as emitted by EventsClient.
    """
    out: list[AnswerDelivery] = []
    main: list[CandTurn] = []
    follow: list[CandTurn] = []
    asked_at: float | None = None
    onset_ms: int | None = None
    questions_asked = 0

    def reset() -> None:
        nonlocal main, follow, asked_at, onset_ms, questions_asked
        main, follow, asked_at, onset_ms, questions_asked = [], [], None, None, 0

    for ev in sorted(events, key=lambda e: e.get("seq") or 0):
        p = ev.get("payload") or {}
        typ = ev.get("type")
        if typ == "turn" and p.get("role") == "agent":
            if is_question(str(p.get("text") or "")):
                # Before the candidate has spoken, every agent prompt is (still) the main question, so a transition
                # sentence or a re-ask does not push the real answer into the follow-up bucket.
                if main:
                    questions_asked = max(questions_asked, 1) + 1
                else:
                    questions_asked = 1
                    asked_at = p["ended_at"] if isinstance(p.get("ended_at"), (int, float)) else None
        elif typ == "turn" and p.get("role") == "candidate":
            t = CandTurn(
                turn_id=str(p.get("turn_id")),
                text=str(p.get("text") or ""),
                words=list(p.get("words") or []),
                started_at=p["started_at"] if isinstance(p.get("started_at"), (int, float)) else None,
            )
            if questions_asked <= 1:
                if not main and asked_at is not None and t.started_at is not None:
                    onset_ms = round((t.started_at - asked_at) * 1000)
                main.append(t)
            else:
                follow.append(t)
        elif typ == "tool_mark_knockout":
            reset()
        elif typ == "tool_record_answer":
            m = features(main)
            f = features(follow) if follow else None
            signals, sounds_read, eligible = judge(m, f)
            out.append(AnswerDelivery(
                question_id=str(p.get("question_id") or "?"), onset_ms=onset_ms, main=m, follow_up=f,
                signals=signals, sounds_read=sounds_read, eligible=eligible,
                turn_ids=[x.turn_id for x in main + follow],
            ))
            reset()
    return out
