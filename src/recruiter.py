"""Recruiter view: joins applicants, call records and the analysis lane's call events.

Three sources, one page:
  C1  data/candidates/<id>.json      the application (src/candidates.py)
  C2  data/calls/<room>.json         Nia's per-answer summaries, written at hang-up
  EV  calls/<room>.events.jsonl      every turn and tool event, appended live by
                                     src/postcall.py; scored by the post-interview lane

Scores come from evals/ai_detect/dashboard.py (vendored, teammate-owned) so this page
and the standalone dashboard always agree. Nothing here edits those files.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, Literal, TypedDict

import candidates

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AI_DETECT = _REPO_ROOT / "evals" / "ai_detect"
if str(_AI_DETECT) not in sys.path:
    sys.path.insert(0, str(_AI_DETECT))

from dashboard import analyzed_calls, detector_table, tier_table  # noqa: E402
from run import evaluate, load_fixtures  # noqa: E402

# Same bands as the dashboard: score >= 0.6 likely, >= 0.35 possible.
THRESHOLD = 0.6
POSSIBLE = 0.35
# A log with no call_ended event counts as live while it was written this recently.
LIVE_WINDOW_S = 120.0
RESUME_EXCERPT_CHARS = 600
NOTE_MAX_CHARS = 500

DECISIONS: frozenset[str] = frozenset({"go", "hold", "nogo"})


class TranscriptLine(TypedDict):
    kind: Literal["turn", "answer", "restart", "identity"]
    role: str  # "agent" | "candidate" | "" for markers
    text: str
    at_s: float | None  # seconds from the first call_started
    interrupted: bool


class AnswerView(TypedDict):
    question_id: str
    question: str
    summary: str
    quality_hint: str
    recorded: bool
    analysis: dict[str, Any] | None  # the dashboard's per-answer observation


class CallView(TypedDict):
    id: str
    candidate_id: str | None
    matched_by: str | None
    caller_number: str | None
    started_at: str | None
    outcome: str  # completed | stopped | dropped | in_progress | incomplete
    in_progress: bool
    duration_s: int | None
    score: int  # 0..100
    band: str  # good | warn | crit, or none when no answer was long enough to judge
    analysis: dict[str, Any]  # the dashboard's call dict (confidence, problems, ...)
    answers: list[AnswerView]
    transcript: list[TranscriptLine]
    restarts: int
    assessment: dict[str, Any] | None  # Nia's post-call assessment, from the C2 record


class DecisionRecord(TypedDict):
    v: str | None  # one of DECISIONS, or None for a note on its own
    note: str
    at: str


# ---------- paths ----------


def events_dir() -> Path:
    """Where src/postcall.py writes: EVENTS_DUMP_DIR, else <repo>/calls."""
    raw = os.getenv("EVENTS_DUMP_DIR")
    return Path(raw).resolve() if raw else _REPO_ROOT / "calls"


def decisions_dir() -> Path:
    return candidates.data_dir() / "decisions"


def _safe_name(subject: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", subject)


# ---------- events ----------


def read_events(path: Path) -> list[dict[str, Any]]:
    """Parse one events log. The agent appends while a call runs, so a final line
    without its newline is still being written and is left for the next read."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if not text.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def _epoch(iso: str | None) -> float | None:
    return datetime.fromisoformat(iso).timestamp() if iso else None


def transcript(events: list[dict[str, Any]]) -> tuple[list[TranscriptLine], int]:
    """Turns in spoken order, with markers where answers were recorded and where the
    agent session restarted. Returns (lines, restart_count)."""
    stamped: list[tuple[float, int, TranscriptLine]] = []
    seen: set[str] = set()
    starts = sorted(
        t
        for t in (
            _epoch(e.get("occurred_at"))
            for e in events
            if e.get("type") == "call_started"
        )
        if t is not None
    )
    origin = starts[0] if starts else None
    for order, e in enumerate(events):
        kind = e.get("type")
        payload = e.get("payload") or {}
        if kind == "turn":
            turn_id = payload.get("turn_id") or f"seq-{order}"
            if turn_id in seen:
                continue
            seen.add(turn_id)
            at = payload.get("started_at") or _epoch(e.get("occurred_at"))
            if at is None:
                continue
            stamped.append(
                (
                    float(at),
                    order,
                    TranscriptLine(
                        kind="turn",
                        role=str(payload.get("role") or ""),
                        text=str(payload.get("text") or ""),
                        at_s=None,
                        interrupted=bool(payload.get("interrupted")),
                    ),
                )
            )
        elif kind == "tool_record_answer":
            at = _epoch(e.get("occurred_at"))
            if at is not None:
                stamped.append(
                    (
                        at,
                        order,
                        TranscriptLine(
                            kind="answer",
                            role="",
                            text=str(payload.get("question_id") or ""),
                            at_s=None,
                            interrupted=False,
                        ),
                    )
                )
        elif kind == "tool_mark_knockout":
            at = _epoch(e.get("occurred_at"))
            if at is not None:
                stamped.append(
                    (
                        at,
                        order,
                        TranscriptLine(
                            kind="identity",
                            role="",
                            text="Identity confirmed"
                            if str(payload.get("confirmed")).lower() == "true"
                            else "Identity not confirmed",
                            at_s=None,
                            interrupted=False,
                        ),
                    )
                )
    for at in starts[1:]:
        stamped.append(
            (
                at,
                -1,
                TranscriptLine(
                    kind="restart",
                    role="",
                    text="Agent session restarted",
                    at_s=None,
                    interrupted=False,
                ),
            )
        )
    stamped.sort(key=lambda s: (s[0], s[1]))
    lines: list[TranscriptLine] = []
    for at, _, line in stamped:
        line["at_s"] = round(at - origin, 1) if origin is not None else None
        lines.append(line)
    return lines, max(0, len(starts) - 1)


# ---------- per-call analysis, cached by file state ----------

_cache: dict[Path, tuple[tuple[float, int], dict[str, Any]]] = {}


def _band(score: float) -> str:
    return "crit" if score >= THRESHOLD else "warn" if score >= POSSIBLE else "good"


def _analyse(path: Path) -> dict[str, Any]:
    """Events plus the dashboard's analysis for one log; recomputed only when the
    file changes."""
    stat = path.stat()
    key = (stat.st_mtime, stat.st_size)
    hit = _cache.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]
    events = read_events(path)
    call_id = path.name.removesuffix(".events.jsonl")
    [analysis] = analyzed_calls([], [(call_id, events)])
    started = next((e for e in events if e.get("type") == "call_started"), None)
    ended = next((e for e in events if e.get("type") == "call_ended"), None)
    meta = (started or {}).get("payload") or {}
    entry = {
        "id": call_id,
        "events": events,
        "analysis": analysis,
        "candidate_id": meta.get("candidate_id"),
        "matched_by": meta.get("matched_by"),
        "started_at": (started or {}).get("occurred_at"),
        "ended": ended,
        "mtime": stat.st_mtime,
    }
    _cache[path] = (key, entry)
    return entry


def _event_logs() -> list[Path]:
    folder = events_dir()
    if not folder.is_dir():
        return []
    return list(folder.glob("*.events.jsonl"))


def _records_by_call() -> dict[str, dict[str, Any]]:
    return {r["call_id"]: r for r in candidates.load_calls() if r.get("call_id")}


def _plan_questions(candidate_id: str | None) -> dict[str, str]:
    if candidate_id:
        try:
            plan = candidates.load(candidate_id).get("plan") or []
        except FileNotFoundError:
            plan = []
    else:
        plan = []
    if not plan:
        plan = candidates.WALK_IN_PLAN
    return {q["id"]: q["text"] for q in plan}


def _call_view(entry: dict[str, Any], record: dict[str, Any] | None) -> CallView:
    events: list[dict[str, Any]] = entry["events"]
    analysis: dict[str, Any] = entry["analysis"]
    ended = entry["ended"]
    candidate_id = entry["candidate_id"] or (record or {}).get("candidate_id")

    # Live first: a restarted session can leave a "dropped" record mid-call. Then
    # Nia's record, which names how the interview went (completed / stopped /
    # dropped); call_ended only says the session closed, "completed" even for a
    # hang-up.
    if ended is None and time.time() - entry["mtime"] < LIVE_WINDOW_S:
        outcome = "in_progress"
    elif record is not None and record.get("outcome"):
        outcome = str(record["outcome"])
    elif ended is not None:
        outcome = str((ended.get("payload") or {}).get("outcome") or "completed")
    else:
        outcome = "incomplete"  # log stopped without call_ended or a record

    duration_s: int | None = None
    start, end = _epoch(entry["started_at"]), _epoch((ended or {}).get("occurred_at"))
    if start is not None and end is not None:
        duration_s = round(end - start)

    # Answers: questions in plan order, Nia's summary from the live event (or the
    # C2 record), the analysis lane's observation beside it.
    questions = _plan_questions(candidate_id)
    by_record = {a["question_id"]: a for a in (record or {}).get("answers", [])}
    by_event: dict[str, dict[str, Any]] = {}
    for e in events:
        if e.get("type") == "tool_record_answer":
            payload = e.get("payload") or {}
            if payload.get("question_id"):
                by_event[str(payload["question_id"])] = payload
    by_analysis = {a["question_id"]: a for a in analysis.get("answers", [])}
    ids = list(dict.fromkeys([*questions, *by_record, *by_event, *by_analysis]))
    answers: list[AnswerView] = []
    for qid in ids:
        rec = by_record.get(qid, {})
        ev = by_event.get(qid, {})
        summary = str(ev.get("summary") or rec.get("summary") or "")
        answers.append(
            AnswerView(
                question_id=qid,
                question=str(rec.get("question") or questions.get(qid, "")),
                summary=summary,
                quality_hint=str(
                    ev.get("quality_hint") or rec.get("quality_hint") or ""
                ),
                recorded=bool(ev) or bool(rec.get("recorded")),
                analysis=by_analysis.get(qid),
            )
        )

    lines, restarts = transcript(events)
    confidence = analysis.get("confidence") or {}
    score = float(confidence.get("score") or 0.0)
    judged = bool(confidence.get("eligible_answers"))
    return CallView(
        id=entry["id"],
        candidate_id=candidate_id,
        matched_by=entry["matched_by"] or (record or {}).get("matched_by"),
        caller_number=(record or {}).get("caller_number"),
        started_at=entry["started_at"],
        outcome=outcome,
        in_progress=outcome == "in_progress",
        duration_s=duration_s,
        score=round(score * 100),
        band=_band(score) if judged else "none",
        analysis=analysis,
        answers=answers,
        transcript=lines,
        restarts=restarts,
        assessment=(record or {}).get("assessment"),
    )


def all_calls() -> list[CallView]:
    """Every call with an events log, newest first."""
    records = _records_by_call()
    views = [
        _call_view(entry, records.get(entry["id"]))
        for entry in (_analyse(p) for p in _event_logs())
    ]
    views.sort(key=lambda v: v["started_at"] or "", reverse=True)
    return views


# ---------- decisions ----------


def load_decision(subject: str) -> DecisionRecord | None:
    path = decisions_dir() / f"{_safe_name(subject)}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_decision(subject: str, v: str | None, note: str) -> DecisionRecord:
    """Validate and store a recruiter decision; ValueError on a bad value."""
    if v is not None and v not in DECISIONS:
        raise ValueError(f"decision must be one of {sorted(DECISIONS)} or null")
    if len(note) > NOTE_MAX_CHARS:
        raise ValueError(f"note longer than {NOTE_MAX_CHARS} characters")
    record = DecisionRecord(
        v=v,
        note=note.strip(),
        at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    path = decisions_dir() / f"{_safe_name(subject)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


# ---------- page payloads ----------


def _call_summary(c: CallView) -> dict[str, Any]:
    return {
        "id": c["id"],
        "started_at": c["started_at"],
        "outcome": c["outcome"],
        "in_progress": c["in_progress"],
        "score": c["score"],
        "band": c["band"],
        "recorded": sum(a["recorded"] for a in c["answers"]),
        "questions": len(c["answers"]),
        "caller_number": c["caller_number"],
    }


def roster() -> dict[str, Any]:
    """Sidebar data: every applicant (newest first) and the calls no applicant owns."""
    calls = all_calls()
    by_candidate: dict[str, list[CallView]] = {}
    unmatched: list[CallView] = []
    # Two files can carry one id (data/candidates/candidate.example.json does);
    # load_all is newest first, so the first one wins.
    people: list[candidates.Candidate] = []
    known: set[str] = set()
    for cand in candidates.load_all():
        if cand["id"] not in known:
            known.add(cand["id"])
            people.append(cand)
    for c in calls:
        if c["candidate_id"] in known:
            by_candidate.setdefault(str(c["candidate_id"]), []).append(c)
        else:
            unmatched.append(c)
    applicants = []
    for cand in people:
        theirs = by_candidate.get(cand["id"], [])
        applicants.append(
            {
                "id": cand["id"],
                "name": cand["name"],
                "role": cand["role"],
                "company": cand["company"],
                "applied_at": cand["created_at"],
                "calls": len(theirs),
                "latest": _call_summary(theirs[0]) if theirs else None,
                "decision": load_decision(cand["id"]),
            }
        )
    return {
        "applicants": applicants,
        "unmatched": [
            {**_call_summary(c), "decision": load_decision(c["id"])} for c in unmatched
        ],
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def candidate_report(candidate_id: str) -> dict[str, Any]:
    """Everything about one applicant. FileNotFoundError if there is no such file."""
    cand = candidates.load(candidate_id)
    resume = cand["resume_text"]
    stored = candidates.resume_path(cand)
    return {
        "kind": "candidate",
        "candidate": {
            **{k: v for k, v in cand.items() if k != "resume_text"},
            "resume_excerpt": resume[:RESUME_EXCERPT_CHARS],
            "resume_chars": len(resume),
            # the original upload, when it was kept (applications before file
            # storage existed have only the extracted text)
            "resume_url": f"/resumes/{candidate_id}" if stored else None,
            "resume_kind": stored.suffix.lower().lstrip(".") if stored else None,
        },
        "decision": load_decision(candidate_id),
        "calls": [c for c in all_calls() if c["candidate_id"] == candidate_id],
    }


def call_report(call_id: str) -> dict[str, Any]:
    """One call nobody applied for (console run, walk-in). KeyError if unknown."""
    for c in all_calls():
        if c["id"] == call_id:
            return {
                "kind": "call",
                "candidate": None,
                "decision": load_decision(call_id),
                "calls": [c],
            }
    raise KeyError(call_id)


def known_subject(subject: str) -> bool:
    """A decision may only be filed against an applicant or a call that exists."""
    if any(c["id"] == subject for c in candidates.load_all()):
        return True
    return any(p.name.removesuffix(".events.jsonl") == subject for p in _event_logs())


@cache
def detector_performance() -> dict[str, Any]:
    """The labelled stress test behind the score; fixed data, computed once."""
    rows = evaluate(load_fixtures(), "synthetic")
    return {
        "n_fixtures": len(rows),
        "detectors": detector_table(rows),
        "tiers": tier_table(rows),
    }
