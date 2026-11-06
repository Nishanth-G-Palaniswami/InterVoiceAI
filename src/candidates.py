"""The JSON contract between the intake UI and the phone agent. Pure stdlib.

C1  data/candidates/<id>.json   written by the intake UI (src/intake_app.py), read by
                                the agent at call start. See docs/plans/02-intake-ui.md.
C2  data/calls/<room>.json      written by the agent when the interview finishes, read
                                by the scorecard page.

Both workstreams import this module instead of re-implementing the schema.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

SCHEMA_VERSION = 1
RESUME_MAX_CHARS = 6000
NAME_MATCH_RATIO = 0.8

_REPO_ROOT = Path(__file__).resolve().parents[1]


class Question(TypedDict):
    id: str  # "q1".."q4"
    text: str  # asked verbatim on the call
    topic: str  # short label, e.g. "recent role"
    grounded_in: str  # the resume/profile detail it builds on, or ""


class Candidate(TypedDict):
    schema_version: int
    id: str
    created_at: str  # ISO 8601 with offset
    name: str
    phone: str  # digits only; matched on the last 10
    email: str
    role: str  # role applied for
    company: str  # optional, "" if unknown
    education: str
    years_experience: int | None
    answers: dict[str, str]  # current_job, biggest_project, why_this_role
    resume_text: str  # plain text, at most RESUME_MAX_CHARS
    resume_filename: str
    plan: list[Question] | None  # None from the UI; the agent fills it


def data_dir() -> Path:
    """Root for runtime data. Override with DATA_DIR; default <repo>/data."""
    return Path(os.getenv("DATA_DIR") or _REPO_ROOT / "data")


def candidates_dir() -> Path:
    return data_dir() / "candidates"


def calls_dir() -> Path:
    return data_dir() / "calls"


# ---------- normalisation ----------


def normalize_phone(raw: str) -> str:
    """Digits only. '+1 (212) 555-0142' -> '12125550142'."""
    return re.sub(r"\D", "", raw or "")


def _phone_key(digits: str) -> str:
    return digits[-10:]


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "candidate"


def new_id(name: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return f"c_{stamp}_{_slug(name)}"


# ---------- C1: candidates ----------


def new_candidate(
    *,
    name: str,
    phone: str,
    role: str,
    email: str = "",
    company: str = "",
    education: str = "",
    years_experience: int | None = None,
    answers: dict[str, str] | None = None,
    resume_text: str = "",
    resume_filename: str = "",
) -> Candidate:
    """Build a validated C1 record. Raises ValueError listing every missing field."""
    digits = normalize_phone(phone)
    missing = [
        label
        for label, ok in (
            ("name", bool(name.strip())),
            ("phone (10 or 11 digits)", 10 <= len(digits) <= 11),
            ("role", bool(role.strip())),
        )
        if not ok
    ]
    if missing:
        raise ValueError(f"Missing or invalid: {', '.join(missing)}")
    now = datetime.now().astimezone()
    return Candidate(
        schema_version=SCHEMA_VERSION,
        id=new_id(name, now),
        created_at=now.isoformat(timespec="seconds"),
        name=name.strip(),
        phone=digits,
        email=email.strip(),
        role=role.strip(),
        company=company.strip(),
        education=education.strip(),
        years_experience=years_experience,
        answers=dict(answers or {}),
        resume_text=resume_text[:RESUME_MAX_CHARS],
        resume_filename=resume_filename,
        plan=None,
    )


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # encoding="utf-8" is BOM-less; PowerShell redirects are not.
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def save(candidate: Candidate) -> Path:
    return _write_json(candidates_dir() / f"{candidate['id']}.json", candidate)


def load(candidate_id: str) -> Candidate:
    path = candidates_dir() / f"{candidate_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_all() -> list[Candidate]:
    """Every profile, newest first. Missing folder means no candidates yet."""
    folder = candidates_dir()
    if not folder.is_dir():
        return []
    records = [json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")]
    records.sort(key=lambda c: c["created_at"], reverse=True)
    return records


def find_by_phone(number: str | None) -> Candidate | None:
    """Caller-ID lookup on the last ten digits; newest profile wins."""
    if not number:
        return None
    key = _phone_key(normalize_phone(number))
    if len(key) < 10:
        return None
    for candidate in load_all():
        if _phone_key(candidate["phone"]) == key:
            return candidate
    return None


def find_by_name(spoken: str) -> Candidate | None:
    """Fuzzy match on the full name (ratio >= 0.8), else a unique first-name match."""
    key = _name_key(spoken)
    if not key:
        return None
    candidates = load_all()
    best: tuple[float, Candidate | None] = (0.0, None)
    for candidate in candidates:
        ratio = difflib.SequenceMatcher(None, key, _name_key(candidate["name"])).ratio()
        if ratio > best[0]:
            best = (ratio, candidate)
    if best[0] >= NAME_MATCH_RATIO:
        return best[1]
    first = key.split()[0]
    by_first = [c for c in candidates if _name_key(c["name"]).split()[:1] == [first]]
    return by_first[0] if len(by_first) == 1 else None


WALK_IN_PLAN: list[Question] = [
    Question(
        id="q1",
        text="Tell me about your most recent job and what a typical day looked like.",
        topic="recent role",
        grounded_in="",
    ),
    Question(
        id="q2",
        text="What is the project or piece of work you are proudest of, and what was your part in it?",
        topic="proudest work",
        grounded_in="",
    ),
    Question(
        id="q3",
        text="Why are you interested in this role, and what makes you a fit for it?",
        topic="role fit",
        grounded_in="",
    ),
    Question(
        id="q4",
        text="Describe a time something went wrong at work. What did you do, and what happened next?",
        topic="behavioural",
        grounded_in="",
    ),
]


def walk_in_candidate(name: str) -> Candidate:
    """A profile for a caller we could not match. Not saved; uses the generic plan."""
    now = datetime.now().astimezone()
    return Candidate(
        schema_version=SCHEMA_VERSION,
        id="walk-in",
        created_at=now.isoformat(timespec="seconds"),
        name=name.strip() or "the candidate",
        phone="",
        email="",
        role="the role you applied for",
        company="",
        education="",
        years_experience=None,
        answers={},
        resume_text="",
        resume_filename="",
        plan=list(WALK_IN_PLAN),
    )


# ---------- C2: calls ----------


def _safe_room(room: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", room) or "call"


def save_call(room: str, data: dict[str, Any]) -> Path:
    return _write_json(calls_dir() / f"{_safe_room(room)}.json", data)


def load_call(call_id: str) -> dict[str, Any]:
    path = calls_dir() / f"{_safe_room(call_id)}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_calls() -> list[dict[str, Any]]:
    """Every call record, newest first (by started_at, then file name)."""
    folder = calls_dir()
    if not folder.is_dir():
        return []
    records = [json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")]
    records.sort(
        key=lambda c: (c.get("started_at") or "", c.get("call_id") or ""), reverse=True
    )
    return records
