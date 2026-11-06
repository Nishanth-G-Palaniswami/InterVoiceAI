"""Post-call assessment: how well the candidate answered, for the recruiter.

One LLM call after the interview has ended (never during the call), over the
recorded answers plus the profile. The result is written into the call record
(contract C2) under "assessment". It judges job-relevant substance only; delivery
and the integrity signals are a separate, independent view. The recruiting team
decides.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from livekit.agents import inference, llm

from candidates import Candidate

logger = logging.getLogger("assessment")

ASSESS_MODEL = "google/gemma-4-31b-it"
RECOMMENDATIONS = ("advance", "follow_up", "do_not_advance")
ASSESSMENT_NOTE = (
    "Assessment by an AI from the transcript. The recruiting team decides."
)
RESUME_CHARS = 1500
MAX_ITEMS = 3

SYSTEM_PROMPT = (
    "You are a fair, careful recruiter writing a phone-screen assessment. "
    "You answer with JSON only, no prose and no code fences."
)


def _answers_block(answers: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for a in answers:
        main = a.get("main") or {}
        if not a.get("recorded") or not main.get("text"):
            parts.append(f"{a['question_id']}: {a['question']}\n  (not answered)")
            continue
        follow = (a.get("follow_up") or {}).get("text") or "(none)"
        parts.append(
            f"{a['question_id']}: {a['question']}\n"
            f"  answer: {main['text']}\n"
            f"  follow-up answer: {follow}"
        )
    return "\n".join(parts)


def _prompt(candidate: Candidate | None, answers: list[dict[str, Any]]) -> str:
    if candidate is None:
        profile = "Unknown caller, no application on file."
    else:
        years = candidate["years_experience"]
        facts = [
            f"Role applied for: {candidate['role']}",
            f"Company: {candidate['company'] or 'not given'}",
            f"Education: {candidate['education'] or 'not given'}",
            f"Years of experience: {years if years is not None else 'not given'}",
        ]
        facts += [
            f"{key.replace('_', ' ').capitalize()}: {value}"
            for key, value in candidate["answers"].items()
            if value
        ]
        resume = candidate["resume_text"][:RESUME_CHARS].strip() or "(none)"
        profile = "\n".join(facts) + f"\nResume excerpt:\n{resume}"
    return f"""Assess this phone screen for the recruiting team.

Candidate profile:
{profile}

Interview (the agent's question, then the candidate's transcribed answer and
their answer to one follow-up):
{_answers_block(answers)}

Rules:
- Judge job-relevant substance only: specifics, ownership, outcomes, fit for the
  role. Ignore grammar, accent, fluency, pauses and transcription errors.
- Never consider or mention age, race, religion, disability, pregnancy, family,
  nationality or immigration status.
- Score each answered question 1 to 5: 1 = nothing relevant, 3 = adequate but
  general, 5 = specific, owned, with a clear outcome. Unanswered questions get
  null.
- "evidence" is a verbatim quote of at most 15 words copied from that answer.

Return one JSON object with the keys:
"questions": a list with one object per question id in order, each with "id",
"score" (1-5 or null), "evidence" (quote or ""), "note" (one sentence);
"strengths": up to {MAX_ITEMS} short items; "concerns": up to {MAX_ITEMS} short items;
"recommendation": one of "advance", "follow_up", "do_not_advance";
"rationale": two sentences."""


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text.lower())).strip()


def _question_id(raw: Any) -> str:
    """Models write "q1", "Q1", "1" or "question 1"; the record uses "q1"."""
    digits = re.search(r"\d+", str(raw or ""))
    return f"q{int(digits.group())}" if digits else str(raw or "").strip()


def parse_assessment(text: str, answers: list[dict[str, Any]]) -> dict[str, Any]:
    """Strict parse of the model output, with every quote checked against the
    transcript. Raises ValueError on anything unexpected."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"assessment is not a JSON object: {text[:200]!r}")
    data = json.loads(body[start : end + 1])
    if data.get("recommendation") not in RECOMMENDATIONS:
        raise ValueError(f"bad recommendation: {data.get('recommendation')!r}")
    by_id = {a["question_id"]: a for a in answers}
    questions: list[dict[str, Any]] = []
    for item in data.get("questions") or []:
        qid = _question_id(item.get("id"))
        if qid not in by_id:
            raise ValueError(f"unknown question id in assessment: {item.get('id')!r}")
        score = item.get("score")
        if score is not None:
            score = int(score)
            if not 1 <= score <= 5:
                raise ValueError(f"score out of range for {qid}: {score}")
        quote = str(item.get("evidence") or "").strip()
        a = by_id[qid]
        spoken = " ".join(
            side.get("text", "")
            for side in (a.get("main") or {}, a.get("follow_up") or {})
        )
        questions.append(
            {
                "id": qid,
                "score": score,
                "evidence": quote,
                "evidence_verified": bool(quote) and _norm(quote) in _norm(spoken),
                "note": str(item.get("note") or "").strip(),
            }
        )
    if not questions:
        raise ValueError("assessment has no questions")
    scores = [q["score"] for q in questions if q["score"] is not None]
    return {
        "status": "ok",
        "model": ASSESS_MODEL,
        "questions": questions,
        "overall": round(sum(scores) / len(scores), 1) if scores else None,
        "strengths": [str(s) for s in data.get("strengths") or []][:MAX_ITEMS],
        "concerns": [str(s) for s in data.get("concerns") or []][:MAX_ITEMS],
        "recommendation": data["recommendation"],
        "rationale": str(data.get("rationale") or "").strip(),
        "note": ASSESSMENT_NOTE,
    }


async def assess_call(
    candidate: Candidate | None, answers: list[dict[str, Any]]
) -> dict[str, Any]:
    """One Inference LLM call over the recorded answers; LIVEKIT_* creds only."""
    if not any(
        a.get("recorded") and (a.get("main") or {}).get("text") for a in answers
    ):
        return {
            "status": "skipped",
            "reason": "no recorded answers",
            "note": ASSESSMENT_NOTE,
        }
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content=SYSTEM_PROMPT)
    chat_ctx.add_message(role="user", content=_prompt(candidate, answers))
    model = inference.LLM(model=ASSESS_MODEL)
    try:
        response = await model.chat(chat_ctx=chat_ctx).collect()
    finally:
        await model.aclose()
    result = parse_assessment(response.text, answers)
    logger.info(
        "assessment: %s (overall %s) %s",
        result["recommendation"],
        result["overall"],
        [(q["id"], q["score"]) for q in result["questions"]],
    )
    return result
