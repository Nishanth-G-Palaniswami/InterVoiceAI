"""Turns a candidate profile into the four questions the interviewer will ask.

One LLM call through LiveKit Inference (no extra keys), made at call start when the
profile has no plan yet. The result is written back into the profile (contract C1)
so the intake UI can show it and a second call reuses it.
"""

from __future__ import annotations

import json
import logging
import re

from livekit.agents import inference, llm

from candidates import Candidate, Question

logger = logging.getLogger("question-plan")

PLAN_MODEL = "google/gemma-4-31b-it"
QUESTION_COUNT = 4
RESUME_CHARS_FOR_PLAN = 4000

SYSTEM_PROMPT = (
    "You write phone-screen questions for a recruiter. "
    "You answer with JSON only, no prose and no code fences."
)


def _prompt(c: Candidate) -> str:
    answers = "\n".join(
        f"- {key.replace('_', ' ')}: {value}"
        for key, value in c["answers"].items()
        if value
    )
    years = c["years_experience"]
    return f"""Write exactly {QUESTION_COUNT} interview questions for a phone screen.

Candidate: {c["name"]}
Role applied for: {c["role"]} at {c["company"] or "the company"}
Education: {c["education"] or "not given"}
Years of experience: {years if years is not None else "not given"}
Profile answers:
{answers or "- none"}

Resume:
{c["resume_text"][:RESUME_CHARS_FOR_PLAN] or "(no resume text)"}

Rules:
- q1 and q2 build on specific, named details from the resume or the profile answers
  (an employer, a project, a number, a responsibility). Name that detail in the question.
- q3 asks about fit for the role applied for.
- q4 is a behavioural question about a time something went wrong at work.
- Each question is one spoken sentence a person can answer in under a minute.
  No lists, no compound questions, no jargon the candidate did not use.
- Never ask about age, race, religion, disability, pregnancy, family plans,
  nationality or immigration status.

Return a JSON array of {QUESTION_COUNT} objects with the keys "id" ("q1" to
"q{QUESTION_COUNT}"), "text", "topic" (two or three words) and "grounded_in" (the
resume or profile detail the question builds on, or an empty string)."""


def parse_plan(text: str) -> list[Question]:
    """Strict parse of the model output. Raises ValueError on anything unexpected."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = body.find("["), body.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"plan is not a JSON array: {text[:200]!r}")
    data = json.loads(body[start : end + 1])
    if not isinstance(data, list) or len(data) != QUESTION_COUNT:
        raise ValueError(f"expected {QUESTION_COUNT} questions, got: {text[:200]!r}")
    plan: list[Question] = []
    for number, item in enumerate(data, start=1):
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            raise ValueError(f"question {number} has no text: {item!r}")
        plan.append(
            Question(
                id=f"q{number}",
                text=str(item["text"]).strip(),
                topic=str(item.get("topic", "")).strip(),
                grounded_in=str(item.get("grounded_in", "")).strip(),
            )
        )
    return plan


async def generate_plan(candidate: Candidate) -> list[Question]:
    """One Inference LLM call; only LIVEKIT_* credentials are needed."""
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content=SYSTEM_PROMPT)
    chat_ctx.add_message(role="user", content=_prompt(candidate))
    model = inference.LLM(model=PLAN_MODEL)
    try:
        response = await model.chat(chat_ctx=chat_ctx).collect()
    finally:
        await model.aclose()
    plan = parse_plan(response.text)
    logger.info("plan for %s: %s", candidate["id"], [q["text"] for q in plan])
    return plan
