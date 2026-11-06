"""Dispatch metadata contract (BUILD_PLAN §4.7). The agent reads everything it needs from here."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


class Competency(BaseModel):
    id: str
    name: str
    anchors: dict[str, str] = Field(default_factory=dict)


class FaqEntry(BaseModel):
    q: str
    a: str


class RuntimeConfig(BaseModel):
    prompt_version: int
    prompt_text: str
    rubric_version: int
    question_bank_version: int
    competencies: list[Competency]
    faq: list[FaqEntry] = Field(default_factory=list)


class Knockout(BaseModel):
    id: str
    question: str
    pass_if: str
    fail_if: str = ""
    clarify: str = ""


class Question(BaseModel):
    id: str
    competency: str
    question: str
    probe: str = ""
    short: str = ""


class PlanStep(BaseModel):
    stage: Literal["open", "identity", "knockouts", "core", "candidate_q", "close"]
    id: str | None = None
    ids: list[str] = Field(default_factory=list)
    max_questions: int | None = None
    min_core_if_short_on_time: int | None = None


class Plan(BaseModel):
    plan_version: int
    plan_source: Literal["template", "model", "template_fallback"] = "template"
    steps: list[PlanStep]
    knockouts: list[Knockout]
    questions: list[Question]
    time_budget_s: dict[str, int] = Field(default_factory=lambda: {"soft_close": 420, "hard_close": 540})
    avoid: list[str] = Field(default_factory=list)

    def knockout_ids(self) -> list[str]:
        return [k.id for k in self.knockouts]

    def question_ids(self) -> list[str]:
        return [q.id for q in self.questions]


class DispatchMetadata(BaseModel):
    schema_version: int = 1
    application_id: str
    attempt_id: str
    call_id: str
    mode: Literal["inbound", "browser"]  # inbound = phone call-in to the LiveKit number (ADR-016)
    phone_e164: str | None = None
    candidate_identity: str
    candidate_first_name: str
    role_title: str
    org_name: str
    round_index: int = 0
    round_purpose: str = "Initial phone screen"
    plan: Plan
    keyterms: list[str] = Field(default_factory=list)
    runtime_config: RuntimeConfig
    ambience: bool = False
    record_egress: bool = True
    max_call_s: int = 900


def parse_dispatch_metadata(raw: str | None) -> DispatchMetadata | None:
    """Return None when there is no metadata (a phone call-in: claim it from the web, BUILD_PLAN §4.4).

    Raises ValueError on malformed metadata so the job fails loudly instead of improvising.
    """
    if not raw or not raw.strip():
        return None
    try:
        return DispatchMetadata.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"invalid dispatch metadata: {e}") from e


# ---------- role readiness gate (BUILD_PLAN §4.10) ----------
class KnockoutAnswers(BaseModel):
    id: str
    pass_: str = Field(alias="pass")
    fail: str
    unclear: str
    question_back: str


class QuestionAnswers(BaseModel):
    id: str
    strong: str
    thin: str


class AnswerBank(BaseModel):
    """Candidate lines for the role-agnostic personas, written by the web (Gemini) for this role's questions."""
    first_name: str = "Taylor"
    knockouts: list[KnockoutAnswers]
    questions: list[QuestionAnswers]


class ReadinessMetadata(BaseModel):
    schema_version: int = 1
    kind: Literal["readiness"]
    run_id: str
    role_template_id: str
    role: dict  # the compiled RoleConfig, same shape as packages/rubrics/<role>.json
    prompt_version: int
    prompt_text: str
    answer_bank: AnswerBank
    personas: list[str] = Field(default_factory=list)  # empty = all
    max_run_s: int = 600


def peek_kind(raw: str | None) -> str | None:
    """`readiness` for a gate job, None for an interview (or no metadata). Never raises."""
    if not raw or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data.get("kind") if isinstance(data, dict) else None


def parse_readiness_metadata(raw: str) -> ReadinessMetadata:
    try:
        return ReadinessMetadata.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"invalid readiness metadata: {e}") from e
