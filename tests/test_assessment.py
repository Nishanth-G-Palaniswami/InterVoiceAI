"""Parsing and quote verification of the post-call assessment (no LLM needed)."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from assessment import ASSESSMENT_NOTE, assess_call, parse_assessment

ANSWERS = [
    {
        "question_id": "q1",
        "question": "Tell me about the fall-prevention rollout.",
        "recorded": True,
        "main": {
            "text": "We started a huddle every shift and set bed alarms for high risk patients."
        },
        "follow_up": {"text": "I wrote the rounding script and trained twelve nurses."},
    },
    {
        "question_id": "q2",
        "question": "Why this role?",
        "recorded": False,
        "main": None,
        "follow_up": None,
    },
]


def _model_output(**overrides):
    data = {
        "questions": [
            {
                "id": "q1",
                "score": 4,
                "evidence": "set bed alarms for high risk patients",
                "note": "Specific and owned.",
            },
            {"id": "q2", "score": None, "evidence": "", "note": "Not answered."},
        ],
        "strengths": ["Concrete process detail", "Trained others"],
        "concerns": ["Fit question unanswered"],
        "recommendation": "follow_up",
        "rationale": "Strong on the first question. The role-fit question was not answered.",
    }
    data.update(overrides)
    return "```json\n" + json.dumps(data) + "\n```"


def test_parse_scores_and_verifies_quotes() -> None:
    result = parse_assessment(_model_output(), ANSWERS)
    assert result["status"] == "ok" and result["recommendation"] == "follow_up"
    assert result["overall"] == 4.0
    q1, q2 = result["questions"]
    assert q1["score"] == 4 and q1["evidence_verified"] is True
    assert q2["score"] is None and q2["evidence_verified"] is False
    assert result["note"] == ASSESSMENT_NOTE


def test_unverifiable_quote_is_marked_not_fabricated_away() -> None:
    out = _model_output(
        questions=[
            {"id": "q1", "score": 5, "evidence": "reduced falls by half", "note": ""}
        ]
    )
    result = parse_assessment(out, ANSWERS)
    assert result["questions"][0]["score"] == 5
    assert result["questions"][0]["evidence_verified"] is False


def test_bad_recommendation_or_unknown_question_raises() -> None:
    with pytest.raises(ValueError):
        parse_assessment(_model_output(recommendation="hire"), ANSWERS)
    with pytest.raises(ValueError):
        parse_assessment(_model_output(questions=[{"id": "q9", "score": 3}]), ANSWERS)
    with pytest.raises(ValueError):
        parse_assessment(_model_output(questions=[{"id": "q1", "score": 7}]), ANSWERS)


def test_no_recorded_answers_is_skipped_without_an_llm_call() -> None:
    result = asyncio.run(assess_call(None, [ANSWERS[1]]))
    assert result["status"] == "skipped"


def test_bare_numeric_ids_map_to_question_ids() -> None:
    out = _model_output(
        questions=[
            {"id": "1", "score": 3, "evidence": "", "note": ""},
            {"id": "Question 2", "score": None, "evidence": "", "note": ""},
        ]
    )
    result = parse_assessment(out, ANSWERS)
    assert [q["id"] for q in result["questions"]] == ["q1", "q2"]
