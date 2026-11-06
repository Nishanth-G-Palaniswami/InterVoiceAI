"""Turn split and C2 scoring of the InterviewAgent, without a live session."""

import json
import sys
from pathlib import Path

from livekit.agents import llm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import candidates
from my_agent import InterviewAgent, Turn, call_integrity, score_answer

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "candidate.example.json"


def _agent() -> InterviewAgent:
    profile = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    profile["plan"] = candidates.WALK_IN_PLAN
    return InterviewAgent(
        candidate=profile, caller_number="+12125550142", call_id="t", matched_by="phone"
    )


def _timed(
    text: str,
    *,
    start: float = 0.0,
    step: float = 0.35,
    gaps: dict[int, float] | None = None,
):
    """Even-paced words; `gaps` adds a pause before the given word index."""
    words, t = [], start
    for i, w in enumerate(text.split()):
        t += (gaps or {}).get(i, 0.0)
        words.append((w, round(t, 2), round(t + step * 0.8, 2)))
        t += step
    return words


def test_take_turns_splits_on_agent_messages() -> None:
    agent = _agent()
    history = llm.ChatContext()
    history.add_message(role="assistant", content="Question one?")
    first = history.add_message(role="user", content="Part one of my answer.")
    history.add_message(role="user", content="And part two.")
    agent._timings[first.id] = (_timed("Part one of my answer."), 1800)

    turns = agent.take_turns(history)
    assert [t.text for t in turns] == ["Part one of my answer.", "And part two."]
    assert turns[0].onset_ms == 1800 and len(turns[0].words) == 5
    assert turns[1].words == [] and turns[1].onset_ms is None

    # handed out once: nothing new since, so the next take is empty
    assert agent.take_turns(history) == []
    history.add_message(role="assistant", content="Follow-up?")
    history.add_message(role="user", content="Follow-up answer.")
    assert [t.text for t in agent.take_turns(history)] == ["Follow-up answer."]


def test_fluent_read_answer_is_flagged_and_hesitant_is_not() -> None:
    agent = _agent()
    # 60 distinct alphabetic tokens: delivery counts an immediate repeat as a restart
    # (no h/m/r/u: "ah", "mm", "er", "uh" would count as fillers)
    read_text = " ".join(a + b for a in "abcdefgi" for b in "abcdefgi")[: 60 * 3 - 1]
    hesitant_text = "um so I think uh we did the um rollout and I I mean we tried"
    a_read = agent._answers["q1"]
    a_read.main = [Turn("m", read_text, _timed(read_text, step=0.3), 5200)]
    a_read.follow_up = [
        Turn("f", hesitant_text, _timed(hesitant_text, gaps={3: 0.9, 8: 1.2}), 400)
    ]
    a_read.recorded = True
    entry, obs = score_answer(a_read)
    assert entry["main"]["words"] == 60 and entry["main"]["wpm"] is not None
    assert entry["sounds_read"] is True
    assert "voice: almost no fillers" in entry["signals"]
    assert any("follow-up much less fluent" in s for s in entry["signals"])

    a_nat = agent._answers["q2"]
    a_nat.main = [Turn("n", hesitant_text, _timed(hesitant_text, gaps={3: 0.9}), 600)]
    a_nat.recorded = True
    entry_nat, obs_nat = score_answer(a_nat)
    assert entry_nat["sounds_read"] is False
    assert entry_nat["follow_up"] is None

    integrity = call_integrity([obs, obs_nat])
    assert integrity["verdict"] in {"natural", "inconclusive", "likely_assisted"}
    assert integrity["flagged_answers"] >= 1
    assert integrity["note"]


def test_finish_refuses_while_an_answer_is_pending(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    agent = _agent()
    q1 = agent._answers["q1"]
    q1.main = [Turn("m", "An answer with some detail in it.", [], None)]
    q1.follow_up_requested = True  # gate asked for the follow-up, not recorded yet

    refusal = agent.finish(candidate_wants_to_stop=False)
    assert refusal.startswith("Not finished: q1")
    assert "follow-up" in refusal and not (tmp_path / "calls").exists()

    saved = agent.finish(candidate_wants_to_stop=True)
    assert saved.startswith("Saved.")
    record = json.loads((tmp_path / "calls" / "t.json").read_text(encoding="utf-8"))
    assert record["outcome"] == "stopped"
    assert [a["recorded"] for a in record["answers"]] == [False, False, False, False]


def test_text_only_turns_score_without_timings() -> None:
    agent = _agent()
    a = agent._answers["q3"]
    a.main = [
        Turn("x", "We moved the clinic to a new scheduler last spring.", [], None)
    ]
    a.recorded = True
    entry, obs = score_answer(a)
    assert entry["main"]["words"] == 0 and entry["main"]["word_count"] == 10
    assert entry["sounds_read"] is False
    assert obs.verdict in {"ineligible", "none", "reads_generated"}
