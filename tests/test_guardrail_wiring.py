"""The pre-TTS guardrail on the live InterviewAgent: model text is checked per
sentence in llm_node, scripted speech in tts_node, tool calls pass through, and an
interrupted reply never releases its buffered text."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from livekit.agents import Agent, llm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import candidates
from interviewer.guardrail import SAFE_LINE
from my_agent import InterviewAgent

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "candidate.example.json"


@pytest.fixture
def agent() -> InterviewAgent:
    profile = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    profile["plan"] = candidates.WALK_IN_PLAN
    return InterviewAgent(
        candidate=profile, caller_number="+12125550142", call_id="t", matched_by="phone"
    )


def _chunk(content=None, tool=None, extra=None):
    calls = (
        [llm.FunctionToolCall(name=tool[0], arguments=tool[1], call_id="c1")]
        if tool
        else []
    )
    return llm.ChatChunk(
        id="x",
        delta=llm.ChoiceDelta(
            role="assistant", content=content, tool_calls=calls, extra=extra
        ),
    )


def _fake_llm(monkeypatch, chunks):
    async def fake(agent, chat_ctx, tools, model_settings):
        for c in chunks:
            await asyncio.sleep(0)
            yield c

    monkeypatch.setattr(Agent.default, "llm_node", staticmethod(fake))


async def _texts(gen) -> str:
    return "".join([c async for c in gen if isinstance(c, str)])


async def test_pay_figure_is_replaced_and_the_tool_call_passes(agent, monkeypatch):
    _fake_llm(
        monkeypatch,
        [
            _chunk("Thanks. "),
            _chunk("It pays about fif"),
            _chunk("ty dollars an hour. "),
            _chunk(tool=("record_answer", '{"question_id":"q1"}'), extra={"sig": "a"}),
            _chunk("Next question"),
        ],
    )
    out = [c async for c in agent.llm_node(llm.ChatContext(), [], None)]
    spoken = "".join(c for c in out if isinstance(c, str))
    tools = [c for c in out if isinstance(c, llm.ChatChunk) and c.delta.tool_calls]
    assert "fifty" not in spoken and SAFE_LINE in spoken
    assert spoken.startswith("Thanks. ") and spoken.endswith("Next question")
    assert tools[0].delta.tool_calls[0].name == "record_answer"
    assert tools[0].delta.extra == {"sig": "a"} and tools[0].delta.content is None
    assert agent.guardrail_events == [{"rule": "pay_figure", "source": "llm"}]


@pytest.mark.parametrize(
    "line",
    [
        "Great news, you got the job. ",
        "You can start next Monday. ",
        "How old are you? ",
    ],
)
async def test_offers_start_dates_and_protected_questions_are_replaced(
    agent, monkeypatch, line
):
    _fake_llm(monkeypatch, [_chunk(line)])
    spoken = await _texts(agent.llm_node(llm.ChatContext(), [], None))
    assert spoken.strip() == SAFE_LINE
    assert len(agent.guardrail_events) == 1


async def test_own_tool_written_as_text_is_not_spoken(agent, monkeypatch):
    _fake_llm(
        monkeypatch,
        [_chunk("Thanks. confirm_identity(confirmed=true) Let's begin. ")],
    )
    spoken = await _texts(agent.llm_node(llm.ChatContext(), [], None))
    assert "confirm_identity" not in spoken and "Let's begin." in spoken


async def test_interrupted_reply_releases_nothing(agent, monkeypatch):
    stall = asyncio.Event()

    async def fake(agent_, chat_ctx, tools, model_settings):
        yield _chunk("The pay is fifty dollars")  # no sentence end yet: held back
        await stall.wait()  # the candidate interrupts here
        yield _chunk(" an hour. ")

    monkeypatch.setattr(Agent.default, "llm_node", staticmethod(fake))
    released = []

    async def consume():
        async for c in agent.llm_node(llm.ChatContext(), [], None):
            released.append(c)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released == []


async def test_scripted_speech_is_checked_in_tts_node(agent, monkeypatch):
    received: list[str] = []

    async def fake_tts(agent_, text, model_settings):
        async for t in text:
            received.append(t)
        return
        yield  # an async generator, like the real node

    monkeypatch.setattr(Agent.default, "tts_node", staticmethod(fake_tts))

    async def say():
        yield "Hi Sam, thanks for calling. "
        yield "The salary is 90k a year. "

    async for _ in agent.tts_node(say(), None):
        pass
    joined = "".join(received)
    assert (
        "90k" not in joined and SAFE_LINE in joined and "thanks for calling" in joined
    )
    assert agent.guardrail_events == [{"rule": "pay_figure", "source": "tts"}]


def test_call_record_lists_guardrail_events(agent, tmp_path, monkeypatch):
    monkeypatch.setattr(
        candidates, "save_call", lambda cid, data: tmp_path / f"{cid}.json"
    )
    captured = {}
    orig = candidates.save_call
    monkeypatch.setattr(
        candidates,
        "save_call",
        lambda cid, data: captured.update(data) or orig(cid, data),
    )
    agent.guardrail_events.append({"rule": "pay_figure", "source": "llm"})
    agent.write_call("completed")
    assert captured["guardrail_events"] == [{"rule": "pay_figure", "source": "llm"}]
