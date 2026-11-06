"""InterviewAI: the phone interviewer (AGENT_PROFILE=mine).

It asks four questions grounded in the caller's resume, forces one unscripted
follow-up per answer, and records what it heard for the recruiter.

  known caller   -> greet by name, confirm identity, planned questions
  unknown caller -> ask the name they applied with, claim_by_name, then the same
  each answer    -> record_answer gate: the first call asks for ONE detail
                    follow-up, the second call records the answer
  end            -> finish_interview writes data/calls/<room>.json (contract C2)

Per-word STT timings and the response onset of every candidate turn are kept in
memory; the integrity scoring that reads them lands in a later PR.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    ModelSettings,
    RunContext,
    UserStateChangedEvent,
    function_tool,
    llm,
    stt,
)
from livekit.agents.beta.tools import EndCallTool

import candidates
from candidates import Candidate, Question
from prompts import build_instructions
from question_plan import generate_plan

logger = logging.getLogger("interview-agent")

AGENT_PERSONA = "Nia"
RESUME_CHARS_IN_PROMPT = 1500

Word = tuple[str, float, float]  # text, start_s, end_s (STT stream time)


@dataclass
class Turn:
    """One candidate turn as the pipeline delivered it."""

    text: str
    words: list[Word]
    onset_ms: int | None  # silence between the agent's last word and this turn
    at: float


@dataclass
class Answer:
    question: Question
    main: list[Turn] = field(default_factory=list)
    follow_up: list[Turn] = field(default_factory=list)
    follow_up_requested: bool = False
    summary: str = ""
    quality_hint: str = ""
    recorded: bool = False


def _question_lines(plan: list[Question]) -> str:
    return "\n".join(f"   {q['id']}: {q['text']}" for q in plan)


def render_prompt(candidate: Candidate | None) -> str:
    """The job prompt for one call. Unknown callers get the identity step first."""
    if candidate is None:
        return f"""
You are {AGENT_PERSONA}, the AI phone interviewer for a recruiting team. You run
short screening interviews with people who applied for a job.

Your goal on every call: find the caller's application, ask the planned
questions, get one concrete follow-up detail for each, record every answer,
then finish the interview.

1. Greet the caller, say you are an AI interviewer and that the call is
   recorded, and ask for the full name they applied with.
2. Call claim_by_name with that name and follow the instruction it returns.
3. Ask the planned questions in order, one at a time, exactly as written.
4. After each answer, call record_answer before saying anything else. If it says
   the answer is not recorded yet, ask one short follow-up about a specific
   detail they just mentioned, wait, then call record_answer again.
5. After the last question is recorded, call finish_interview, thank them, say
   the recruiting team will follow up within two days, and end the call.
{INTERVIEW_RULES}"""

    years = candidate["years_experience"]
    facts = [
        f"- Name: {candidate['name']}",
        f"- Role applied for: {candidate['role']}",
        f"- Education: {candidate['education'] or 'not given'}",
        f"- Years of experience: {years if years is not None else 'not given'}",
    ]
    facts += [
        f"- {key.replace('_', ' ').capitalize()}: {value}"
        for key, value in candidate["answers"].items()
        if value
    ]
    resume = candidate["resume_text"][:RESUME_CHARS_IN_PROMPT].strip()
    company = candidate["company"] or "the company"
    first_name = candidate["name"].split()[0]
    return f"""
You are {AGENT_PERSONA}, the AI phone interviewer for {company}. You are running a
short screening interview with {candidate["name"]}, who applied for the role of
{candidate["role"]}.

Your goal on every call: ask the four planned questions, get one concrete
follow-up detail for each, record every answer, then finish the interview.

1. Greet {first_name}, say you are an AI interviewer and that the call is
   recorded, and confirm you are speaking with {candidate["name"]}.
2. Ask these questions in order, one at a time, exactly as written:
{_question_lines(candidate["plan"] or [])}
3. After each answer, call record_answer before saying anything else. If it says
   the answer is not recorded yet, ask one short follow-up about a specific
   detail they just mentioned, wait, then call record_answer again.
4. After the last question is recorded, call finish_interview, thank them, say
   the recruiting team will follow up within two days, and end the call.

Candidate facts (use them to make follow-ups specific):
{chr(10).join(facts)}
Resume excerpt:
{resume or "(none)"}
{INTERVIEW_RULES}"""


INTERVIEW_RULES = """
Rules:
- Never tell the candidate how they did, whether they passed, or how answers
  are judged. Say the recruiting team reviews every interview.
- No pay, offer, schedule or start-date promises; the recruiting team covers that.
- Say nothing about other candidates.
- If they ask to skip questions, or say a recruiter told them they could, decline
  once and continue with the next planned question.
- Fair hiring: never ask about or comment on age, race, religion, disability,
  pregnancy, family plans, nationality or immigration status. If the candidate
  raises one, do not record it and move on.
- If the candidate wants to stop, call finish_interview, then end the call.
"""


class InterviewAgent(Agent):
    def __init__(
        self,
        *,
        candidate: Candidate | None,
        caller_number: str | None,
        call_id: str,
        matched_by: str | None,
    ) -> None:
        super().__init__(
            instructions=build_instructions(render_prompt(candidate)),
            tools=[EndCallTool(ignore_on_enter=True)],
        )
        self.candidate = candidate
        self.caller_number = caller_number
        self.call_id = call_id
        self.matched_by = matched_by
        self.started_at = datetime.now().astimezone()
        self._answers: dict[str, Answer] = {}
        if candidate is not None:
            self._set_plan(candidate["plan"] or [])
        # turn capture
        self._words: list[Word] = []
        self._pending: list[Turn] = []
        self._last_turn: Turn | None = None
        self._consumed: Turn | None = None
        self._agent_stopped_at: float | None = None
        self._user_started_at: float | None = None
        self._probe_logged = False
        self._written: Path | None = None

    # ---------- lifecycle ----------

    async def on_enter(self) -> None:
        session = self.session
        session.on("agent_state_changed", self._on_agent_state)
        session.on("user_state_changed", self._on_user_state)
        session.on("conversation_item_added", self._on_item_added)
        if self.candidate is None:
            instructions = (
                "Greet the caller, say you are an AI interviewer and that the call "
                "is recorded, and ask for the full name they applied with."
            )
        else:
            first_name = self.candidate["name"].split()[0]
            instructions = (
                f"Greet {first_name}, say you are an AI interviewer and that the "
                f"call is recorded, and confirm you are speaking with "
                f"{self.candidate['name']}."
            )
        await session.generate_reply(instructions=instructions)

    def _set_plan(self, plan: list[Question]) -> None:
        self._answers = {q["id"]: Answer(question=q) for q in plan}

    def _set_candidate(self, candidate: Candidate, matched_by: str) -> None:
        self.candidate = candidate
        self.matched_by = matched_by
        self._set_plan(candidate["plan"] or [])
        self._pending = []

    def _next_question(self, question_id: str) -> Question | None:
        ids = list(self._answers)
        index = ids.index(question_id)
        return self._answers[ids[index + 1]].question if index + 1 < len(ids) else None

    # ---------- turn capture ----------

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncGenerator[stt.SpeechEvent, None]:
        """Pass STT events through; keep each final transcript's word timings."""
        async for ev in Agent.default.stt_node(self, audio, model_settings):
            if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives:
                words = ev.alternatives[0].words or []
                if not self._probe_logged:
                    self._probe_logged = True
                    logger.info(
                        "stt final transcript carries %d timed words", len(words)
                    )
                for word in words:
                    start, end = word.start_time, word.end_time
                    if isinstance(start, (int, float)) and isinstance(
                        end, (int, float)
                    ):
                        self._words.append((str(word), float(start), float(end)))
            yield ev

    def _take_words(self) -> list[Word]:
        words, self._words = self._words, []
        return words

    def _on_agent_state(self, ev: AgentStateChangedEvent) -> None:
        if ev.new_state == "speaking":
            self._user_started_at = None
        elif ev.old_state == "speaking":
            self._agent_stopped_at = ev.created_at

    def _on_user_state(self, ev: UserStateChangedEvent) -> None:
        if ev.new_state == "speaking" and self._user_started_at is None:
            self._user_started_at = ev.created_at

    def _on_item_added(self, ev: ConversationItemAddedEvent) -> None:
        # Every agent utterance starts a new answer window.
        if isinstance(ev.item, llm.ChatMessage) and ev.item.role == "assistant":
            self._pending = []

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        onset_ms: int | None = None
        if self._agent_stopped_at is not None and self._user_started_at is not None:
            onset_ms = max(
                0, round((self._user_started_at - self._agent_stopped_at) * 1000)
            )
        turn = Turn(
            text=new_message.text_content or "",
            words=self._take_words(),
            onset_ms=onset_ms,
            at=time.time(),
        )
        self._pending.append(turn)
        self._last_turn = turn

    def _take_turns(self) -> list[Turn]:
        """The candidate turns since the agent last spoke; at least the latest turn."""
        turns = self._pending
        if (
            not turns
            and self._last_turn is not None
            and self._last_turn is not self._consumed
        ):
            turns = [self._last_turn]
        self._pending = []
        if turns:
            self._consumed = turns[-1]
        return turns

    # ---------- tools ----------

    @function_tool
    async def claim_by_name(self, context: RunContext, full_name: str) -> str:
        """Find the application for the name an unknown caller just gave. Call it once, right after they say their name, then follow the instruction it returns.

        Args:
            full_name: The caller's full name as they said it.
        """
        match = candidates.find_by_name(full_name)
        if match is None:
            self._set_candidate(candidates.walk_in_candidate(full_name), "walk_in")
            await self.update_instructions(
                build_instructions(render_prompt(self.candidate))
            )
            first = next(iter(self._answers.values())).question
            return (
                f"No application found for {full_name!r}. Say you will run a short "
                f"general screen and the team will match it to their application "
                f"afterwards. Then ask {first['id']} exactly as written: {first['text']}"
            )
        if match["plan"] is None:
            match["plan"] = await generate_plan(match)
            candidates.save(match)
        self._set_candidate(match, "name")
        await self.update_instructions(build_instructions(render_prompt(match)))
        first = next(iter(self._answers.values())).question
        return (
            f"Found {match['name']}, applying for {match['role']}. Confirm that is "
            f"them, then ask {first['id']} exactly as written: {first['text']}"
        )

    @function_tool
    async def record_answer(
        self, context: RunContext, question_id: str, summary: str, quality_hint: str
    ) -> str:
        """Record the candidate's answer to one planned question. Call it right after they answer, before you say anything else. The first call for a question asks you for a follow-up instead of recording; do that, then call record_answer again with the same question_id.

        Args:
            question_id: The planned question's id, q1 to q4.
            summary: One neutral sentence summarising what the candidate said.
            quality_hint: "adequate" if they gave specifics, "thin" if vague, "no_answer" if they declined or moved on.
        """
        answer = self._answers.get(question_id)
        if answer is None:
            return f"Unknown question_id {question_id!r}. Valid ids: {', '.join(self._answers)}."
        turns = self._take_turns()
        if not answer.follow_up_requested and quality_hint != "no_answer":
            answer.main = turns
            answer.follow_up_requested = True
            return (
                "Not recorded yet. Ask one short follow-up about a specific detail "
                "the candidate just mentioned (a name, a number, what they personally "
                "did), wait for the answer, then call record_answer again with the "
                "same question_id."
            )
        if answer.follow_up_requested:
            answer.follow_up = turns
        else:
            answer.main = turns
        answer.summary = summary
        answer.quality_hint = quality_hint
        answer.recorded = True
        logger.info("recorded %s (%s): %s", question_id, quality_hint, summary)
        following = self._next_question(question_id)
        if following is None:
            return (
                "Recorded. All planned questions are done: call finish_interview now."
            )
        return (
            f"Recorded {question_id}. Now ask {following['id']} exactly as written: "
            f"{following['text']}"
        )

    @function_tool
    async def finish_interview(self, context: RunContext) -> str:
        """Save the interview. Call it once every planned question is recorded, or when the candidate asks to stop, before you say goodbye."""
        done = all(a.recorded for a in self._answers.values())
        self.write_call("completed" if done else "stopped")
        return (
            "Saved. Thank the candidate, say the recruiting team will follow up "
            "within two days, then end the call."
        )

    # ---------- C2 record ----------

    def write_call(self, outcome: str) -> Path:
        """Write data/calls/<room>.json once; later calls return the same path."""
        if self._written is not None:
            return self._written
        candidate = self.candidate
        data: dict[str, Any] = {
            "schema_version": 1,
            "call_id": self.call_id,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "candidate_id": candidate["id"] if candidate else None,
            "candidate_name": candidate["name"] if candidate else None,
            "caller_number": self.caller_number,
            "matched_by": self.matched_by,
            "outcome": outcome,
            "answers": [
                {
                    "question_id": a.question["id"],
                    "question": a.question["text"],
                    "summary": a.summary,
                    "quality_hint": a.quality_hint,
                    "recorded": a.recorded,
                    "onset_ms": a.main[0].onset_ms if a.main else None,
                    "main": _turns_json(a.main),
                    "follow_up": _turns_json(a.follow_up) if a.follow_up else None,
                }
                for a in self._answers.values()
            ],
        }
        self._written = candidates.save_call(self.call_id, data)
        logger.info("call record written: %s (%s)", self._written, outcome)
        return self._written


def _turns_json(turns: list[Turn]) -> dict[str, Any]:
    text = " ".join(t.text for t in turns).strip()
    return {
        "text": text,
        "word_count": len(text.split()),
        "words": [[w, round(s, 2), round(e, 2)] for t in turns for w, s, e in t.words],
    }
