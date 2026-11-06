"""InterviewAI: the phone interviewer (AGENT_PROFILE=mine).

It asks four questions grounded in the caller's resume, forces one unscripted
follow-up per answer, and records what it heard for the recruiter.

  known caller   -> greet by name, confirm identity, planned questions
  unknown caller -> ask the name they applied with, claim_by_name, then the same
  each answer    -> record_answer gate: the first call asks for ONE detail
                    follow-up, the second call records the answer
  end            -> finish_interview writes data/calls/<room>.json (contract C2)
                    with the integrity observations from interviewer.delivery
                    (voice side) and interviewer.ai_text (text side)

The candidate's turns are read from the session history, so the same code works
on a phone call, in the Agent Console and in the text debugger. Per-word STT
timings and the response onset are kept per turn for the voice-side signals.
"""

from __future__ import annotations

import asyncio
import logging
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
from assessment import ASSESSMENT_NOTE, assess_call
from candidates import Candidate, Question
from interviewer.ai_text import Observation, call_confidence, observe, report_lines
from interviewer.delivery import AnswerDelivery, CandTurn, features, judge
from prompts import build_instructions
from question_plan import generate_plan

logger = logging.getLogger("interview-agent")

AGENT_PERSONA = "Nia"
RESUME_CHARS_IN_PROMPT = 1500
INTEGRITY_NOTE = (
    "Observations for the recruiter, not a decision. Thresholds are uncalibrated."
)
# call_confidence bands -> the C2 verdict words the scorecard shows
BAND_VERDICT = {"low": "natural", "medium": "inconclusive", "high": "likely_assisted"}

Word = tuple[str, float, float]  # text, start_s, end_s (STT stream time)


@dataclass
class Turn:
    """One candidate turn as the pipeline delivered it."""

    item_id: str
    text: str
    words: list[Word]
    onset_ms: int | None  # silence between the agent's last word and this turn


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
   recorded, and confirm you are speaking with {candidate["name"]}. Once they
   confirm, call confirm_identity.
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
- If the candidate wants to stop, call finish_interview with
  candidate_wants_to_stop set to true, then end the call.
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
        self._words: list[Word] = []  # timed words of the turn in progress
        self._timings: dict[str, tuple[list[Word], int | None]] = {}  # by item id
        self._taken: set[str] = set()  # user items already assigned to an answer
        self._agent_stopped_at: float | None = None
        self._user_started_at: float | None = None
        self._probe_logged = False
        self._written: Path | None = None
        self._record: dict[str, Any] | None = None  # the C2 data as written
        self._assess_task: asyncio.Task[None] | None = None
        self.identity_confirmed: bool | None = None
        # user messages in the history when the identity check was asked;
        # confirm_identity is only valid once the candidate has spoken since
        self._identity_baseline = 0
        # postcall event dump (set by agent.py); emits tool_record_answer once
        # per recorded answer so the analysis lane splits main/follow-up right
        self.postcall: Any | None = None

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

    def take_words(self) -> list[list[Any]]:
        """Timed words of the latest candidate turn, relative to its first word.

        Read-only view for postcall.attach_postcall; it never clears the buffer.
        """
        words = self._words
        if not words and self._timings:
            words = next(reversed(self._timings.values()))[0]
        if not words:
            return []
        t0 = words[0][1]
        return [[w, round(s - t0, 2), round(e - t0, 2)] for w, s, e in words]

    def _on_agent_state(self, ev: AgentStateChangedEvent) -> None:
        if ev.new_state == "speaking":
            self._user_started_at = None
        elif ev.old_state == "speaking":
            self._agent_stopped_at = ev.created_at

    def _on_user_state(self, ev: UserStateChangedEvent) -> None:
        if ev.new_state == "speaking" and self._user_started_at is None:
            self._user_started_at = ev.created_at

    def _on_item_added(self, ev: ConversationItemAddedEvent) -> None:
        """A committed candidate message: file its word timings and onset."""
        if not isinstance(ev.item, llm.ChatMessage) or ev.item.role != "user":
            return
        onset_ms: int | None = None
        if self._agent_stopped_at is not None and self._user_started_at is not None:
            onset_ms = max(
                0, round((self._user_started_at - self._agent_stopped_at) * 1000)
            )
        words, self._words = self._words, []
        self._timings[ev.item.id] = (words, onset_ms)

    def take_turns(self, history: llm.ChatContext) -> list[Turn]:
        """Candidate messages since the agent last spoke, oldest first.

        Each message is handed out once; later calls only see newer messages.
        """
        collected: list[Turn] = []
        for item in reversed(history.items):
            if not isinstance(item, llm.ChatMessage):
                continue
            if item.role == "assistant":
                break
            if item.role != "user" or item.id in self._taken:
                continue
            words, onset_ms = self._timings.get(item.id, ([], None))
            collected.append(Turn(item.id, item.text_content or "", words, onset_ms))
        collected.reverse()
        self._taken.update(t.item_id for t in collected)
        return collected

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
                f"general screen now and the recruiting team will follow up about "
                f"applying. Then ask {first['id']} exactly as written: {first['text']}"
            )
        if match["plan"] is None:
            match["plan"] = await generate_plan(match)
            candidates.save(match)
        self._set_candidate(match, "name")
        await self.update_instructions(build_instructions(render_prompt(match)))
        self._identity_baseline = _user_messages(self.session.history)
        return (
            f"Found {match['name']}, applying for {match['role']}. Ask them to "
            f"confirm that is them, wait for the answer, then call confirm_identity."
        )

    @function_tool
    async def confirm_identity(self, context: RunContext, confirmed: bool) -> str:
        """Record whether the caller confirmed they are the candidate on file. Call it right after they answer the identity check, before the first question.

        Args:
            confirmed: true if they confirmed, false if they said they are someone else.
        """
        if _user_messages(self.session.history) <= self._identity_baseline:
            return (
                "Not yet: the candidate has not answered the identity check. Ask "
                "them to confirm, wait for the answer, then call confirm_identity."
            )
        self.identity_confirmed = confirmed
        if not confirmed or not self._answers:
            return "Ask for the full name they applied with, then call claim_by_name."
        first = next(iter(self._answers.values())).question
        return (
            f"Identity confirmed. Ask {first['id']} exactly as written: {first['text']}"
        )

    @function_tool
    async def record_answer(
        self, context: RunContext, question_id: str, summary: str, quality_hint: str
    ) -> str:
        """Record the candidate's answer to one planned question. Call it right after they answer, before you say anything else. The first call for a question asks you for a follow-up instead of recording; do that, then call record_answer again with the same question_id.

        Args:
            question_id: The planned question's id, q1 to q4.
            summary: One neutral sentence summarising what the candidate said.
            quality_hint: "adequate" if they gave specifics, "thin" if vague or short, "no_answer" only if they explicitly declined, said they do not know, or asked to skip.
        """
        answer = self._answers.get(question_id)
        if answer is None:
            return f"Unknown question_id {question_id!r}. Valid ids: {', '.join(self._answers)}."
        turns = self.take_turns(self.session.history)
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
        if self.postcall is not None:
            self.postcall.emit(
                "tool_record_answer",
                {
                    "question_id": question_id,
                    "summary": summary[:800],
                    "quality_hint": quality_hint,
                },
            )
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
    async def finish_interview(
        self, context: RunContext, candidate_wants_to_stop: bool
    ) -> str:
        """Save the interview and move to the goodbye. Call it on its own, after the last record_answer result has come back, never in the same turn as record_answer.

        Args:
            candidate_wants_to_stop: true only if the candidate asked to stop or leave before every question was recorded.
        """
        return self.finish(candidate_wants_to_stop)

    def finish(self, candidate_wants_to_stop: bool) -> str:
        pending = [a for a in self._answers.values() if not a.recorded]
        if pending and not candidate_wants_to_stop:
            first = pending[0]
            step = (
                "ask its follow-up, then call record_answer again"
                if first.follow_up_requested
                else "ask it, then call record_answer"
            )
            return (
                f"Not finished: {first.question['id']} is not recorded yet. "
                f"{step[0].upper()}{step[1:]}: {first.question['text']} "
                "Call finish_interview with candidate_wants_to_stop=true only if "
                "the candidate asked to stop."
            )
        self.write_call("completed" if not pending else "stopped")
        # The assessment runs while the goodbye is being spoken; the shutdown
        # callback awaits it, so a hang-up right after still gets it.
        self.start_assessment()
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
        scored = [score_answer(a) for a in self._answers.values()]
        observations = [obs for _, obs in scored if obs is not None]
        data: dict[str, Any] = {
            "schema_version": 1,
            "call_id": self.call_id,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "candidate_id": candidate["id"] if candidate else None,
            "candidate_name": candidate["name"] if candidate else None,
            "caller_number": self.caller_number,
            "matched_by": self.matched_by,
            "identity_confirmed": self.identity_confirmed,
            "outcome": outcome,
            "answers": [entry for entry, _ in scored],
            "integrity": call_integrity(observations),
        }
        self._written = candidates.save_call(self.call_id, data)
        self._record = data
        logger.info("call record written: %s (%s)", self._written, outcome)
        for line in report_lines(observations):
            logger.info(line)
        return self._written

    def start_assessment(self) -> asyncio.Task[None] | None:
        """Kick off the post-call assessment once, if an event loop is running
        (it always is on a call); later calls return the same task."""
        if self._assess_task is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return None
            self._assess_task = loop.create_task(self._assess())
        return self._assess_task

    async def assess(self) -> None:
        """Make sure the assessment has been added to the written record.
        Called at shutdown (after the caller is gone), so it never delays a reply."""
        task = self.start_assessment()
        if task is not None:
            await task

    async def _assess(self) -> None:
        if self._record is None or "assessment" in self._record:
            return
        try:
            result = await assess_call(self.candidate, self._record["answers"])
        except Exception as err:  # recorded in the file and the log, never hidden
            logger.exception("assessment failed for %s", self.call_id)
            result = {
                "status": "failed",
                "error": f"{type(err).__name__}: {err}",
                "note": ASSESSMENT_NOTE,
            }
        self._record["assessment"] = result
        candidates.save_call(self.call_id, self._record)
        logger.info("assessment saved for %s: %s", self.call_id, result["status"])


def _user_messages(history: llm.ChatContext) -> int:
    return sum(
        1
        for item in history.items
        if isinstance(item, llm.ChatMessage) and item.role == "user"
    )


# ---------- scoring glue over interviewer.delivery / interviewer.ai_text ----------


def _cand_turns(turns: list[Turn]) -> list[CandTurn]:
    return [
        CandTurn(
            turn_id=t.item_id,
            text=t.text,
            words=[[w, round(s, 2), round(e, 2)] for w, s, e in t.words],
        )
        for t in turns
    ]


def _side(turns: list[Turn], feats: Any) -> dict[str, Any]:
    text = " ".join(t.text for t in turns).strip()
    out = {"text": text, "word_count": len(text.split())}
    out.update(vars(feats))  # words, seconds, wpm, *_per100
    return out


def score_answer(a: Answer) -> tuple[dict[str, Any], Observation | None]:
    """One C2 answer entry plus the observation used for the call verdict.

    A question the candidate never answered gets no observation at all.
    """
    if not a.main:
        entry = {
            "question_id": a.question["id"],
            "question": a.question["text"],
            "summary": a.summary,
            "quality_hint": a.quality_hint,
            "recorded": a.recorded,
            "onset_ms": None,
            "main": None,
            "follow_up": None,
            "signals": [],
            "sounds_read": False,
            "verdict": "not_answered",
            "text_signals": None,
        }
        return entry, None
    main_f = features(_cand_turns(a.main))
    follow_f = features(_cand_turns(a.follow_up)) if a.follow_up else None
    signals, sounds_read, eligible = judge(main_f, follow_f)
    onset_ms = a.main[0].onset_ms if a.main else None
    delivery = AnswerDelivery(
        question_id=a.question["id"],
        onset_ms=onset_ms,
        main=main_f,
        follow_up=follow_f,
        signals=signals,
        sounds_read=sounds_read,
        eligible=eligible,
        turn_ids=[t.item_id for t in [*a.main, *a.follow_up]],
    )
    main_text = " ".join(t.text for t in a.main).strip()
    obs = observe(a.question["id"], main_text, delivery)
    entry = {
        "question_id": a.question["id"],
        "question": a.question["text"],
        "summary": a.summary,
        "quality_hint": a.quality_hint,
        "recorded": a.recorded,
        "onset_ms": onset_ms,
        "main": _side(a.main, main_f),
        "follow_up": _side(a.follow_up, follow_f) if follow_f else None,
        "signals": obs.evidence,
        "sounds_read": sounds_read,
        "verdict": obs.verdict,
        "text_signals": obs.text.to_dict(),
    }
    return entry, obs


def call_integrity(observations: list[Observation]) -> dict[str, Any]:
    conf = call_confidence(observations)
    return {
        "verdict": BAND_VERDICT[conf["band"]],
        "band": conf["band"],
        "score": conf["score"],
        "flagged_answers": len(conf["flagged"]),
        "flagged": conf["flagged"],
        "strong": conf["strong"],
        "eligible_answers": conf["eligible_answers"],
        "evidence": [f"{o.question_id}: {e}" for o in observations for e in o.evidence],
        "note": INTEGRITY_NOTE,
    }
