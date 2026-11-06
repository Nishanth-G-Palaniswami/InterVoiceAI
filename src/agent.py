"""Entry point: wires the voice pipeline and picks which agent answers.

You mostly edit my_agent.py. Come here to change models, silence handling or
turn-taking.

  .env.local
    AGENT_PROFILE = booking | screener | mine     (which agent answers)
    MODEL_PRESET  = budget  | quality             (see models.py)
    AGENT_NAME    = take2-agent                   (must match your dispatch rule)
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    STTContextOptions,
    TurnHandlingOptions,
    UserStateChangedEvent,
    cli,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics

import candidates
from booking_agent import BookingAgent
from models import load_models
from my_agent import InterviewAgent
from postcall import attach_postcall
from question_plan import generate_plan
from screener_agent import ScreenerAgent

logger = logging.getLogger("agent")

load_dotenv(".env.local")

AGENTS: dict[str, type[Agent]] = {
    "booking": BookingAgent,
    "screener": ScreenerAgent,
}
INTERVIEW_PROFILE = "mine"

AGENT_NAME = os.getenv("AGENT_NAME", "take2-agent")

# Silence: after this many quiet seconds the caller counts as "away".
# Interview candidates think before answering, so this is longer than the
# receptionist default of eight seconds.
SILENCE_SECONDS = 12.0
# Check in this many times, then say goodbye and hang up.
MAX_CHECK_INS = 2
# Console and the text debugger have no room participant; don't wait on one.
PARTICIPANT_WAIT_SECONDS = 1.5


def agent_profile() -> str:
    profile = os.getenv("AGENT_PROFILE", "booking").strip().lower()
    if profile != INTERVIEW_PROFILE and profile not in AGENTS:
        choices = ", ".join([*AGENTS, INTERVIEW_PROFILE])
        raise ValueError(f"Unknown AGENT_PROFILE '{profile}'. Choose one of: {choices}")
    return profile


async def caller_number(ctx: JobContext) -> str | None:
    """The caller ID of an inbound phone call, or None for any other session."""
    try:
        participant = await asyncio.wait_for(
            ctx.wait_for_participant(), timeout=PARTICIPANT_WAIT_SECONDS
        )
    except asyncio.TimeoutError:
        return None
    if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        return None
    return participant.attributes.get("sip.phoneNumber")


async def build_interview_agent(ctx: JobContext) -> InterviewAgent:
    """Match the caller to a profile by caller ID and make sure it has a plan."""
    number = await caller_number(ctx)
    candidate = candidates.find_by_phone(number)
    if candidate is not None and candidate["plan"] is None:
        candidate["plan"] = await generate_plan(candidate)
        candidates.save(candidate)
    logger.info(
        "caller %s -> %s", number, candidate["id"] if candidate else "no profile"
    )
    return InterviewAgent(
        candidate=candidate,
        caller_number=number,
        call_id=ctx.room.name,
        matched_by="phone" if candidate else None,
    )


def handle_silence(session: AgentSession) -> None:
    """Ask "are you still there?" when the caller goes quiet, then hang up."""
    check_ins = 0
    tasks: set[asyncio.Task[None]] = set()

    async def check_in() -> None:
        nonlocal check_ins
        check_ins += 1
        if check_ins <= MAX_CHECK_INS:
            await session.generate_reply(
                instructions="The caller has gone quiet. In a few words, ask if they are still there."
            )
        else:
            await session.generate_reply(
                instructions="The caller is not responding. Say a short, polite goodbye."
            )
            session.shutdown()

    @session.on("user_state_changed")
    def _on_user_state(ev: UserStateChangedEvent) -> None:
        nonlocal check_ins
        if ev.new_state == "speaking":
            check_ins = 0
        elif ev.new_state == "away":
            task = asyncio.create_task(check_in())
            tasks.add(task)
            task.add_done_callback(tasks.discard)


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    models = load_models()
    profile = agent_profile()

    # Connect first so the caller's SIP attributes are readable before the
    # session starts (session.start connects on its own otherwise).
    await ctx.connect()

    keyterms = ["Take2", "LiveKit"]
    agent: Agent
    if profile == INTERVIEW_PROFILE:
        agent = await build_interview_agent(ctx)
        if agent.candidate is not None:
            keyterms += [agent.candidate["name"], agent.candidate["company"]]
    else:
        agent = AGENTS[profile]()

    session = AgentSession(
        # Ears: speech-to-text
        stt=models.stt,
        # Bias transcription toward words it would otherwise misspell.
        stt_context_options=STTContextOptions(
            keyterms=[k for k in keyterms if k],
            keyterm_detection={"enabled": True},
        ),
        # Brain: the LLM
        llm=models.llm,
        # Voice: text-to-speech
        tts=models.tts,
        expressive=models.expressive,
        turn_handling=TurnHandlingOptions(
            # Decides when the caller has finished speaking, using both the words
            # and the audio (pitch, pauses). Much better than a fixed silence timer.
            turn_detection=inference.TurnDetector(),
            # Start drafting a reply before the turn is confirmed. Cuts latency.
            preemptive_generation={"enabled": True},
        ),
        user_away_timeout=SILENCE_SECONDS,
    )

    handle_silence(session)

    async def log_usage() -> None:
        # Shows model usage per call so you can see what you're spending.
        logger.info("session usage: %s", session.usage)

    ctx.add_shutdown_callback(log_usage)

    if isinstance(agent, InterviewAgent):
        interview = agent

        async def finish_record() -> None:
            # A hang-up before finish_interview still leaves a record; then the
            # post-call assessment is added, after the caller is gone.
            interview.write_call("dropped")
            await interview.assess()

        ctx.add_shutdown_callback(finish_record)
        # Live event dump (calls/<room>.events.jsonl) for the post-interview
        # analysis lane and its dashboard. record_answer runs twice per question
        # (the first call is the follow-up gate), so the agent emits the
        # tool_record_answer event itself, once, when an answer is recorded.
        # The identity tools are passed as the lane's "knockout" boundary: its
        # answer splitter resets at that event, so the name/confirmation turns
        # never count as the first answer.
        interview.postcall = attach_postcall(
            session,
            agent,
            ctx.room.name,
            answer_tools=set(),
            knockout_tools={"claim_by_name", "confirm_identity"},
            meta={
                "candidate_id": agent.candidate["id"] if agent.candidate else None,
                "matched_by": agent.matched_by,
            },
        )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Removes background noise and other voices before STT hears it.
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
            # Hang up the phone line when the session ends.
            delete_room_on_close=True,
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
