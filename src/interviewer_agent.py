"""LiveKit worker entrypoint: `uv run python agent.py dev` locally, `lk agent deploy` for LiveKit Cloud.

Flow (BUILD_PLAN §4.3, §4.4, §5.2): browser: parse dispatch metadata, wait for the candidate identity ·
phone call-in (no metadata): claim the caller from the web by caller ID, else by the name they applied with →
start session (Insights recording off) → scripted opening with the recording notice → `notice_played` →
LLM-led interview → `call_ended` from the shutdown callback.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import time

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from livekit.agents.voice.room_io import AudioInputOptions, RoomOptions
from livekit.plugins import noise_cancellation

from interviewer import gemini_auth
from interviewer.agent import InterviewAgent, dumps, turn_record
from interviewer.config import Settings
from interviewer.events import EventsClient
from interviewer.inbound import (LINES, NAME_QUESTION, STANDALONE_NAME_QUESTION, UNMATCHED_LINE, InboundClient,
                                 first_word, standalone_metadata)
from interviewer.evals.readiness import run_readiness
from interviewer.metadata import DispatchMetadata, parse_dispatch_metadata, parse_readiness_metadata, peek_kind
from interviewer.outcomes import Outcome, derive_outcome
from interviewer.prompt import opening_line
from interviewer.session_opts import LLM_BEHAVIOUR, make_llm, make_tts
from interviewer.silence import SilenceTimer
from interviewer.state import InterviewState

load_dotenv(".env.local")
gemini_auth.configure()
log = logging.getLogger("interviewer")

server = AgentServer(num_idle_processes=2, drain_timeout=600)



def build_session(meta: DispatchMetadata, s: Settings) -> AgentSession:
    if s.stt_model.startswith("deepgram/flux"):
        stt = inference.STT(s.stt_model, extra_kwargs={"eot_threshold": 0.7, "eager_eot_threshold": 0.5})
        turn_detection = "stt"
    else:
        stt = inference.STT(s.stt_model, language="en",  # fillers kept for delivery signals (BUILD_PLAN §5.7)
                            extra_kwargs={"interim_results": True, "smart_format": False, "filler_words": True})
        turn_detection = inference.TurnDetector()
    return AgentSession(
        stt=stt,
        llm=make_llm(s.gemini_live_model),
        tts=make_tts(s.tts_model, s.cartesia_voice_id),
        stt_context_options={"keyterms": meta.keyterms[:50]},
        turn_handling={
            "turn_detection": turn_detection,
            "endpointing": {"min_delay": 0.5, "max_delay": s.endpointing_max_delay},
            "interruption": {"min_duration": 0.5, "min_words": 0, "false_interruption_timeout": 2.0,
                             "resume_false_interruption": True},
            "preemptive_generation": {"enabled": True, "preemptive_tts": False},
        },
        user_away_timeout=None,  # SilenceTimer owns silence handling
        **LLM_BEHAVIOUR,
    )


async def ask_once(ctx: JobContext, s: Settings, identity: str, question: str, timeout: float = 15.0) -> str:
    """Speak one question and return the caller's next final transcript ("" on timeout). STT + TTS only, no LLM."""
    session = AgentSession(stt=inference.STT(s.stt_model, language="en"), tts=make_tts(s.tts_model, s.cartesia_voice_id))
    heard: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    @session.on("user_input_transcribed")
    def _final(ev) -> None:
        if ev.is_final and ev.transcript.strip() and not heard.done():
            heard.set_result(ev.transcript.strip())

    await session.start(agent=Agent(instructions="Say only what you are told."), room=ctx.room, record=False,
                        room_options=RoomOptions(participant_identity=identity))
    try:
        # Bounded so a caller never waits on it. (Speech plays only once the caller subscribes to the agent's
        # audio; a phone caller always does. A test caller without --auto-subscribe never hears it.)
        await asyncio.wait_for(session.say(question, allow_interruptions=False).wait_for_playout(), 8)
        return await asyncio.wait_for(heard, timeout)
    except asyncio.TimeoutError:
        log.warning("name question got no answer (or never played)")
        return ""
    finally:
        await session.aclose()


async def say_and_close(ctx: JobContext, s: Settings, identity: str, line: str) -> None:
    session = AgentSession(tts=make_tts(s.tts_model, s.cartesia_voice_id))
    await session.start(agent=Agent(instructions="Say only what you are told."), room=ctx.room, record=False,
                        room_options=RoomOptions(participant_identity=identity))
    try:
        await asyncio.wait_for(session.say(line, allow_interruptions=False).wait_for_playout(), 10)
    except asyncio.TimeoutError:
        log.warning("closing line never played out")
    finally:
        await session.aclose()
        await ctx.delete_room()


async def claim_inbound(ctx: JobContext, s: Settings) -> DispatchMetadata | None:
    """Phone call-in (BUILD_PLAN §4.4): match the caller to an application waiting for a call, or turn them away."""
    participant = await ctx.wait_for_participant()
    caller = participant.attributes.get("sip.phoneNumber") or None
    identity = participant.identity
    ctx.log_context_fields = {"kind": "inbound", "room": ctx.room.name}
    if s.standalone_role:
        name = first_word(await ask_once(ctx, s, identity, STANDALONE_NAME_QUESTION))
        return standalone_metadata(s.standalone_role, room=ctx.room.name, participant_identity=identity, first_name=name)

    client = InboundClient(base_url=s.events_base_url, secret=s.events_secret)
    try:
        meta, reason = await client.claim(room=ctx.room.name, participant_identity=identity, caller_e164=caller) \
            if caller else (None, "no_match")
        if meta is None and reason == "no_match":  # caller ID withheld or a different phone: ask for the name once
            name = await ask_once(ctx, s, identity, NAME_QUESTION)
            if name:
                meta, reason = await client.claim(room=ctx.room.name, participant_identity=identity,
                                                  caller_e164=caller, name=name)
    finally:
        await client.aclose()
    if meta is not None:
        log.info("inbound claimed by %s call_id=%s", "caller id" if caller and meta.phone_e164 == caller else "name",
                 meta.call_id)
        return meta

    events = EventsClient(base_url=s.events_base_url, secret=s.events_secret,
                          call_id=f"inbound-{ctx.room.name}", attempt_id="inbound")
    events.start()
    events.emit("inbound_rejected", {"caller": caller, "room": ctx.room.name, "reason": reason})
    await events.close()
    await say_and_close(ctx, s, identity, LINES.get(reason, UNMATCHED_LINE))
    return None


AGENT_NAME = os.environ.get("AGENT_NAME", "interviewer")  # must match the dispatch rule


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    s = Settings.from_env()
    t_entry = time.monotonic()
    if peek_kind(ctx.job.metadata) == "readiness":
        # Role readiness gate (BUILD_PLAN §4.10): text-mode persona runs in an empty room, report posted to the web.
        rmeta = parse_readiness_metadata(ctx.job.metadata)
        ctx.log_context_fields = {"kind": "readiness", "run_id": rmeta.run_id, "room": ctx.room.name}
        await ctx.connect()
        try:
            await run_readiness(ctx, s, rmeta)
        finally:
            await ctx.delete_room()
        return
    meta = parse_dispatch_metadata(ctx.job.metadata)
    # Connect before session.start: with a fixed participant_identity, RoomIO's transcription output reads
    # local_participant while starting and crashes ("cannot access local participant before connecting").
    await ctx.connect()
    if meta is None:
        if s.kill_switch:
            log.warning("KILL_SWITCH set; not taking the call")
            await ctx.delete_room()
            return
        meta = await claim_inbound(ctx, s)
        if meta is None:
            return
    ctx.log_context_fields = {"application_id": meta.application_id, "attempt_id": meta.attempt_id,
                              "call_id": meta.call_id, "room": ctx.room.name}
    if s.kill_switch:
        log.warning("KILL_SWITCH set; not starting the interview")
        await ctx.delete_room()
        return

    t_job = time.monotonic()
    startup: dict[str, int | None] = {}
    state = InterviewState(meta)
    events = EventsClient(base_url=s.events_base_url, secret=s.events_secret,
                          call_id=meta.call_id, attempt_id=meta.attempt_id, local=bool(s.standalone_role))
    events.start()

    async def publish(topic: str, data: dict) -> None:
        try:
            await ctx.room.local_participant.send_text(dumps(data), topic=topic)
        except Exception as e:  # observers are optional; never break the call for them
            log.debug("publish %s failed: %s", topic, e)

    session = build_session(meta, s)
    agent = InterviewAgent(meta, state, events, publish)
    agent._last_guardrail_ms = None
    turn_count = 0

    # ---- silence handling (one episode at a time) ----
    async def hang_up(reason: str) -> None:
        silence.stop()
        try:
            await ctx.delete_room()
        finally:
            ctx.shutdown(reason)

    # The timer never speaks once the goodbye is under way (call e17a0efa: "Take your time." after end_call).
    def closing() -> bool:
        return state.stage in ("close", "ended")

    async def nudge() -> None:
        if closing():
            return
        state.silence_events.append("nudge")
        await session.say("Take your time.")

    async def reask() -> None:
        if closing():
            return
        state.silence_events.append("reask")
        qid = state.next_knockout() if state.stage == "knockouts" else state.next_question()
        q = next((x for x in meta.plan.questions if x.id == qid), None)
        if q and q.short:
            await session.say(q.short)
        else:
            await session.generate_reply(instructions="The candidate has been silent. Ask your last question again, in fewer words.")

    async def giveup() -> None:
        if closing():
            return
        state.silence_events.append("giveup")
        state.explicit_outcome = Outcome.SILENCE_TIMEOUT
        await session.say("I'll follow up by email. Thanks for your time.", allow_interruptions=False)
        await hang_up("silence_timeout")

    silence = SilenceTimer(on_nudge=nudge, on_reask=reask, on_giveup=giveup)

    @session.on("agent_state_changed")
    def _agent_state(ev) -> None:
        if ev.new_state == "listening" and ev.old_state == "speaking" and state.stage not in ("open", "close", "ended"):
            # People think before telling a story: wait longer after a core question.
            silence.arm(*((10.0, 20.0) if state.stage == "core" else ()))

    @session.on("user_state_changed")
    def _user_state(ev) -> None:
        if ev.new_state == "speaking":
            silence.cancel()

    @session.on("error")
    def _error(ev) -> None:
        # A non-retryable LLM error (e.g. Gemini 5xx after partial output) drops the reply; the session tolerates a
        # few of these, so ask the candidate to repeat instead of leaving dead air.
        err = ev.error
        if getattr(err, "type", "") == "llm_error" and not err.recoverable and state.stage not in ("close", "ended"):
            log.warning("llm error, asking candidate to repeat: %s", err.error)
            asyncio.create_task(session.say("Sorry, I lost you for a second. Could you say that again?"))

    @session.on("conversation_item_added")
    def _item(ev) -> None:
        nonlocal turn_count
        rec = turn_record(ev.item, guardrail_ms=getattr(agent, "_last_guardrail_ms", None))
        if rec is None:
            return
        turn_count += 1
        rec["seq"] = turn_count
        if rec["role"] == "candidate":
            rec["words"] = agent.take_words()
        if rec["interrupted"]:
            state.interruptions += 1
        events.emit("turn", rec, key=rec["turn_id"])
        asyncio.create_task(publish("interview.metrics", {"turn_id": rec["turn_id"], "role": rec["role"],
                                                          "seq": turn_count, **rec["metrics"]}))

    @ctx.room.on("participant_connected")
    def _observer(p: rtc.RemoteParticipant) -> None:
        if p.identity.startswith("observer-"):
            asyncio.create_task(publish("interview.state", state.snapshot()))
        if p.identity == meta.candidate_identity:
            state.candidate_joined = True

    @ctx.room.on("participant_disconnected")
    def _left(p: rtc.RemoteParticipant) -> None:
        if p.identity == meta.candidate_identity and state.stage != "ended":
            silence.stop()
            ctx.shutdown("candidate_left")

    async def on_shutdown(reason: str) -> None:
        silence.stop()
        outcome = derive_outcome(explicit=state.explicit_outcome, closed_normally=state.closed_normally,
                                 interview_answers=state.interview_answers, candidate_joined=state.candidate_joined)
        state.stage = "ended"
        events.emit("call_ended", {
            "outcome": outcome.value, "reason": reason, "duration_s": round(state.elapsed_s, 1),
            "transcript_rev": turn_count, "knockouts": state.knockouts, "answers": state.answers,
            "followups": state.followups, "guardrail_events": len(state.guardrail_events),
            "interruptions": state.interruptions, "silence_events": state.silence_events, "startup_ms": startup,
            "plan_source": meta.plan.plan_source,
        }, key="final")
        await events.close(timeout=15)
        if s.standalone_role:
            # Post-call observation block (standalone/demo only): AI-assisted-answer signals from this call's events.
            from interviewer.ai_text import answer_texts, observe, report_lines
            from interviewer.delivery import answer_delivery
            try:
                deliveries = {d.question_id: d for d in answer_delivery(events.sent)}
                texts = answer_texts(events.sent)
                for line in report_lines([observe(q, texts.get(q, ""), deliveries.get(q)) for q in texts]):
                    log.info(line)
            except Exception as e:  # never let the report break shutdown
                log.warning("post-call observation report failed: %s", e)

    ctx.add_shutdown_callback(on_shutdown)

    nc = noise_cancellation.BVCTelephony() if meta.mode == "inbound" else noise_cancellation.BVC()
    await session.start(
        agent=agent, room=ctx.room, record=False,  # Agent insights would record before the notice (ADR-013)
        room_options=RoomOptions(participant_identity=meta.candidate_identity,
                                 audio_input=AudioInputOptions(noise_cancellation=nc)),
    )
    events.emit("call_started", {"mode": meta.mode, "room": ctx.room.name,
                                 "prompt_version": meta.runtime_config.prompt_version,
                                 "rubric_version": meta.runtime_config.rubric_version,
                                 "endpointing_max_delay": s.endpointing_max_delay})

    # ---- reach the candidate ----
    if meta.mode == "inbound":
        # The caller is already on the line (claim_inbound); nothing to dial (ADR-016).
        state.candidate_joined = True
        startup["claimed_ms"] = round((t_job - t_entry) * 1000)  # job start → caller matched
    else:
        try:
            await asyncio.wait_for(ctx.wait_for_participant(identity=meta.candidate_identity), timeout=20)
        except asyncio.TimeoutError:
            state.explicit_outcome = Outcome.BROWSER_NOT_READY
            await hang_up("browser_not_ready")
            return
        state.candidate_joined = True

    # ---- opening: AI disclosure + recording notice (stated, not asked) ----
    state.stage = "identity"
    handle = session.say(opening_line(meta), allow_interruptions=True)
    startup["first_audio_requested_ms"] = round((time.monotonic() - t_job) * 1000)
    await handle.wait_for_playout()
    if not handle.interrupted:
        state.notice_played = True
        events.emit("notice_played", {"at": dt.datetime.now(dt.timezone.utc).isoformat()})
    else:
        # Interrupted before the notice finished: say it again on its own before anything is recorded.
        h2 = session.say("Quick note: this call is recorded so the hiring team can review your application.")
        await h2.wait_for_playout()
        state.notice_played = True
        events.emit("notice_played", {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "repeated": True})
    await publish("interview.state", state.snapshot())

    async def call_cap() -> None:
        await asyncio.sleep(meta.max_call_s - 30)
        if state.stage != "ended":
            await session.say("We're at time, so I'll stop here. The team will be in touch by email. Thanks!",
                              allow_interruptions=False)
            await hang_up("max_duration")

    asyncio.create_task(call_cap())

    async def heartbeat() -> None:
        # Lets the web tell a quiet call from a dead worker (BUILD_PLAN §4.3 step 4).
        while state.stage != "ended":
            await asyncio.sleep(60)
            events.emit("heartbeat", {"stage": state.stage, "elapsed_s": round(state.elapsed_s)})

    asyncio.create_task(heartbeat())


if __name__ == "__main__":
    cli.run_app(server)
