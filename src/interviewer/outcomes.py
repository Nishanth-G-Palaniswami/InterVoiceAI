"""Call outcome enum and derivation (BUILD_PLAN §4.8). Policy lives in the web; the agent only reports."""

from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    NO_ANSWER = "no_answer"
    DROPPED = "dropped"
    WORKER_LOST = "worker_lost"  # set by the web only
    SILENCE_TIMEOUT = "silence_timeout"
    RECORDING_REFUSED = "recording_refused"
    RESCHEDULE_REQUESTED = "reschedule_requested"
    WRONG_PERSON = "wrong_person"
    TERMINATED_HOSTILE = "terminated_hostile"
    BROWSER_NOT_READY = "browser_not_ready"


PARTIAL_MIN_ANSWERS = 2


def derive_outcome(*, explicit: Outcome | None, closed_normally: bool, interview_answers: int,
                   candidate_joined: bool) -> Outcome:
    """explicit wins (set by a tool or flow branch). Otherwise the call ended by close or by hang-up."""
    if explicit is not None:
        return explicit
    if not candidate_joined:
        return Outcome.NO_ANSWER
    if closed_normally:
        return Outcome.COMPLETED
    return Outcome.PARTIAL if interview_answers >= PARTIAL_MIN_ANSWERS else Outcome.DROPPED
