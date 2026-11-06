"""Phone call-in (BUILD_PLAN §4.4, ADR-016). A call to the role's LiveKit number arrives with no dispatch metadata;
the agent claims it from the web, which returns the same metadata a browser dispatch carries (`mode: "inbound"`).

Claim order: caller ID, then the full name the candidate applied with (asked once). The web decides; the agent never
interviews an unmatched caller. Standalone mode (`STANDALONE_ROLE`) skips the web for the event-starter pivot.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
import uuid

import httpx

from .evals.meta import metadata_from_role
from .events import sign
from .metadata import DispatchMetadata, parse_dispatch_metadata

log = logging.getLogger("interviewer.inbound")

NAME_QUESTION = "Thanks for calling. What's the full name you applied with?"
STANDALONE_NAME_QUESTION = "Thanks for calling. Before we start, what's your first name?"
UNMATCHED_LINE = ("I can't find an application for this call. Please apply on our careers page first, "
                  "then call back. Goodbye.")
BUSY_LINE = "It looks like your interview is already in progress on another call. Goodbye."
ERROR_LINE = "Sorry, I can't start your interview right now. Please try again in a few minutes. Goodbye."
LINES = {"busy": BUSY_LINE, "error": ERROR_LINE}


class InboundClient:
    """`POST /api/agent/inbound`, signed like the events (X-Timestamp + X-Signature over the raw body)."""

    def __init__(self, *, base_url: str, secret: str, client: httpx.AsyncClient | None = None, attempts: int = 3) -> None:
        self._url = base_url.rstrip("/") + "/api/agent/inbound"
        self._secret = secret
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=3.0))
        self._own = client is None
        self._attempts = attempts

    async def claim(self, *, room: str, participant_identity: str, caller_e164: str | None = None,
                    name: str | None = None) -> tuple[DispatchMetadata | None, str]:
        """(metadata, reason). reason is "matched", "no_match", "busy", or "error"."""
        body = json.dumps({"room": room, "participant_identity": participant_identity,
                           "caller_e164": caller_e164, "name": name}, separators=(",", ":")).encode()
        for attempt in range(1, self._attempts + 1):
            ts = str(int(time.time()))
            try:
                r = await self._client.post(self._url, content=body, headers={
                    "content-type": "application/json", "x-timestamp": ts,
                    "x-signature": sign(self._secret, ts, body), "x-request-id": str(uuid.uuid4())})
                if r.status_code == 200:
                    return parse_dispatch_metadata(json.dumps(r.json()["metadata"])), "matched"
                if r.status_code == 404:
                    return None, "no_match"
                if r.status_code == 409:
                    return None, "busy"
                log.warning("claim attempt %d status=%s body=%s", attempt, r.status_code, r.text[:200])
            except httpx.HTTPError as e:
                log.warning("claim attempt %d failed: %s", attempt, e)
        return None, "error"

    async def aclose(self) -> None:
        if self._own:
            await self._client.aclose()


def standalone_metadata(role_path: str, *, room: str, participant_identity: str, first_name: str) -> DispatchMetadata:
    """Pivot-only: interview any caller with one local role JSON and the newest checked-in prompt, no web."""
    role = json.loads(pathlib.Path(role_path).read_text())
    prompts = sorted(pathlib.Path(role_path).resolve().parent.glob("prompts/interviewer.v*.md"),
                     key=lambda p: int(p.stem.rsplit("v", 1)[1]))
    if not prompts:
        raise FileNotFoundError("no prompts/interviewer.v*.md next to the role file")
    meta = metadata_from_role(role, prompt_text=prompts[-1].read_text(), prompt_version=int(prompts[-1].stem.rsplit("v", 1)[1]),
                              first_name=first_name or "there", call_id=f"standalone-{room}", mode="inbound")
    return meta.model_copy(update={"candidate_identity": participant_identity})


def first_word(text: str) -> str:
    """'It's Maya.' → 'Maya'; used for the standalone first name only."""
    words = [w.strip(".,!?").capitalize() for w in text.split() if w.strip(".,!?")]
    for skip in ("It's", "Its", "I'm", "Im", "My", "Name's", "Name", "Is", "This", "Hi", "Hey", "Yeah", "Sure", "Um", "Uh"):
        while words and words[0] == skip:
            words.pop(0)
    return words[0] if words else ""
