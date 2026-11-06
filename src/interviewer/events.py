"""Agent → web events (BUILD_PLAN §4.7). HMAC-signed, idempotent, ordered by seq, never blocking a turn.

Signature: X-Signature = hex(HMAC-SHA256(secret, timestamp + "." + raw_body)), X-Timestamp = unix seconds.
The idempotency key is fixed when the event is created; each retry re-signs with a fresh timestamp.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import pathlib
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger("interviewer.events")


def sign(secret: str, timestamp: str, body: bytes) -> str:
    return hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()


class EventsClient:
    def __init__(self, *, base_url: str, secret: str, call_id: str, attempt_id: str,
                 client: httpx.AsyncClient | None = None, max_attempts: int = 6, local: bool = False) -> None:
        self._url = base_url.rstrip("/") + "/api/agent/events"
        self._local = local  # standalone pivot mode: log events instead of posting (BUILD_PLAN §4.4)
        self._secret = secret
        self.call_id = call_id
        self.attempt_id = attempt_id
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0))
        self._own_client = client is None
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._seq = 0
        self._max_attempts = max_attempts
        self._task: asyncio.Task | None = None
        self.sent: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._worker())

    def emit(self, type_: str, payload: dict[str, Any] | None = None, *, key: str | None = None) -> dict[str, Any]:
        """Queue an event; returns immediately. `key` makes the idempotency key semantic (e.g. turn id)."""
        self._seq += 1
        ev = {
            "schema_version": 1,
            "call_id": self.call_id,
            "attempt_id": self.attempt_id,
            "seq": self._seq,
            "idempotency_key": f"{self.call_id}:{type_}:{key or self._seq}",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "type": type_,
            "payload": payload or {},
        }
        self._queue.put_nowait(ev)
        return ev

    async def close(self, timeout: float = 10.0) -> None:
        """Flush everything queued (used in the shutdown callback), then stop."""
        self._queue.put_nowait(None)
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout)
            except asyncio.TimeoutError:
                log.error("events flush timed out; %d events unsent", self._queue.qsize())
        if self._own_client:
            await self._client.aclose()

    async def _worker(self) -> None:
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            await self._post(ev)

    async def _post(self, ev: dict[str, Any]) -> None:
        body = json.dumps(ev, separators=(",", ":")).encode()
        if self._local:
            log.info("event (local) %s", body.decode()[:500])
            self.sent.append(ev)
            try:  # full record for post-call analysis (evals/ai_detect/analyze_call.py); gitignored
                dump_dir = pathlib.Path(os.environ.get("EVENTS_DUMP_DIR", "calls"))
                dump_dir.mkdir(parents=True, exist_ok=True)
                with (dump_dir / f"{self.call_id}.events.jsonl").open("a") as f:
                    f.write(body.decode() + "\n")
            except OSError as e:
                log.warning("event dump failed: %s", e)
            return
        delay = 0.25
        for attempt in range(1, self._max_attempts + 1):
            ts = str(int(time.time()))
            headers = {"content-type": "application/json", "x-timestamp": ts,
                       "x-signature": sign(self._secret, ts, body), "x-request-id": str(uuid.uuid4())}
            try:
                r = await self._client.post(self._url, content=body, headers=headers)
                if r.status_code < 300 or r.status_code == 409:
                    # 409 = rejected as stale/illegal by business rules: retrying won't help.
                    self.sent.append(ev)
                    if r.status_code == 409:
                        log.warning("event rejected 409 type=%s seq=%s body=%s", ev["type"], ev["seq"], r.text[:200])
                    return
                if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                    log.error("event rejected %s type=%s: %s", r.status_code, ev["type"], r.text[:200])
                    self.failed.append(ev)
                    return
            except httpx.HTTPError as e:
                log.warning("event post failed (attempt %d) type=%s: %s", attempt, ev["type"], e)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 4.0)
        log.error("event dropped after %d attempts type=%s seq=%s", self._max_attempts, ev["type"], ev["seq"])
        self.failed.append(ev)
