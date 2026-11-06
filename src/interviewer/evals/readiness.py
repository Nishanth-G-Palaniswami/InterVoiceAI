"""Role readiness gate (BUILD_PLAN §4.10 step 4): run the role-agnostic personas against a draft role through the
real InterviewAgent in text mode, then post a signed report to the web. Runs as a LiveKit job (kind=readiness)
so it works wherever the worker runs; the job never touches Postgres."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx
from livekit.agents import APIStatusError
from livekit.agents.llm import LLM

from ..events import sign
from ..metadata import ReadinessMetadata
from .harness import check, run_persona
from .meta import metadata_from_role
from .personas import build_personas

log = logging.getLogger("interviewer.readiness")

MAX_TEXTS = 14
MAX_TEXT = 400


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trim(v: Any, n: int = 300) -> Any:
    if isinstance(v, str):
        return v[:n]
    if isinstance(v, dict):
        return {k: _trim(x, n) for k, x in v.items()}
    if isinstance(v, list):
        return [_trim(x, n) for x in v]
    return v


async def run_personas(meta: ReadinessMetadata, model_for: Callable[[str], LLM], tts: Any = None, *,
                       concurrency: int = 3, retry_failed: bool = True, model_name: str = "") -> dict:
    """Run every persona (a failing one gets one more attempt: Gemini is not deterministic) and build the report."""
    personas = build_personas(meta.role, meta.answer_bank)
    if meta.personas:
        personas = {k: v for k, v in personas.items() if k in meta.personas}
    sem = asyncio.Semaphore(concurrency)
    started = _now()

    async def one(name: str, p: dict) -> dict:
        async with sem:
            last: dict = {}
            for attempt in (1, 2) if retry_failed else (1,):
                dm = metadata_from_role(meta.role, prompt_text=meta.prompt_text, prompt_version=meta.prompt_version,
                                        first_name=meta.answer_bank.first_name, call_id=f"readiness-{meta.run_id}-{name}")
                t0 = time.monotonic()
                try:
                    run = await run_persona(dm, p, model_for(name), tts)
                except APIStatusError as e:  # vendor 429/5xx is not a role failure: one immediate retry
                    if not (e.status_code == 429 or e.status_code >= 500):
                        raise
                    run = await run_persona(dm, p, model_for(name), tts)
                except Exception as e:
                    last = {"name": name, "pass": False, "attempts": attempt, "fails": ["run error"],
                            "error": f"{type(e).__name__}: {str(e)[:300]}", "outcome": None, "tool_calls": [],
                            "agent_texts": [], "knockouts": {}, "answered": [], "guardrail_events": 0,
                            "duration_s": round(time.monotonic() - t0, 1)}
                    continue
                fails = check(p, run)
                last = {
                    "name": name, "pass": not fails, "attempts": attempt, "fails": fails, "outcome": run.outcome,
                    "tool_calls": [[n, _trim(a)] for n, a in run.tool_calls][:30],
                    "agent_texts": [t[:MAX_TEXT] for t in run.agent_texts][:MAX_TEXTS],
                    "knockouts": _trim(run.state.knockouts), "answered": list(run.state.answers),
                    "guardrail_events": len(run.state.guardrail_events), "duration_s": round(time.monotonic() - t0, 1),
                }
                if not fails:
                    break
            log.info("persona %s pass=%s attempts=%s", name, last.get("pass"), last.get("attempts"))
            return last

    tasks = {name: asyncio.create_task(one(name, p)) for name, p in personas.items()}
    done, pending = await asyncio.wait(tasks.values(), timeout=meta.max_run_s)
    for t in pending:
        t.cancel()
    results = []
    for name, t in tasks.items():
        if t in done and not t.cancelled() and t.exception() is None:
            results.append(t.result())
        else:
            err = "timed out" if t in pending else f"{type(t.exception()).__name__}: {str(t.exception())[:300]}"
            results.append({"name": name, "pass": False, "attempts": 0, "fails": [err], "error": err, "outcome": None,
                            "tool_calls": [], "agent_texts": [], "knockouts": {}, "answered": [], "guardrail_events": 0,
                            "duration_s": 0})
    passed = sum(r["pass"] for r in results)
    errored = sum(1 for r in results if r.get("error"))
    status = "timeout" if pending else ("pass" if passed == len(results) else "fail")
    return {
        "schema_version": 1, "run_id": meta.run_id, "status": status, "started_at": started, "finished_at": _now(),
        "model": model_name, "prompt_version": meta.prompt_version,
        "summary": {"total": len(results), "passed": passed, "failed": len(results) - passed, "errored": errored},
        "personas": results,
    }


class ReadinessClient:
    """Posts the report to the web with the same HMAC signing as EventsClient."""

    def __init__(self, *, base_url: str, secret: str, client: httpx.AsyncClient | None = None, max_attempts: int = 6) -> None:
        self._url = base_url.rstrip("/") + "/api/agent/readiness"
        self._secret = secret
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))
        self._own = client is None
        self._max = max_attempts

    async def post_report(self, report: dict) -> bool:
        body = json.dumps(report, separators=(",", ":")).encode()
        delay = 0.5
        try:
            for attempt in range(1, self._max + 1):
                ts = str(int(time.time()))
                try:
                    r = await self._client.post(self._url, content=body, headers={
                        "content-type": "application/json", "x-timestamp": ts,
                        "x-signature": sign(self._secret, ts, body), "x-request-id": str(uuid.uuid4())})
                    if r.status_code < 300 or r.status_code == 409:
                        return True
                    if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                        log.error("readiness report rejected status=%s body=%s", r.status_code, r.text[:300])
                        return False
                except httpx.HTTPError as e:
                    log.warning("readiness report post failed attempt=%s: %s", attempt, e)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 8)
            return False
        finally:
            if self._own:
                await self._client.aclose()


async def run_readiness(ctx: Any, settings: Any, meta: ReadinessMetadata) -> dict:
    """LiveKit job entry for kind=readiness. Keeps the (empty) room alive while it runs, then deletes it."""
    from ..session_opts import make_llm, make_tts

    client = ReadinessClient(base_url=settings.events_base_url, secret=settings.events_secret)
    if settings.kill_switch:
        report = {"schema_version": 1, "run_id": meta.run_id, "status": "error", "started_at": _now(), "finished_at": _now(),
                  "model": "", "prompt_version": meta.prompt_version, "error": "kill_switch",
                  "summary": {"total": 0, "passed": 0, "failed": 0, "errored": 0}, "personas": []}
    else:
        llm = make_llm(settings.gemini_live_model)
        tts = make_tts(settings.tts_model, settings.cartesia_voice_id)  # live session options: expressive markup guide
        try:
            report = await run_personas(meta, lambda _name: llm, tts, model_name=settings.gemini_live_model)
        except Exception as e:
            log.exception("readiness run crashed")
            report = {"schema_version": 1, "run_id": meta.run_id, "status": "error", "started_at": _now(),
                      "finished_at": _now(), "model": settings.gemini_live_model, "prompt_version": meta.prompt_version,
                      "error": f"{type(e).__name__}: {str(e)[:300]}",
                      "summary": {"total": 0, "passed": 0, "failed": 0, "errored": 0}, "personas": []}
    ok = await client.post_report(report)
    log.info("readiness %s status=%s %s/%s posted=%s", meta.run_id, report["status"],
             report["summary"]["passed"], report["summary"]["total"], ok)
    return report
