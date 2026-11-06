"""Silence handling (BUILD_PLAN §5.2, edge-case playbook).

One episode at a time. It arms when the agent finishes asking and the candidate may answer.
Timeline of one episode: nudge at `nudge_s` (episode continues) → re-ask at `reask_s` (episode
resets) → give up `giveup_s` after the re-ask. Candidate speech, disconnect, or stage change cancels.
Speech the timer itself triggers must not re-arm it; callers wrap that speech in `own_speech()`.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable

Action = Callable[[], Awaitable[None]]


class SilenceTimer:
    def __init__(self, *, on_nudge: Action, on_reask: Action, on_giveup: Action,
                 nudge_s: float = 6.0, reask_s: float = 15.0, giveup_s: float = 30.0) -> None:
        self._on_nudge, self._on_reask, self._on_giveup = on_nudge, on_reask, on_giveup
        self.nudge_s, self.reask_s, self.giveup_s = nudge_s, reask_s, giveup_s
        self._task: asyncio.Task | None = None
        self._own = 0
        self.generation = 0
        self.events: list[str] = []

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    @contextlib.asynccontextmanager
    async def own_speech(self):
        self._own += 1
        try:
            yield
        finally:
            self._own -= 1

    def arm(self, nudge_s: float | None = None, reask_s: float | None = None) -> None:
        """Agent finished speaking and is listening. Ignored while the timer's own speech plays.
        `nudge_s` / `reask_s` override the defaults for this episode (longer after a story question)."""
        if self._own or self.active:
            return
        self.generation += 1
        self._task = asyncio.create_task(self._run(self.generation, nudge_s or self.nudge_s, reask_s or self.reask_s))

    def cancel(self) -> None:
        """Candidate spoke, disconnected, or the stage changed: end the episode."""
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    stop = cancel

    async def _run(self, gen: int, nudge_s: float, reask_s: float) -> None:
        try:
            await asyncio.sleep(nudge_s)
            self.events.append("nudge")
            async with self.own_speech():
                await self._on_nudge()
            await asyncio.sleep(max(0.0, reask_s - nudge_s))
            self.events.append("reask")
            async with self.own_speech():
                await self._on_reask()
            await asyncio.sleep(self.giveup_s)
            self.events.append("giveup")
            async with self.own_speech():
                await self._on_giveup()
        except asyncio.CancelledError:
            pass
