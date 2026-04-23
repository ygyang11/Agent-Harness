"""Single-line live-updating terminal heartbeat with debounce.

Generic mechanism — callers supply a ``writer(frame, elapsed)`` callback that
produces the line body. Output is framed with ``CLEAR_LINE`` so each tick
overwrites in place. Writes go straight to ``console.file`` (bypassing Rich)
so the heartbeat can coexist with an active Rich Live region without one
clobbering the other's cursor bookkeeping.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from rich.console import Console

CLEAR_LINE = "\r\x1b[2K"


class LiveLine:
    def __init__(
        self,
        console: Console,
        lock: asyncio.Lock,
        writer: Callable[[int, int], str],
        debounce_s: float,
        tick_s: float,
    ) -> None:
        self._console = console
        self._lock = lock
        self._writer = writer
        self._debounce_s = debounce_s
        self._tick_s = tick_s
        self._task: asyncio.Task[None] | None = None
        self._visible: bool = False

    async def start(self) -> None:
        # Restart cleanly: cancel prior task + clear stale visible line.
        await self.stop()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        await self._cancel()
        async with self._lock:
            self._clear()

    async def stop_no_lock(self) -> None:
        await self._cancel()
        self._clear()

    def clear_no_lock(self) -> None:
        self._clear()

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    @property
    def visible(self) -> bool:
        return self._visible

    async def _cancel(self) -> None:
        t, self._task = self._task, None
        if t is not None and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    def _clear(self) -> None:
        if self._visible:
            f = self._console.file
            f.write(CLEAR_LINE)
            f.flush()
            self._visible = False

    async def _loop(self) -> None:
        try:
            await asyncio.sleep(self._debounce_s)
            start = time.monotonic()
            frame = 0
            while True:
                elapsed = int(time.monotonic() - start)
                async with self._lock:
                    f = self._console.file
                    f.write(CLEAR_LINE + self._writer(frame, elapsed))
                    f.flush()
                    self._visible = True
                frame += 1
                await asyncio.sleep(self._tick_s)
        except asyncio.CancelledError:
            pass
