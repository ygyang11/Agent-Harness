"""Input multiplexer — single stdin reader with priority queue.

Prevents concurrent input() calls that cause readline lock contention.
Higher priority requests are served first when the current input() completes.
Uses daemon threads so blocked input() calls don't prevent process exit.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(order=True)
class _InputRequest:
    priority: int  # stored negative: higher priority = lower sort value
    prompt: str = field(compare=False)
    future: asyncio.Future[str] = field(compare=False)


async def _async_input(prompt: str) -> str:
    """Read input in a daemon thread — won't block process exit."""
    loop = asyncio.get_running_loop()
    result_future: asyncio.Future[str] = loop.create_future()

    def _reader() -> None:
        try:
            line = input(prompt)
            loop.call_soon_threadsafe(result_future.set_result, line)
        except BaseException as e:
            loop.call_soon_threadsafe(result_future.set_exception, e)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return await result_future


class InputMultiplexer:
    """Serialize all stdin reads through a single input() call."""

    def __init__(self) -> None:
        self._queue: list[_InputRequest] = []
        self._processing = False

    async def read(self, prompt: str = "", priority: int = 0) -> str:
        """Request a line of input. Higher priority = served first."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        req = _InputRequest(priority=-priority, prompt=prompt, future=future)
        self._queue.append(req)
        self._queue.sort()

        if self._processing and priority > 0:
            print(
                f"\n[!] Approval pending: {prompt.strip()}\n"
                f"[>] Press Enter (Ctrl+U to clear current input), "
                f"then respond to approval."
            )

        if not self._processing:
            self._processing = True
            asyncio.ensure_future(self._process_queue())

        return await future

    async def _process_queue(self) -> None:
        try:
            while self._queue:
                req = self._queue.pop(0)
                if req.future.cancelled():
                    continue
                try:
                    line = await _async_input(req.prompt)
                    if not req.future.done():
                        req.future.set_result(line)
                except BaseException as e:
                    if not req.future.done():
                        req.future.set_exception(e)
        finally:
            self._processing = False


_mux = InputMultiplexer()


async def mux_input(prompt: str = "", priority: int = 0) -> str:
    """Drop-in async replacement for input(). Priority 0=chat, 10=approval."""
    return await _mux.read(prompt, priority)
