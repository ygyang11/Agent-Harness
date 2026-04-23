"""SIGINT handler with per-turn task binding.

``bind_work(task)`` captures ``task`` in the handler closure — stale
signals from prior turns hit their own (done) task, not the current one.

Any ``prompt_async(handle_sigint=True)`` caller MUST ``restore_current()``
on exit (prompt_toolkit removes our handler without restoring).
Assumes serial foreground work (``_active`` is a single slot).
"""
from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

_active: Callable[[], None] | None = None


def _noop() -> None:
    pass


def _install(handler: Callable[[], None]) -> bool:
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, handler)
        return True
    except NotImplementedError:
        return False


def install_idle() -> bool:
    """Install no-op handler. Call at REPL startup; absorbs SIGINT between turns."""
    global _active
    _active = _noop
    return _install(_noop)


def uninstall() -> bool:
    """Remove handler entirely. Call at shutdown."""
    global _active
    _active = None
    try:
        asyncio.get_running_loop().remove_signal_handler(signal.SIGINT)
        return True
    except NotImplementedError:
        return False


def restore_current() -> bool:
    """Reinstall the active handler (task-bound or idle). Use after borrowing SIGINT."""
    handler = _active or _noop
    return _install(handler)


@contextmanager
def bind_work(task: asyncio.Task[Any]) -> Iterator[None]:
    """Bind SIGINT to cancel ``task`` for the body's lifetime."""
    global _active

    def _cancel_bound() -> None:
        if not task.done():
            task.cancel()

    prev = _active
    _active = _cancel_bound
    _install(_cancel_bound)
    try:
        yield
    finally:
        _active = prev
        _install(prev if prev is not None else _noop)
