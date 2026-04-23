"""SIGINT coordination — install/uninstall/bind_work/restore_current."""
from __future__ import annotations

import asyncio
import signal

import pytest

from agent_cli.runtime import sigint


def _current_handler() -> object:
    loop = asyncio.get_running_loop()
    handle = loop._signal_handlers.get(signal.SIGINT)  # type: ignore[attr-defined]
    return handle._callback if handle is not None else None


async def test_install_idle_registers_noop() -> None:
    sigint.install_idle()
    try:
        assert _current_handler() is sigint._noop
    finally:
        sigint.uninstall()


async def test_uninstall_removes_handler() -> None:
    sigint.install_idle()
    sigint.uninstall()
    assert _current_handler() is None


async def test_bind_work_installs_task_bound_and_restores() -> None:
    sigint.install_idle()
    try:
        async def _forever() -> None:
            await asyncio.sleep(3600)

        task = asyncio.create_task(_forever())
        try:
            await asyncio.sleep(0)
            idle_handler = _current_handler()
            with sigint.bind_work(task):
                work_handler = _current_handler()
                assert work_handler is not idle_handler
            # restored after context exit
            assert _current_handler() is idle_handler
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        sigint.uninstall()


async def test_bind_work_cancels_bound_task_on_signal() -> None:
    sigint.install_idle()
    try:
        async def _forever() -> None:
            await asyncio.sleep(3600)

        task = asyncio.create_task(_forever())
        await asyncio.sleep(0)
        with sigint.bind_work(task):
            _current_handler()()  # simulate SIGINT dispatch
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
    finally:
        sigint.uninstall()


async def test_stale_handler_noop_on_done_task() -> None:
    """A stale SIGINT from a previous turn calls cancel() on a done task."""
    sigint.install_idle()
    try:
        async def _quick() -> str:
            return "ok"

        old_task = asyncio.create_task(_quick())
        # Capture the bound handler WHILE old_task is live
        with sigint.bind_work(old_task):
            stale_handler = _current_handler()
        await old_task  # old_task completes
        assert old_task.done()
        # Stale handler fires now (simulated late delivery) — must not raise
        stale_handler()
    finally:
        sigint.uninstall()


async def test_restore_current_reinstalls_active_after_borrow() -> None:
    """Simulates prompt_toolkit borrowing SIGINT and removing it on exit."""
    sigint.install_idle()
    try:
        async def _forever() -> None:
            await asyncio.sleep(3600)

        task = asyncio.create_task(_forever())
        try:
            await asyncio.sleep(0)
            with sigint.bind_work(task):
                bound = _current_handler()
                # Third party removes our handler (simulates prompt_toolkit exit)
                asyncio.get_running_loop().remove_signal_handler(signal.SIGINT)
                assert _current_handler() is None
                # Caller restores
                sigint.restore_current()
                assert _current_handler() is bound
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        sigint.uninstall()


async def test_restore_current_outside_bind_work_restores_idle() -> None:
    sigint.install_idle()
    try:
        asyncio.get_running_loop().remove_signal_handler(signal.SIGINT)
        assert _current_handler() is None
        sigint.restore_current()
        assert _current_handler() is sigint._noop
    finally:
        sigint.uninstall()
