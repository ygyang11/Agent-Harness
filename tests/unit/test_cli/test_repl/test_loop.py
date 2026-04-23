"""REPL loop helpers + race + shutdown."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from agent_cli.repl.loop import (
    _cancel_and_collect,
    _cancel_loser,
    _handle_line,
    _InputOutcome,
    _run,
)

# ---- helpers --------------------------------------------------------------


def _done_future(result_spec: object) -> asyncio.Future[str]:
    """Build an already-done Future resolving to ``result_spec`` — a return
    value or an exception class/instance. Future duck-types as the awaitable
    ``_cancel_and_collect`` needs (``done``/``cancel``/``await``)."""
    fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    if isinstance(result_spec, type) and issubclass(result_spec, BaseException):
        fut.set_exception(result_spec())
    elif isinstance(result_spec, BaseException):
        fut.set_exception(result_spec)
    else:
        assert isinstance(result_spec, str)
        fut.set_result(result_spec)
    return fut


async def test_cancel_and_collect_returns_line_when_done() -> None:
    task = _done_future("hello")
    outcome = await _cancel_and_collect(task, MagicMock())
    assert outcome == _InputOutcome(line="hello")


async def test_cancel_and_collect_classifies_eof_when_done() -> None:
    task = _done_future(EOFError)
    outcome = await _cancel_and_collect(task, MagicMock())
    assert outcome.eof is True
    assert outcome.line is None


async def test_cancel_and_collect_classifies_keyboard_interrupt_when_done() -> None:
    task = _done_future(KeyboardInterrupt)
    outcome = await _cancel_and_collect(task, MagicMock())
    assert outcome.interrupted is True


async def test_cancel_and_collect_classifies_cancelled_when_done() -> None:
    task = _done_future(asyncio.CancelledError)
    outcome = await _cancel_and_collect(task, MagicMock())
    assert outcome.cancelled is True
    assert outcome.buffered == ""


async def test_cancel_and_collect_captures_buffer_on_cancel() -> None:
    pt_session = MagicMock()
    pt_session.default_buffer.text = "partial typing"

    hold = asyncio.Event()

    async def _wait_forever() -> str:
        await hold.wait()
        return ""

    task: asyncio.Task[str] = asyncio.create_task(_wait_forever())
    await asyncio.sleep(0)

    outcome = await _cancel_and_collect(task, pt_session)
    assert outcome.cancelled is True
    assert outcome.buffered == "partial typing"
    assert outcome.line is None


async def test_cancel_and_collect_returns_real_line_if_race_completed() -> None:
    async def _instant() -> str:
        return "real input"

    task: asyncio.Task[str] = asyncio.create_task(_instant())
    await task
    outcome = await _cancel_and_collect(task, MagicMock())
    assert outcome == _InputOutcome(line="real input")


async def test_cancel_loser_handles_none() -> None:
    await _cancel_loser(None)


async def test_cancel_loser_skips_done_task() -> None:
    async def _quick() -> str:
        return "x"

    done_task: asyncio.Task[str] = asyncio.create_task(_quick())
    await done_task
    await _cancel_loser(done_task)
    assert done_task.done()
    assert done_task.result() == "x"


async def test_cancel_loser_cancels_pending_task_and_awaits() -> None:
    async def _forever() -> str:
        await asyncio.sleep(3600)
        return ""

    task: asyncio.Task[str] = asyncio.create_task(_forever())
    await asyncio.sleep(0)
    await _cancel_loser(task)
    assert task.cancelled()


# ---- race (via _run) ------------------------------------------------------


async def test_run_cancelled_error_resets_state_and_ends_step() -> None:
    agent = MagicMock()
    agent.context.state.reset = MagicMock()
    agent.run = AsyncMock(side_effect=asyncio.CancelledError())
    adapter = MagicMock()
    adapter.end_step = AsyncMock()
    console = MagicMock()

    await _run(agent, "text", "session-1", console, adapter)

    agent.context.state.reset.assert_called_once()
    adapter.end_step.assert_awaited_once()


async def test_run_swallows_generic_exception_to_keep_repl_alive() -> None:
    agent = MagicMock()
    agent.run = AsyncMock(side_effect=RuntimeError("boom"))
    adapter = MagicMock()
    adapter.end_step = AsyncMock()
    console = MagicMock()

    await _run(agent, "text", "session-1", console, adapter)

    adapter.end_step.assert_awaited_once()
    # hooks.on_error owns error rendering now — _run must not print.
    console.print.assert_not_called()


async def test_run_ends_step_on_success() -> None:
    agent = MagicMock()
    agent.run = AsyncMock(return_value=None)
    adapter = MagicMock()
    adapter.end_step = AsyncMock()

    await _run(agent, "text", "session-1", MagicMock(), adapter)

    agent.run.assert_awaited_once()
    adapter.end_step.assert_awaited_once()


# ---- _handle_line ---------------------------------------------------------


async def test_handle_line_cancelled_keeps_repl_alive() -> None:
    registry = MagicMock()
    registry.dispatch = AsyncMock(side_effect=asyncio.CancelledError())
    console = MagicMock()

    should_exit = await _handle_line(
        "/slow", MagicMock(), console, registry, "sid", AsyncMock(), MagicMock(),
        MagicMock(),
    )

    assert should_exit is False
    assert any(
        c.args and "cancelled" in str(c.args[0]).lower()
        for c in console.print.call_args_list
    )


