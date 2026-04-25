"""REPL loop helpers + race + shutdown."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.repl.loop import (
    _cancel_and_collect,
    _cancel_loser,
    _handle_line,
    _InputOutcome,
    _run,
    run_repl,
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
    from agent_cli.runtime.shell import ShellState

    registry = MagicMock()
    registry.dispatch = AsyncMock(side_effect=asyncio.CancelledError())
    console = MagicMock()
    pt_session = MagicMock()
    pt_session.completer = MagicMock()

    should_exit = await _handle_line(
        "/slow",
        MagicMock(),
        console,
        registry,
        "sid",
        AsyncMock(),
        MagicMock(),
        MagicMock(),
        ShellState(),
        pt_session,
    )

    assert should_exit is False
    assert any(
        c.args and "cancelled" in str(c.args[0]).lower() for c in console.print.call_args_list
    )


async def test_handle_line_shell_lane_dispatches_to_exec_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_cli.runtime.shell import ShellState

    registry = MagicMock()
    registry.dispatch = AsyncMock(return_value=None)
    console = MagicMock()
    pt_session = MagicMock()
    pt_session.completer = MagicMock()
    agent = MagicMock()

    captured: list[tuple[object, ...]] = []

    async def fake_exec_shell(state, command, ag, comp, ad):
        captured.append((command, ag, comp))

    monkeypatch.setattr(
        "agent_cli.runtime.shell.exec_shell",
        fake_exec_shell,
    )

    should_exit = await _handle_line(
        "!ls -la",
        agent,
        console,
        registry,
        "sid",
        AsyncMock(),
        MagicMock(),
        MagicMock(),
        ShellState(),
        pt_session,
    )

    assert should_exit is False
    assert len(captured) == 1
    assert captured[0][0] == "ls -la"
    registry.dispatch.assert_not_called()


async def test_handle_line_bare_bang_is_noop() -> None:
    from agent_cli.runtime.shell import ShellState

    registry = MagicMock()
    registry.dispatch = AsyncMock(return_value=None)
    console = MagicMock()
    pt_session = MagicMock()
    pt_session.completer = MagicMock()

    should_exit = await _handle_line(
        "!",
        MagicMock(),
        console,
        registry,
        "sid",
        AsyncMock(),
        MagicMock(),
        MagicMock(),
        ShellState(),
        pt_session,
    )

    assert should_exit is False
    registry.dispatch.assert_not_called()


async def test_handle_line_shell_cancellation_renders_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_cli.runtime.shell import ShellState

    console = MagicMock()
    pt_session = MagicMock()
    pt_session.completer = MagicMock()

    async def cancelling_exec(state, command, ag, comp, ad):
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        "agent_cli.runtime.shell.exec_shell",
        cancelling_exec,
    )

    await _handle_line(
        "!sleep 30",
        MagicMock(),
        console,
        MagicMock(),
        "sid",
        AsyncMock(),
        MagicMock(),
        MagicMock(),
        ShellState(),
        pt_session,
    )

    assert any(
        c.args and "cancelled" in str(c.args[0]).lower() for c in console.print.call_args_list
    )


# ---- run_repl: paste expand integration ----------------------------------


def _make_run_repl_mocks(prompt_results: list[object]):
    """Build the dependency mocks; pt_session.prompt_async runs through
    ``prompt_results`` (str values returned, exception classes/instances raised).
    """
    pt_session = MagicMock()
    pt_session.completer = None
    pt_session.key_bindings = None

    side_effects: list[object] = []
    for r in prompt_results:
        if isinstance(r, type) and issubclass(r, BaseException):
            side_effects.append(r())
        else:
            side_effects.append(r)
    pt_session.prompt_async = AsyncMock(side_effect=side_effects)

    @asynccontextmanager
    async def _lock():
        yield

    adapter = MagicMock()
    adapter.lock = _lock
    adapter.print_inline = AsyncMock()
    adapter.end_step = AsyncMock()
    adapter.begin_run = MagicMock()

    handler = MagicMock()
    empty_q: asyncio.Queue[object] = asyncio.Queue()
    handler.pending_queue = MagicMock(return_value=empty_q)
    handler.cancel_pending = MagicMock()

    agent = MagicMock()
    agent.run = AsyncMock(return_value=None)
    agent.llm.model_name = "test-model"
    agent.context.short_term_memory.token_count = 0
    agent.context.short_term_memory.max_tokens = 1000

    registry = MagicMock()
    registry.dispatch = AsyncMock(return_value=None)

    backend = MagicMock()
    console = MagicMock()

    return {
        "agent": agent,
        "console": console,
        "registry": registry,
        "backend": backend,
        "adapter": adapter,
        "handler": handler,
        "pt_session": pt_session,
    }


async def _drive_run_repl(mocks: dict[str, object]) -> None:
    from agent_cli.runtime.shell import ShellState

    await run_repl(
        mocks["agent"],
        mocks["console"],
        "sid",
        mocks["registry"],
        mocks["backend"],
        mocks["adapter"],
        mocks["handler"],
        mocks["pt_session"],
        ShellState(),
    )


async def test_run_repl_expired_placeholder_prints_notice_and_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _make_run_repl_mocks(["[Pasted text #99]", EOFError])
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.has_running",
        lambda agent: False,
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.collect_results",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.shutdown",
        AsyncMock(),
    )

    captured: list[str] = []

    async def fake_handle_line(line, *args, **kwargs):
        captured.append(line)
        return False

    monkeypatch.setattr("agent_cli.repl.loop._handle_line", fake_handle_line)

    await _drive_run_repl(mocks)

    assert captured == ["[Pasted text unavailable]"]
    mocks["adapter"].print_inline.assert_awaited_once()


async def test_run_repl_whitespace_only_expansion_skips_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A short whitespace-only line never triggers the placeholder path; this
    # test pins that empty/whitespace input goes through the strip() guard
    # without invoking _handle_line, regardless of paste store state.
    mocks = _make_run_repl_mocks(["   \n  ", EOFError])
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.has_running",
        lambda agent: False,
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.collect_results",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.shutdown",
        AsyncMock(),
    )

    called = MagicMock()

    async def fake_handle_line(*args, **kwargs):
        called()
        return False

    monkeypatch.setattr("agent_cli.repl.loop._handle_line", fake_handle_line)

    await _drive_run_repl(mocks)

    called.assert_not_called()
    mocks["adapter"].print_inline.assert_not_called()


async def test_run_repl_plain_input_no_notice_no_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _make_run_repl_mocks(["hello world", EOFError])
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.has_running",
        lambda agent: False,
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.collect_results",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.shutdown",
        AsyncMock(),
    )

    captured: list[str] = []

    async def fake_handle_line(line, *args, **kwargs):
        captured.append(line)
        return False

    monkeypatch.setattr("agent_cli.repl.loop._handle_line", fake_handle_line)

    await _drive_run_repl(mocks)

    assert captured == ["hello world"]
    mocks["adapter"].print_inline.assert_not_called()


async def test_run_repl_passes_paste_processor_to_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_cli.repl.paste import PastePlaceholderProcessor

    mocks = _make_run_repl_mocks([EOFError])
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.has_running",
        lambda agent: False,
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.collect_results",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "agent_cli.repl.loop.background.shutdown",
        AsyncMock(),
    )

    await _drive_run_repl(mocks)

    pt_session = mocks["pt_session"]
    call_kwargs = pt_session.prompt_async.call_args.kwargs
    procs = call_kwargs.get("input_processors")
    assert procs is not None
    assert any(isinstance(p, PastePlaceholderProcessor) for p in procs)


async def test_prompt_with_lock_restores_input_processors() -> None:
    """Regression: prompt_toolkit persists input_processors back onto the
    shared session. _prompt_with_lock must snapshot/restore so subsequent
    consumers (notably approval prompts) don't inherit the REPL highlighter.
    """
    from contextlib import asynccontextmanager

    from agent_cli.repl.loop import _prompt_with_lock
    from agent_cli.repl.paste import PastePlaceholderProcessor

    sentinel = object()
    pt_session = MagicMock()
    pt_session.input_processors = sentinel
    pt_session.prompt_async = AsyncMock(return_value="x")

    @asynccontextmanager
    async def _lock():
        yield

    adapter = MagicMock()
    adapter.lock = _lock

    await _prompt_with_lock(
        pt_session,
        adapter,
        default="",
        bottom_toolbar=lambda: None,
        input_processors=[PastePlaceholderProcessor()],
    )

    assert pt_session.input_processors is sentinel
