"""REPL main loop with 4-way race: bg approval / bg result / user input / bg wake."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console

from agent_cli.adapter import CliAdapter
from agent_cli.approval_handler import CliApprovalHandler, _PendingApproval
from agent_cli.commands.base import CommandContext
from agent_cli.commands.registry import CommandRegistry
from agent_cli.render.ui import make_status_bar_text
from agent_cli.repl.completer import build_command_completer
from agent_cli.repl.keybindings import build_keybindings, reset_ctrl_c_state
from agent_cli.runtime import background
from agent_cli.runtime.session import make_save_session
from agent_cli.runtime.sigint import bind_work
from agent_cli.theme import APPROVAL, COMPRESSION, PROMPT
from agent_harness.agent.base import BaseAgent
from agent_harness.core.message import Message
from agent_harness.session.base import BaseSession

_DEFAULT_PROMPT = f"{PROMPT} "
_ToolbarText = Callable[[], HTML]


@dataclass
class _InputOutcome:
    line: str | None = None
    eof: bool = False
    interrupted: bool = False
    cancelled: bool = False
    buffered: str = ""


@dataclass
class _LoopState:
    pending_input: str = ""
    pending_from_race: _PendingApproval | None = None
    deferred_line: str | None = None


async def _cancel_and_collect(
    task: asyncio.Task[str],
    pt_session: PromptSession[str],
) -> _InputOutcome:
    # Unified: done tasks resolve immediately via `await`; pending tasks get
    # their buffer snapshotted then cancelled, both paths share one except.
    if task.done():
        buffered = ""
    else:
        buffered = pt_session.default_buffer.text
        task.cancel()
    try:
        line = await task
        return _InputOutcome(line=line)
    except EOFError:
        return _InputOutcome(eof=True)
    except KeyboardInterrupt:
        return _InputOutcome(interrupted=True)
    except asyncio.CancelledError:
        return _InputOutcome(cancelled=True, buffered=buffered)


async def _cancel_loser(task: asyncio.Task[Any] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _prompt_with_lock(
    pt_session: PromptSession[str],
    adapter: CliAdapter,
    default: str,
    bottom_toolbar: _ToolbarText,
) -> str:
    async with adapter.lock():
        try:
            return await pt_session.prompt_async(
                _DEFAULT_PROMPT,
                default=default,
                set_exception_handler=False,
                bottom_toolbar=bottom_toolbar,
                # prompt_toolkit's handle_sigint=True installs its own
                # SIGINT handler via add_signal_handler and removes it
                # on exit without restoring ours.
                handle_sigint=False,
            )
        finally:
            reset_ctrl_c_state()

 
async def run_repl(
    agent: BaseAgent,
    console: Console,
    session_id: str,
    registry: CommandRegistry,
    session_backend: BaseSession,
    adapter: CliAdapter,
    handler: CliApprovalHandler,
    pt_session: PromptSession[str],
) -> None:
    pt_session.completer = build_command_completer(registry)
    pt_session.key_bindings = build_keybindings()
    save = make_save_session(agent, session_id, session_backend)

    state = _LoopState()

    try:
        while True:
            # Priority ladder (1→4). Step 4's race doesn't dispatch inline — winners
            # snapshot into _LoopState and are drained by N+1's 1-3 in fixed order,
            # so race-arrival order never leaks into handling order.

            # 1 bg approval drain — bg task is blocked on a future, highest priority
            pending = state.pending_from_race
            state.pending_from_race = None
            if pending is None:
                try:
                    pending = handler.pending_queue().get_nowait()
                except asyncio.QueueEmpty:
                    pass
            if pending is not None:
                if pending.future.done():
                    # Orphan from a cancelled bg task; skip without prompting.
                    continue
                async with adapter.lock():
                    console.print(
                        f"\n[accent]{APPROVAL} Background approval requested[/accent]",
                    )
                try:
                    await handler.resolve_pending(pending)
                except Exception as e:
                    # Future already carries the exception for the bg task;
                    # surface to user and keep REPL alive
                    console.print(f"[error]Approval handler failed: {e}[/error]")
                continue

            # 2 bg result collect — auto-trigger agent.run on completion
            if await background.collect_results(agent):
                async with adapter.lock():
                    console.print(
                        f"\n[info]{COMPRESSION} Background task completed[/info]",
                    )
                await _run(
                    agent,
                    Message.system(
                        "[Background Task Notification] Process the completed "
                        "background task results.",
                        metadata={"is_background_result": True},
                    ),
                    session_id,
                    console,
                    adapter,
                )
                continue

            # 3 deferred input from last race — honour user's Enter before new prompt
            if state.deferred_line is not None:
                raw = state.deferred_line
                state.deferred_line = None
                # strip() only for the empty-check; pass raw to preserve
                # indentation / multiline structure for pasted code blocks.
                if raw.strip():
                    if await _handle_line(
                        raw, agent, console, registry, session_id, save, adapter, handler,
                    ):
                        break
                continue

            # 4 prompt + race against bg events
            input_task: asyncio.Task[str] = asyncio.create_task(
                _prompt_with_lock(
                    pt_session,
                    adapter,
                    state.pending_input,
                    make_status_bar_text(agent),
                ),
            )
            state.pending_input = ""

            wait_set: set[asyncio.Task[Any]] = {input_task}
            bg_wait: asyncio.Task[Any] | None = None
            if background.has_running(agent):
                bg_wait = asyncio.create_task(background.wait_next(agent))
                wait_set.add(bg_wait)
            approval_wait: asyncio.Task[Any] = asyncio.create_task(
                handler.pending_queue().get(),
            )
            wait_set.add(approval_wait)

            await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

            # Post-snapshot completion still counts — `.done()` over `in done`.
            if approval_wait.done():
                state.pending_from_race = approval_wait.result()
            else:
                await _cancel_loser(approval_wait)

            # bg_wait is only a wake signal; drop payload, cancel if still pending.
            if bg_wait is not None:
                if bg_wait.done():
                    bg_wait.result()
                else:
                    await _cancel_loser(bg_wait)

            outcome = await _cancel_and_collect(input_task, pt_session)
            if outcome.eof or outcome.interrupted:
                break
            if outcome.line is not None:
                state.deferred_line = outcome.line
            elif outcome.buffered:
                state.pending_input = outcome.buffered

    finally:
        if background.has_running(agent):
            count = background.cancel_all(agent)
            if count:
                console.print(f"[dim]Cancelled {count} background task(s).[/dim]")
        await background.shutdown(agent)
        handler.cancel_pending()
        await adapter.end_step()


async def _handle_line(
    line: str,
    agent: BaseAgent,
    console: Console,
    registry: CommandRegistry,
    session_id: str,
    save: Callable[[], Awaitable[None]],
    adapter: CliAdapter,
    handler: CliApprovalHandler,
) -> bool:
    console.print()
    ctx = CommandContext(
        agent=agent,
        session_id=session_id,
        registry=registry,
        save_session=save,
        approval_handler=handler,
    )
    task = asyncio.create_task(registry.dispatch(line, ctx))
    try:
        with bind_work(task):
            result = await task
    except asyncio.CancelledError:
        # Keep REPL alive on Ctrl+C during a command
        console.print("[dim]Command cancelled.[/dim]")
        console.print()
        return False
    except Exception as e:
        console.print(f"[error]Command failed: {e}[/error]")
        console.print()
        return False
    if result is not None:
        if result.output is not None:
            console.print(result.output)
            console.print()
        if result.should_exit:
            return True
        if result.agent_input:
            await _run(agent, result.agent_input, session_id, console, adapter)
        return False

    await _run(agent, line, session_id, console, adapter)
    return False


async def _run(
    agent: BaseAgent,
    text: str | Message,
    session_id: str,
    console: Console,
    adapter: CliAdapter,
) -> None:
    cancelled = False
    adapter.begin_run()
    task = asyncio.create_task(agent.run(text, session=session_id))
    try:
        with bind_work(task):
            try:
                await task
            except asyncio.CancelledError:
                # CancelledError bypasses base.run()'s `except Exception`; state
                # may be stuck in non-terminal, next run()'s transition(THINKING)
                # would raise.
                agent.context.state.reset()
                cancelled = True
            except Exception:
                # hooks.on_error already rendered; just keep REPL alive.
                pass
    finally:
        # end_step flushes any pending stream / Live buffers; print cancel
        # banner AFTER so it doesn't get overwritten by late-arriving chunks.
        await adapter.end_step()
    if cancelled:
        console.print(
            "[dim]⎯⎯ Interrupted · what should the agent do differently? ⎯⎯[/dim]"
        )
        console.print()