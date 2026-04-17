"""Minimal REPL: read a line, dispatch to ``agent.run``, render errors."""
from __future__ import annotations

import asyncio

from prompt_toolkit import PromptSession
from rich.console import Console

from agent_harness.agent.base import BaseAgent

_EXIT_WORDS = frozenset({"exit", "quit", "/exit", "/quit", "/q"})


async def run_repl(
    agent: BaseAgent,
    console: Console,
    session_id: str,
    *,
    prompt: str = "❯ ",
) -> None:
    pt_session: PromptSession[str] = PromptSession()

    while True:
        try:
            line = (await pt_session.prompt_async(prompt)).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in _EXIT_WORDS:
            break

        try:
            await agent.run(line, session=session_id)
        except asyncio.CancelledError:
            # CancelledError bypasses base.run()'s `except Exception`, so agent
            # state can be stuck in THINKING/ACTING; the next run()'s
            # transition(THINKING) would raise StateTransitionError. Reset to IDLE.
            agent.context.state.reset()
            console.print("[dim]cancelled[/dim]")
        except Exception:
            # hooks.on_error already rendered the message; base.run() set state
            # to ERROR and the next run() will reset it. Swallow to keep REPL alive.
            pass
