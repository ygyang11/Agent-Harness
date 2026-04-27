"""Short-term memory manipulation outside the normal agent.run flow."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token
from typing import Any

from agent_harness.agent.base import BaseAgent
from agent_harness.core.message import Message, ToolCall, ToolResult


_pending_writes: ContextVar[list[asyncio.Future[Any]] | None] = ContextVar(
    "_pending_writes", default=None,
)


def use_pending_tracker(tracker: list[asyncio.Future[Any]] | None) -> Token[Any]:
    return _pending_writes.set(tracker)


def reset_pending_tracker(token: Token[Any]) -> None:
    _pending_writes.reset(token)


async def append_tool_turn(
    agent: BaseAgent,
    pairs: list[tuple[ToolCall, ToolResult]],
    *,
    render: Callable[[list[tuple[ToolCall, ToolResult]]], Awaitable[None]] | None = None,
) -> None:
    """Append ``assistant(tool_calls) + tool×N`` in declaration order.

    Render runs first; if it raises, no memory write. Memory write is
    ``asyncio.shield``-ed: once started, completes even if the outer
    task is cancelled (so no orphan tool_calls survive). Cancel before
    the write phase leaves only the user message — also schema-valid.

    The inner write Task is registered into the current pending_writes
    tracker (if bound by the CLI _run via use_pending_tracker) so cancel-
    rollback can drain it before deciding commit state.
    """
    if not pairs:
        return

    if render is not None:
        await render(pairs)

    write_task = asyncio.ensure_future(_write(agent, pairs))
    tracker = _pending_writes.get()
    if tracker is not None:
        tracker.append(write_task)

    await asyncio.shield(write_task)


def refresh_system_prompt(agent: BaseAgent) -> None:
    agent.system_prompt = agent._prompt_builder.build(agent._make_builder_context())


async def _write(
    agent: BaseAgent,
    pairs: list[tuple[ToolCall, ToolResult]],
) -> None:
    tcs = [tc for tc, _ in pairs]
    await agent.context.short_term_memory.add_message(Message.assistant(content="", tool_calls=tcs))
    for _, tr in pairs:
        await agent.context.short_term_memory.add_message(
            Message.tool(
                tool_call_id=tr.tool_call_id,
                content=tr.content,
                is_error=tr.is_error,
            )
        )
