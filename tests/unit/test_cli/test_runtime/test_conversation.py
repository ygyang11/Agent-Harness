"""Tests for runtime/conversation.py — append_tool_turn."""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.runtime.conversation import append_tool_turn
from agent_harness.core.message import Message, Role, ToolCall, ToolResult


def _agent_with_memory() -> tuple[Any, list[Message]]:
    captured: list[Message] = []
    memory = MagicMock()

    async def add_message(msg: Message) -> None:
        captured.append(msg)

    memory.add_message = add_message
    agent = MagicMock()
    agent.context.short_term_memory = memory
    return agent, captured


def _pair(idx: int, *, is_error: bool = False) -> tuple[ToolCall, ToolResult]:
    tc = ToolCall(id=f"id_{idx}", name="read_file", arguments={"file_path": f"f{idx}"})
    tr = ToolResult(tool_call_id=tc.id, content=f"content_{idx}", is_error=is_error)
    return tc, tr


@pytest.mark.asyncio
async def test_empty_pairs_writes_nothing() -> None:
    agent, captured = _agent_with_memory()
    render = AsyncMock()

    await append_tool_turn(agent, [], render=render)

    assert captured == []
    render.assert_not_called()


@pytest.mark.asyncio
async def test_single_pair_no_render_writes_assistant_then_tool() -> None:
    agent, captured = _agent_with_memory()
    pair = _pair(1)

    await append_tool_turn(agent, [pair])

    assert len(captured) == 2
    assert captured[0].role == Role.ASSISTANT
    assert captured[0].tool_calls == [pair[0]]
    assert captured[1].role == Role.TOOL
    assert captured[1].tool_result is not None
    assert captured[1].tool_result.tool_call_id == "id_1"


@pytest.mark.asyncio
async def test_render_runs_before_memory_write() -> None:
    agent, captured = _agent_with_memory()
    pair = _pair(1)
    timeline: list[str] = []

    async def render(items: list[tuple[ToolCall, ToolResult]]) -> None:
        timeline.append("render")

    orig_add = agent.context.short_term_memory.add_message

    async def add_message(msg: Message) -> None:
        timeline.append("write")
        await orig_add(msg)

    agent.context.short_term_memory.add_message = add_message

    await append_tool_turn(agent, [pair], render=render)

    assert timeline == ["render", "write", "write"]


@pytest.mark.asyncio
async def test_render_exception_skips_memory_write() -> None:
    agent, captured = _agent_with_memory()
    pair = _pair(1)

    async def render(items: list[tuple[ToolCall, ToolResult]]) -> None:
        raise RuntimeError("render fail")

    with pytest.raises(RuntimeError, match="render fail"):
        await append_tool_turn(agent, [pair], render=render)

    assert captured == []


@pytest.mark.asyncio
async def test_multi_pair_assistant_carries_all_tcs() -> None:
    agent, captured = _agent_with_memory()
    pairs = [_pair(1), _pair(2), _pair(3)]

    await append_tool_turn(agent, pairs)

    assert captured[0].role == Role.ASSISTANT
    assert captured[0].tool_calls is not None
    assert [tc.id for tc in captured[0].tool_calls] == ["id_1", "id_2", "id_3"]
    assert [m.tool_result.tool_call_id for m in captured[1:]] == [
        "id_1", "id_2", "id_3",
    ]


@pytest.mark.asyncio
async def test_is_error_pass_through() -> None:
    agent, captured = _agent_with_memory()
    pair = _pair(1, is_error=True)

    await append_tool_turn(agent, [pair])

    assert captured[1].tool_result is not None
    assert captured[1].tool_result.is_error is True


@pytest.mark.asyncio
async def test_shield_completes_writes_when_outer_cancelled() -> None:
    agent, captured = _agent_with_memory()
    pairs = [_pair(1), _pair(2)]

    async def runner() -> None:
        await append_tool_turn(agent, pairs)

    task = asyncio.create_task(runner())
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(captured) == 3, (
        f"shield should let memory writes finish; got {len(captured)} entries"
    )
