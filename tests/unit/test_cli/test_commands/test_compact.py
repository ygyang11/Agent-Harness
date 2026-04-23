import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_cli.commands.builtin.compact import CMD

from .conftest import render_output


async def test_compact_passes_extra_instructions_and_saves() -> None:
    compressor = MagicMock()
    compressor.compress = AsyncMock(return_value=["msg1"])
    compressor.take_last_result = MagicMock(return_value=MagicMock(
        original_count=10, compressed_count=3, summary_tokens=500,
    ))
    agent = MagicMock()
    agent.context.short_term_memory.compressor = compressor
    agent.context.short_term_memory._messages = ["m1", "m2"]
    save = AsyncMock()
    ctx = MagicMock(agent=agent, save_session=save)

    result = await CMD.handler(ctx, "focus on auth")
    compressor.compress.assert_awaited_once()
    assert compressor.compress.await_args.kwargs["extra_instructions"] == "focus on auth"
    save.assert_awaited_once()
    assert "Compacted" in render_output(result.output)


async def test_compact_without_compressor_returns_message() -> None:
    agent = MagicMock()
    agent.context.short_term_memory.compressor = None
    ctx = MagicMock(agent=agent)
    result = await CMD.handler(ctx, "")
    assert "not enabled" in render_output(result.output).lower()


async def test_compact_save_cancel_propagates_without_repl_concerns() -> None:
    """save_session cancel inside /compact bubbles as CancelledError;
    outer _handle_line (tested separately) is what keeps REPL alive."""
    compressor = MagicMock()
    compressor.compress = AsyncMock(return_value=["compressed"])
    agent = MagicMock()
    agent.context.short_term_memory.compressor = compressor
    agent.context.short_term_memory._messages = ["m1", "m2"]
    save = AsyncMock(side_effect=asyncio.CancelledError())
    ctx = MagicMock(agent=agent, save_session=save)

    with pytest.raises(asyncio.CancelledError):
        await CMD.handler(ctx, "")
    compressor.compress.assert_awaited_once()
