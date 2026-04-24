"""End-to-end @-mention injection: real BaseAgent + mocked LLM verifies
that the synthesized assistant(tool_calls) + tool turn lands in memory in
canonical order before the first LLM call."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_app.tools.filesystem.list_dir import list_dir
from agent_app.tools.filesystem.read_file import read_file
from agent_cli.repl.mentions import expand_mentions
from agent_harness.agent.react import ReActAgent
from agent_harness.context.context import AgentContext
from agent_harness.core.message import Message, Role


def _adapter() -> Any:
    a = MagicMock()
    a.render_attachments = AsyncMock()
    return a


@pytest.mark.asyncio
async def test_user_at_file_lands_user_assistant_tool_in_order(
    mock_llm, config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM sees [user, assistant(tool_calls), tool] before first call."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "foo.py").write_text("# hello\nprint('hi')\n")

    mock_llm.add_text_response("I read foo.py and it greets.")

    ctx = AgentContext.create(config)
    agent = ReActAgent(
        name="cli_agent", llm=mock_llm, tools=[read_file, list_dir], context=ctx,
    )
    adapter = _adapter()

    async def cb(a, msg, t):
        if msg.role != Role.USER:
            return
        await expand_mentions(a, adapter, t)

    result = await agent.run(
        "explain @foo.py", after_input_appended=cb,
    )

    assert result.output == "I read foo.py and it greets."

    first_call = mock_llm.call_history[0]
    roles = [m.role for m in first_call]
    assert Role.USER in roles
    assert Role.ASSISTANT in roles
    assert Role.TOOL in roles

    user_idx = roles.index(Role.USER)
    asst_idx = roles.index(Role.ASSISTANT)
    tool_idx = roles.index(Role.TOOL)
    assert user_idx < asst_idx < tool_idx

    assert any(
        "hello" in (m.content or "")
        for m in first_call if m.role == Role.TOOL
    )


@pytest.mark.asyncio
async def test_multiple_mentions_inject_one_assistant_with_n_tool_calls(
    mock_llm, config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "b.py").write_text("b")

    mock_llm.add_text_response("done")

    ctx = AgentContext.create(config)
    agent = ReActAgent(
        name="cli_agent", llm=mock_llm, tools=[read_file, list_dir], context=ctx,
    )
    adapter = _adapter()

    async def cb(a, msg, t):
        if msg.role != Role.USER:
            return
        await expand_mentions(a, adapter, t)

    await agent.run("look at @a.py and @src", after_input_appended=cb)

    first_call = mock_llm.call_history[0]
    asst_msgs = [m for m in first_call if m.role == Role.ASSISTANT]
    tool_msgs = [m for m in first_call if m.role == Role.TOOL]

    assert len(asst_msgs) == 1
    assert asst_msgs[0].tool_calls is not None
    assert len(asst_msgs[0].tool_calls) == 2
    assert len(tool_msgs) == 2


@pytest.mark.asyncio
async def test_no_mentions_no_synthetic_turn(
    mock_llm, config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    mock_llm.add_text_response("hi")

    ctx = AgentContext.create(config)
    agent = ReActAgent(
        name="cli_agent", llm=mock_llm, tools=[read_file, list_dir], context=ctx,
    )
    adapter = _adapter()

    async def cb(a, msg, t):
        if msg.role != Role.USER:
            return
        await expand_mentions(a, adapter, t)

    await agent.run("just text", after_input_appended=cb)

    first_call = mock_llm.call_history[0]
    asst_msgs = [m for m in first_call if m.role == Role.ASSISTANT]

    assert asst_msgs == []
    adapter.render_attachments.assert_not_called()


@pytest.mark.asyncio
async def test_nonexistent_at_silently_skipped(
    mock_llm, config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    mock_llm.add_text_response("ok")

    ctx = AgentContext.create(config)
    agent = ReActAgent(
        name="cli_agent", llm=mock_llm, tools=[read_file, list_dir], context=ctx,
    )
    adapter = _adapter()

    async def cb(a, msg, t):
        if msg.role != Role.USER:
            return
        await expand_mentions(a, adapter, t)

    await agent.run("@nonexistent.py please", after_input_appended=cb)

    first_call = mock_llm.call_history[0]
    asst_msgs = [m for m in first_call if m.role == Role.ASSISTANT]

    assert asst_msgs == []
    user_msg = next(m for m in first_call if m.role == Role.USER)
    assert "@nonexistent.py" in (user_msg.content or "")
