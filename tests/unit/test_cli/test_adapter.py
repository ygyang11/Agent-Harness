import asyncio
from unittest.mock import MagicMock

import pytest

from agent_cli.adapter import CliAdapter
from agent_cli.render import status_lines as status_lines_mod
from agent_cli.theme import FLEXOKI_DARK


def _adapter() -> CliAdapter:
    a = CliAdapter(MagicMock(), FLEXOKI_DARK)
    a.markdown = MagicMock()
    a.tool_display = MagicMock()
    return a


@pytest.fixture
def fast_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(status_lines_mod, "THINKING_DEBOUNCE_S", 0.0)
    monkeypatch.setattr(status_lines_mod, "THINKING_TICK_S", 0.005)


async def test_stream_delta_enters_markdown_phase() -> None:
    a = _adapter()
    await a.on_stream_delta("hi")
    assert a._phase == "markdown"
    a.markdown.update.assert_called_once_with("hi")


async def test_first_tool_call_transitions_markdown_to_tools() -> None:
    a = _adapter()
    await a.on_stream_delta("let me...")
    tc = MagicMock(id="t1")
    tc.name = "read_file"
    await a.on_tool_call(tc)
    assert a._phase == "tools"
    a.markdown.finalize.assert_called_once()
    a.tool_display.add_call.assert_called_once()


async def test_denied_tool_id_skips_on_tool_result() -> None:
    a = _adapter()
    denied = MagicMock(tool_call_id="t1", tool_name="x", reason="nope")
    await a.on_tool_denied(denied)
    assert "t1" in a._denied_ids
    assert a._phase == "tools"
    a.tool_display.add_denied.assert_called_once()

    result = MagicMock(tool_call_id="t1", is_error=True, content="x")
    await a.on_tool_result(result)
    a.tool_display.mark_result.assert_not_called()


async def test_end_step_closes_tool_live_and_flushes_todo() -> None:
    a = _adapter()
    tc = MagicMock(id="t1")
    tc.name = "read_file"
    await a.on_tool_call(tc)
    a.queue_todo([{"content": "a", "status": "pending"}], {"total": 1, "completed": 0})
    await a.end_step()
    a.tool_display.end.assert_called_once()
    a.tool_display.show_todos.assert_called_once()
    assert a._phase == "none"
    assert a._denied_ids == set()
    assert a._pending_todo is None


async def test_end_step_idempotent() -> None:
    a = _adapter()
    await a.end_step()
    await a.end_step()
    assert a._phase == "none"


async def test_end_step_closes_markdown_phase() -> None:
    a = _adapter()
    await a.on_stream_delta("hi")
    await a.end_step()
    a.markdown.finalize.assert_called_once()
    assert a._phase == "none"


async def test_pause_for_stdin_closes_active_live() -> None:
    a = _adapter()
    tc = MagicMock(id="t1")
    tc.name = "read_file"
    await a.on_tool_call(tc)
    await a.pause_for_stdin()
    a.tool_display.pause.assert_called_once()

    a2 = _adapter()
    await a2.on_stream_delta("x")
    await a2.pause_for_stdin()
    a2.markdown.pause.assert_called_once()


async def test_thinking_debounce_cancels_before_visible(
    fast_thinking: None,
) -> None:
    # Even with 0 debounce, immediate stop should cancel before the first tick
    # has a chance to call print. We call on_llm_call + stop back-to-back.
    a = _adapter()
    await a.on_llm_call()
    await a._thinking_line.stop()
    assert a._thinking_line.visible is False
    # console.print may or may not have been called once depending on scheduling;
    # what matters is that after stop, visible is False and no task is running.
    assert a._thinking_line.task is None


async def test_thinking_becomes_visible_after_debounce(
    fast_thinking: None,
) -> None:
    from agent_cli.render.status_lines import _THINKING_WORDS
    from agent_cli.theme import ELLIPSIS_FRAMES, SPINNER_FRAMES

    a = _adapter()
    await a.on_llm_call()
    # Let the loop fire at least once
    await asyncio.sleep(0.02)
    assert a._thinking_line.visible is True
    # Thinking bypasses Rich and writes to console.file directly.
    written = "".join(
        call.args[0] for call in a.console.file.write.call_args_list
    )
    # A flavor word from the pool
    assert any(w in written for w in _THINKING_WORDS)
    # Braille wave spinner frame
    assert any(g in written for g in SPINNER_FRAMES)
    # Breathing ellipsis (one of . / .. / …)
    assert any(e in written for e in ELLIPSIS_FRAMES)
    # Cancel hint removed; early frames don't show parens either
    assert "(ctrl+c to cancel)" not in written
    await a._thinking_line.stop()


async def test_stream_delta_stops_thinking(fast_thinking: None) -> None:
    a = _adapter()
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    assert a._thinking_line.visible is True
    await a.on_stream_delta("hi")
    assert a._thinking_line.visible is False
    assert a._thinking_line.task is None
    assert a._phase == "markdown"


async def test_tool_call_stops_thinking(fast_thinking: None) -> None:
    a = _adapter()
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    tc = MagicMock(id="t1")
    tc.name = "read_file"
    await a.on_tool_call(tc)
    assert a._thinking_line.visible is False
    assert a._thinking_line.task is None
    assert a._phase == "tools"


async def test_end_step_stops_thinking(fast_thinking: None) -> None:
    a = _adapter()
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    await a.end_step()
    assert a._thinking_line.visible is False
    assert a._thinking_line.task is None


async def test_pause_for_stdin_stops_thinking(fast_thinking: None) -> None:
    a = _adapter()
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    await a.pause_for_stdin()
    assert a._thinking_line.visible is False
    assert a._thinking_line.task is None


async def test_thinking_detail_shows_effort_past_threshold(
    monkeypatch: pytest.MonkeyPatch, fast_thinking: None,
) -> None:
    # Zero threshold so first tick already reveals the detail section.
    monkeypatch.setattr(status_lines_mod, "_DETAIL_AFTER_S", 0)
    a = CliAdapter(MagicMock(), FLEXOKI_DARK, effort="high")
    a.markdown = MagicMock()
    a.tool_display = MagicMock()
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    written = "".join(c.args[0] for c in a.console.file.write.call_args_list)
    assert "thinking with high effort" in written
    await a._thinking_line.stop()


async def test_thinking_detail_omits_effort_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch, fast_thinking: None,
) -> None:
    monkeypatch.setattr(status_lines_mod, "_DETAIL_AFTER_S", 0)
    a = _adapter()   # default effort=None
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    written = "".join(c.args[0] for c in a.console.file.write.call_args_list)
    assert "thinking with" not in written
    # elapsed still shown in parens
    assert "(" in written and "s)" in written
    await a._thinking_line.stop()


async def test_thinking_word_stable_within_run(fast_thinking: None) -> None:
    from agent_cli.render.status_lines import _THINKING_WORDS

    a = _adapter()
    await a.on_llm_call()
    await asyncio.sleep(0.03)     # several ticks
    written = "".join(c.args[0] for c in a.console.file.write.call_args_list)
    used = {w for w in _THINKING_WORDS if w in written}
    assert len(used) == 1, f"word should be fixed per-run, saw: {used}"
    await a._thinking_line.stop()
