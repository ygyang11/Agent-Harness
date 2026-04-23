import asyncio
import re
from unittest.mock import MagicMock

import pytest

from agent_cli.adapter import CliAdapter
from agent_cli.render import status_lines as status_lines_mod
from agent_cli.render.live_line import CLEAR_LINE
from agent_cli.theme import FLEXOKI_DARK

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _visible(s: str) -> str:
    return _ANSI_RE.sub("", s)


@pytest.fixture
def fast_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(status_lines_mod, "THINKING_DEBOUNCE_S", 0.0)
    monkeypatch.setattr(status_lines_mod, "THINKING_TICK_S", 0.005)
    monkeypatch.setattr(status_lines_mod, "SUBAGENT_DEBOUNCE_S", 0.0)
    monkeypatch.setattr(status_lines_mod, "SUBAGENT_TICK_S", 0.005)


def _adapter() -> CliAdapter:
    console = MagicMock()
    console.file = MagicMock()
    return CliAdapter(console, FLEXOKI_DARK)


async def test_single_subagent_label_has_no_total_suffix(
    fast_loops: None,
) -> None:
    a = _adapter()
    await a.start_subagent()
    await asyncio.sleep(0.02)
    written = _visible("".join(c.args[0] for c in a.console.file.write.call_args_list))
    assert "SubAgent · " in written
    assert "0 steps" in written
    assert "0 tools" in written
    assert "· total" not in written
    await a.stop_subagent()


async def test_parallel_count_switches_to_aggregate_label(
    fast_loops: None,
) -> None:
    a = _adapter()
    await a.start_subagent()
    await a.start_subagent()
    await asyncio.sleep(0.02)
    written = "".join(c.args[0] for c in a.console.file.write.call_args_list)
    assert "2 Subagents" in written
    assert "· total" in written
    await a.stop_subagent()
    await a.stop_subagent()


async def test_tick_step_and_tool_visible_on_next_frame(
    fast_loops: None,
) -> None:
    a = _adapter()
    await a.start_subagent()
    a.tick_subagent_step()
    a.tick_subagent_step()
    a.tick_subagent_tool()
    await asyncio.sleep(0.02)
    written = "".join(c.args[0] for c in a.console.file.write.call_args_list)
    assert "2 steps" in written
    assert "1 tools" in written
    await a.stop_subagent()


async def test_total_suffix_persists_when_dropping_back_to_one(
    fast_loops: None,
) -> None:
    a = _adapter()
    await a.start_subagent()
    await a.start_subagent()
    await asyncio.sleep(0.02)
    a.console.file.write.reset_mock()

    await a.stop_subagent()
    await asyncio.sleep(0.02)
    written = _visible("".join(c.args[0] for c in a.console.file.write.call_args_list))
    assert "SubAgent · " in written
    assert "· total" in written

    await a.stop_subagent()


async def test_total_suffix_resets_on_fresh_session(
    fast_loops: None,
) -> None:
    a = _adapter()
    await a.start_subagent()
    await a.start_subagent()
    await asyncio.sleep(0.02)
    await a.stop_subagent()
    await a.stop_subagent()

    a.console.file.write.reset_mock()
    await a.start_subagent()
    await asyncio.sleep(0.02)
    written = _visible("".join(c.args[0] for c in a.console.file.write.call_args_list))
    assert "SubAgent · " in written
    assert "· total" not in written
    await a.stop_subagent()


async def test_start_subagent_hands_off_from_parent_thinking(
    fast_loops: None,
) -> None:
    a = _adapter()
    await a.on_llm_call()
    await asyncio.sleep(0.02)
    assert a._thinking_line.visible is True
    assert a._thinking_line.task is not None

    await a.start_subagent()
    assert a._thinking_line.task is None
    assert a._thinking_line.visible is False

    await asyncio.sleep(0.02)
    assert a._subagent_line.visible is True

    await a.stop_subagent()


async def test_print_inline_clears_live_lines_before_printing(
    fast_loops: None,
) -> None:
    a = _adapter()
    await a.start_subagent()
    await a.start_subagent()
    await asyncio.sleep(0.02)
    assert a._subagent_line.visible is True

    await a.stop_subagent()
    assert a._subagent_line.active_count == 1
    assert a._subagent_line.task is not None

    a.console.file.write.reset_mock()
    await a.print_inline("done-line")
    writes = [c.args[0] for c in a.console.file.write.call_args_list]
    assert any(CLEAR_LINE in w for w in writes), (
        "print_inline must emit CLEAR_LINE before Rich console.print to "
        "avoid colliding with active spinner row"
    )

    await a.stop_subagent()


async def test_pause_for_stdin_keeps_subagent_task_alive_after_unlock(
    fast_loops: None,
) -> None:
    a = _adapter()
    await a.start_subagent()
    await asyncio.sleep(0.02)
    assert a._subagent_line.visible is True
    task_before = a._subagent_line.task
    assert task_before is not None and not task_before.done()

    async with a._console_lock:
        await a.pause_for_stdin()
        assert a._subagent_line.task is task_before
        assert not task_before.done()
        assert a._subagent_line.visible is False
        await asyncio.sleep(0.02)
        assert not task_before.done()

    await asyncio.sleep(0.02)
    assert a._subagent_line.visible is True
    assert a._subagent_line.active_count == 1
    await a.stop_subagent()


async def test_end_step_force_stops_subagent_as_defense(
    fast_loops: None,
) -> None:
    a = _adapter()
    await a.start_subagent()
    await a.end_step()
    assert a._subagent_line.active_count == 0
    assert a._subagent_line.task is None
    assert a._subagent_line.was_parallel is False


async def test_concurrent_start_does_not_clobber_was_parallel(
    fast_loops: None,
) -> None:
    """Regression: when two subagents call start_subagent concurrently
    (the real parallel-subagent scenario), the first call's 0→1 branch
    awaits before setting was_parallel; during that await the second
    call's 1→2 elif branch latches was_parallel=True. If state mutations
    aren't inlined BEFORE the awaits, the first call resumes and resets
    was_parallel=False, making the '· total' suffix disappear after one
    of the subagents ends.

    Reproduction: kick both start_subagent calls with asyncio.gather so
    they interleave at the first await point inside the 0→1 branch.
    """
    a = _adapter()
    await asyncio.gather(a.start_subagent(), a.start_subagent())
    assert a._subagent_line.active_count == 2
    assert a._subagent_line.was_parallel is True, (
        "race lost: second start_subagent latched was_parallel=True, "
        "but first start_subagent's post-await reset clobbered it"
    )

    # End one subagent; was_parallel must persist so the remaining spinner
    # frame still shows '· total'.
    await a.stop_subagent()
    await asyncio.sleep(0.02)
    written = _visible("".join(c.args[0] for c in a.console.file.write.call_args_list))
    assert "SubAgent · " in written
    assert "· total" in written
    await a.stop_subagent()
