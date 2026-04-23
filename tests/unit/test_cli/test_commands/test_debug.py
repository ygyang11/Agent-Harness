from unittest.mock import MagicMock

import pytest

from agent_cli import hooks
from agent_cli.commands.builtin.debug import CMD

from .conftest import render_output


@pytest.fixture(autouse=True)
def reset_debug() -> None:
    hooks._debug_enabled[0] = False
    yield
    hooks._debug_enabled[0] = False


async def test_first_call_turns_on() -> None:
    result = await CMD.handler(MagicMock(), "")
    assert hooks.is_debug_enabled() is True
    assert "Debug mode on" in render_output(result.output)


async def test_second_call_turns_off() -> None:
    await CMD.handler(MagicMock(), "")
    result = await CMD.handler(MagicMock(), "")
    assert hooks.is_debug_enabled() is False
    assert "Debug mode off" in render_output(result.output)


async def test_args_ignored() -> None:
    result = await CMD.handler(MagicMock(), "whatever garbage")
    assert hooks.is_debug_enabled() is True
    assert "Debug mode on" in render_output(result.output)


def test_command_metadata() -> None:
    assert CMD.name == "/debug"
    assert "traceback" in CMD.description.lower()
