import sys

from agent_app.tools import BUILTIN_TOOLS
from agent_cli.agent_factory import create_cli_agent
from agent_harness.agent.react import ReActAgent
from agent_harness.hooks.progress import ProgressHooks


def test_cli_agent_is_react_with_expected_defaults() -> None:
    agent = create_cli_agent()
    assert isinstance(agent, ReActAgent)
    assert agent.name == "cli"
    assert agent._stream is True
    assert agent.max_steps == sys.maxsize
    assert isinstance(agent.hooks, ProgressHooks)


def test_cli_agent_exposes_all_builtin_tools() -> None:
    agent = create_cli_agent()
    actual = {t.name for t in agent.tools}
    expected = {t.name for t in BUILTIN_TOOLS}
    assert actual == expected
