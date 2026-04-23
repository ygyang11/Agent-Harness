"""Build a fully-wired ReActAgent for the CLI."""
from __future__ import annotations

import sys

from agent_app.tools import BUILTIN_TOOLS
from agent_harness.agent.react import ReActAgent
from agent_harness.approval.handler import ApprovalHandler
from agent_harness.core.config import HarnessConfig
from agent_harness.hooks.base import DefaultHooks
from agent_harness.hooks.progress import ProgressHooks


def create_cli_agent(
    config: HarnessConfig | None = None,
    hooks: DefaultHooks | None = None,
    approval_handler: ApprovalHandler | None = None,
) -> ReActAgent:
    """Build the CLI's ReActAgent.

    ``max_steps`` is set to ``sys.maxsize`` because the CLI is an interactive
    REPL — a single run should never be terminated by a step ceiling.
    """
    from agent_cli.approval_handler import CliApprovalHandler

    cfg = config or HarnessConfig.get()
    if hooks is None:
        hooks = ProgressHooks()
    if approval_handler is None:
        approval_handler = CliApprovalHandler()

    agent = ReActAgent(
        name="cli",
        tools=BUILTIN_TOOLS,
        hooks=hooks,
        approval_handler=approval_handler,
        config=cfg,
        stream=True,
        max_steps=sys.maxsize,
    )
    if isinstance(approval_handler, CliApprovalHandler):
        approval_handler.bind_agent(agent)
    return agent
