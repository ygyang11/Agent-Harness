"""Build a fully-wired ReActAgent for the CLI."""
from __future__ import annotations

import sys

from agent_app.tools import BUILTIN_TOOLS
from agent_harness.agent.react import ReActAgent
from agent_harness.core.config import HarnessConfig
from agent_harness.hooks.base import DefaultHooks
from agent_harness.hooks.progress import ProgressHooks


def create_cli_agent(
    config: HarnessConfig | None = None,
    hooks: DefaultHooks | None = None,
) -> ReActAgent:
    """Build the CLI's ReActAgent.

    When ``hooks`` is None we construct ProgressHooks explicitly rather than
    going through ``resolve_hooks()``: tracing is enabled by default in
    HarnessConfig, so ``resolve_hooks(None, config)`` would return TracingHooks
    (file-only, no terminal UI) and the CLI would appear silent.

    ``max_steps`` is set to ``sys.maxsize`` because the CLI is an interactive
    REPL — a single run should never be terminated by a step ceiling.
    """
    cfg = config or HarnessConfig.get()
    if hooks is None:
        hooks = ProgressHooks()
    return ReActAgent(
        name="cli",
        tools=BUILTIN_TOOLS,
        hooks=hooks,
        config=cfg,
        stream=True,
        max_steps=sys.maxsize,
    )
