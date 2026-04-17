"""CLIApp — top-level lifecycle and entrypoint."""
from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Agent-Harness interactive CLI",
    )
    parser.add_argument("--version", action="store_true", help="show version and exit")
    return parser


def _discover_and_load_config() -> None:
    """Locate and load HarnessConfig from the first available source.

    Order: ``$CWD/config.yaml`` → ``~/.agent-harness/config.yaml``. When none
    exists, fall back to defaults merged with ``HARNESS_*`` environment vars.
    """
    from agent_harness.core.config import HarnessConfig

    for candidate in (
        Path.cwd() / "config.yaml",
        Path.home() / ".agent-harness" / "config.yaml",
    ):
        if candidate.exists():
            HarnessConfig.load(candidate)
            return
    HarnessConfig.load(None, env_override=True)


async def _async_main() -> int:
    from agent_cli import __version__, _check_deps

    _check_deps()

    from rich.console import Console

    from agent_cli.agent_factory import create_cli_agent
    from agent_cli.render.ui import render_welcome
    from agent_cli.repl.loop import run_repl

    _discover_and_load_config()

    console = Console()
    agent = create_cli_agent()
    session_id = str(uuid.uuid4())

    render_welcome(
        console,
        version=__version__,
        model=agent.llm.model_name,
        cwd=str(Path.cwd()),
    )

    await run_repl(agent, console, session_id)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.version:
        from agent_cli import __version__
        print(f"harness {__version__}")
        return 0
    try:
        return asyncio.run(_async_main())
    except KeyboardInterrupt:
        return 130
