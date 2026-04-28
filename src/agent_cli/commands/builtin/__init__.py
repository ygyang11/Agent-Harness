"""builtin command registration."""
from __future__ import annotations

from agent_cli.commands.builtin import (
    clear,
    compact,
    context,
    debug,
    exit,
    help,
    model,
    status,
    theme,
    usage,
)
from agent_cli.commands.registry import CommandRegistry


def register_builtin(registry: CommandRegistry) -> None:
    for module in (
        help, exit, clear, compact, context, debug, model, status, theme, usage,
    ):
        registry.register_command(module.CMD)
