"""builtin command registration."""
from __future__ import annotations

from agent_cli.commands.builtin import (
    clear,
    compact,
    debug,
    exit,
    help,
    model,
    status,
    theme,
)
from agent_cli.commands.registry import CommandRegistry


def register_builtin(registry: CommandRegistry) -> None:
    for module in (help, exit, clear, compact, debug, model, status, theme):
        registry.register_command(module.CMD)
