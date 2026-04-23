"""/theme — list or switch the CLI color theme (applies on next restart)."""
from __future__ import annotations

from rich.console import Group
from rich.text import Text

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import err, info, ok
from agent_cli.theme import SEP_DOT, THEMES, available_names, save_theme


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    name = args.strip()
    if not name:
        rows: list[Text] = [info("Available themes:"), Text("")]
        for n in available_names():
            row = Text("  ")
            row.append(SEP_DOT, style="muted")
            row.append(" ")
            row.append(n)
            rows.append(row)
        return CommandResult(output=Group(*rows))

    if name not in THEMES:
        return CommandResult(output=err(
            f"Unknown theme: {name}",
            (f" · Available: {', '.join(available_names())}", "muted"),
        ))

    try:
        save_theme(name)
    except OSError as e:
        return CommandResult(output=err(f"Failed to save theme preference: {e}"))
    return CommandResult(output=ok(
        "Theme set to ",
        (name, "bold"),
        (" · Restart to apply", "muted"),
    ))


CMD = Command(
    name="/theme",
    description="Show or switch the CLI color theme (applies on restart)",
    handler=handle,
)
