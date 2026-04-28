"""Inline notice formatters for adapter.print_inline (non-tool)."""

from __future__ import annotations

from rich.text import Text

from agent_cli.theme import TOOL_DONE
from agent_harness.utils.token_counter import truncate_text_by_tokens

_SHELL_LANE_OUTPUT_TOKENS = 10_000
_SHELL_RUN_CLOSE_TAG = "</user-shell-run>"
_SHELL_RUN_CLOSE_TAG_ESCAPED = "</user-shell-run​>"


def _escape_envelope(s: str) -> str:
    return s.replace(_SHELL_RUN_CLOSE_TAG, _SHELL_RUN_CLOSE_TAG_ESCAPED)


def format_warning(message: str) -> Text:
    t = Text()
    t.append(f"{TOOL_DONE}  ", style="error")
    t.append(message, style="muted")
    return t


def format_expired_notice(ids: list[int]) -> Text:
    ids_str = ", ".join(f"#{i}" for i in ids)
    return format_warning(f"Pasted text {ids_str} unavailable")


def format_shell_run(
    command: str,
    exit_code: int,
    output: str,
    post_notices: list[str] | None = None,
) -> str:
    """Format a `!`-lane shell run as the body of a ``Message.user``."""
    truncated = truncate_text_by_tokens(
        output,
        max_tokens=_SHELL_LANE_OUTPUT_TOKENS,
        suffix="\n... (truncated)",
    )
    has_output = bool(truncated.strip())

    if exit_code != 0 and has_output:
        body = f"[exit code {exit_code}]\n{truncated}"
    elif exit_code != 0:
        body = f"[exit code {exit_code}]\n(Completed with no output)"
    elif has_output:
        body = truncated
    else:
        body = "(Completed with no output)"

    if post_notices:
        body = body + "\n" + "\n".join(f"[Accident] {n}" for n in post_notices)

    safe_command = _escape_envelope(command)
    safe_body = _escape_envelope(body)
    return f"<user-shell-run>\n```sh\n{safe_command}\n```\n{safe_body}\n</user-shell-run>"
