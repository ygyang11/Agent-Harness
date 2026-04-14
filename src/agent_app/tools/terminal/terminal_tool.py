"""Terminal tool for shell command execution.

Security model — three layers:
- Layer 1 (Permission): ApprovalPolicy decides if execution is allowed
- Layer 2 (Sandbox): Not implemented yet (Phase 3)
- Layer 3 (Tool): Pure execution — no command filtering

The tool layer handles:
- Timeout enforcement and subprocess cleanup
- Output normalization and truncation
- Optional background execution

Commands run from the workspace root with full system access.
Security is enforced by ApprovalPolicy, not by the tool layer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.utils.token_counter import truncate_text_by_tokens

_MAX_OUTPUT_TOKENS: int = 10_000
_DEFAULT_TIMEOUT: int = 120  # seconds
_MAX_TIMEOUT: int = 600  # seconds (10 minutes)
_MAX_BG_TIMEOUT: int = 86400  # seconds (24 hours) — background tasks can run much longer

TERMINAL_TOOL_DESCRIPTION = (
    "Execute a shell command in a bash subprocess and return its output.\n\n"
    "The working directory defaults to the workspace root. "
    "Shell state (variables, aliases) does not persist between calls — "
    "each invocation starts a fresh subprocess.\n\n"
    "Use this tool for operations without a dedicated equivalent: "
    "git, pytest, pip, npm, make, docker, curl, python/node/bash scripts, etc.\n\n"
    "Set background=true for commands that take a long time where blocking "
    "would be wasteful (e.g. model training, large builds, long data processing). "
    "Results are delivered automatically when complete.\n\n"
    "Examples:\n"
    "  Good:\n"
    "    terminal_tool(command='pytest tests/ -v')\n"
    "    terminal_tool(command='python scripts/migrate.py')\n"
    '    terminal_tool(command=\'git add -A && git commit -m "fix bug"\')\n'
    "    terminal_tool(command='cd src && python -m mymodule', timeout=300)\n"
    "    terminal_tool(command='python train.py --epochs=50', background=true)\n"
    "  Bad:\n"
    "    terminal_tool(command='cat file.txt')       # use dedicated file reading tool\n"
    "    terminal_tool(command='grep -r pattern .')   # use dedicated search tool\n"
    "    terminal_tool(command='find . -name *.py')   # use dedicated file search tool\n"
    "    terminal_tool(command='python scripts/check_env.py', background=true)  # need result before proceeding\n\n"
    "Guidelines:\n"
    "- Always quote file paths containing spaces with double quotes\n"
    "- If a command creates files or directories, verify the parent exists first\n"
    "- Chain dependent commands with && (e.g. 'cd src && pytest')\n"
    "- Use ; only when you don't care if earlier commands fail\n"
    "- Set appropriate timeout for long-running commands (default 120s, max 600s; "
    "background mode allows up to 24h)"
)


def _workspace_root() -> Path:
    return Path.cwd().resolve()


async def _execute_command(
    command: str,
    cwd: Path,
    timeout: int,
) -> tuple[int | None, str]:
    """Execute a shell command and return (exit_code, normalized_output)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return None, f"Error: execution timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return None, f"Error: failed to execute command: {exc}"

    stdout_text = stdout.decode(errors="replace")
    stderr_text = stderr.decode(errors="replace")
    merged = stdout_text if not stderr_text else f"{stdout_text}\n{stderr_text}".strip()

    if not merged:
        return proc.returncode, ""

    output = truncate_text_by_tokens(
        merged,
        max_tokens=_MAX_OUTPUT_TOKENS,
        suffix="\n... (truncated)",
    )
    return proc.returncode, output


class TerminalTool(BaseTool):

    def __init__(self) -> None:
        super().__init__(
            name="terminal_tool",
            description=TERMINAL_TOOL_DESCRIPTION,
            executor_timeout=_MAX_TIMEOUT + 10,
            approval_resource_key="command",
        )
        self._agent: Any = None

    def bind_agent(self, agent: Any) -> None:
        self._agent = agent

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Shell command to execute. Supports full bash syntax "
                            "including pipes, redirects, chaining, and variable expansion."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            f"Maximum execution time in seconds "
                            f"(default {_DEFAULT_TIMEOUT}, max {_MAX_TIMEOUT}; "
                            f"background mode max {_MAX_BG_TIMEOUT})."
                        ),
                        "default": _DEFAULT_TIMEOUT,
                    },
                    "background": {
                        "type": "boolean",
                        "description": (
                            "Run in background. Returns a task ID immediately; "
                            "results are delivered automatically when complete."
                        ),
                        "default": False,
                    },
                },
                "required": ["command"],
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command", "")
        timeout = kwargs.get("timeout", _DEFAULT_TIMEOUT)
        background = kwargs.get("background", False)

        if not command.strip():
            return "Error: command cannot be empty"
        if timeout <= 0:
            return "Error: timeout must be greater than 0"

        if background and self._agent is not None:
            timeout = min(timeout, _MAX_BG_TIMEOUT)
            return self._start_background(command, timeout)

        timeout = min(timeout, _MAX_TIMEOUT)

        exit_code, output = await _execute_command(command, _workspace_root(), timeout)
        if exit_code is None:
            return output
        if exit_code != 0:
            return f"[exit code {exit_code}]\n{output}".rstrip()
        return output or "(no output)"

    def _start_background(self, command: str, timeout: int) -> str:
        async def work() -> tuple[str, str]:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", command,
                cwd=str(_workspace_root()),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except (TimeoutError, asyncio.CancelledError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
                raise
            full_output = stdout.decode(errors="replace")
            if stderr:
                full_output += "\n" + stderr.decode(errors="replace")
            full_output = full_output.strip()
            summary = _build_terminal_summary(proc.returncode, full_output)
            return full_output, summary

        desc = truncate_text_by_tokens(command, max_tokens=12, suffix="...")
        task_id = self._agent._bg_manager.spawn(
            tool_name="terminal_tool",
            description=desc,
            coro=work(),
        )
        return f"Background command {task_id} started: {command}"


def _build_terminal_summary(exit_code: int | None, output: str) -> str:
    lines = output.splitlines()
    head = "\n".join(lines[:3]) if len(lines) > 3 else ""
    tail = "\n".join(lines[-5:]) if len(lines) > 5 else output
    parts: list[str] = []
    if exit_code is not None:
        parts.append(f"Exit code: {exit_code}")
    else:
        parts.append("Exit code: N/A (timeout or error)")
    parts.append(f"Output: {len(lines)} lines")
    if head and head != tail:
        parts.append(f"Head:\n{head}")
    if tail:
        parts.append(f"Tail:\n{tail}")
    return "\n".join(parts)


terminal_tool = TerminalTool()
