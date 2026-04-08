"""Runtime context provider — ephemeral environment info injected per LLM call."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

from agent_harness.core.message import Message

logger = logging.getLogger(__name__)

_GIT_STATUS_MAX_LINES = 50
_GIT_LOG_COUNT = 5


class RuntimeContextProvider:
    """Builds ephemeral runtime context as a system message.

    Injected into the LLM message list at call_llm() time but NOT persisted
    to short-term memory or sessions. Contains information that changes
    between calls: date, platform, cwd, git status, recent commits.
    """

    def build_context_message(self) -> Message | None:
        """Build runtime context system message. Returns None if empty."""
        cwd = str(Path.cwd())
        parts: list[str] = ["# Environment"]

        is_git = self._is_git_repo(cwd)
        parts.append(f"- Primary working directory: {cwd}")
        if is_git:
            parts.append("  - Is a git repository: true")

        parts.append(f"- Platform: {platform.system().lower()}")
        shell = os.environ.get("SHELL", "")
        if shell:
            parts.append(f"- Shell: {shell.rsplit('/', 1)[-1]}")
        parts.append(f"- OS Version: {platform.platform()}")
        parts.append(f"- Current date: {datetime.now().strftime('%Y-%m-%d')}")

        if is_git:
            status = self._read_git_status(cwd)
            if status:
                parts.append(f"\nGit status:\n```\n{status}\n```")

            log = self._read_git_log(cwd)
            if log:
                parts.append(f"\nRecent commits:\n```\n{log}\n```")

        return Message.system("\n".join(parts))

    @staticmethod
    def _is_git_repo(cwd: str) -> bool:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=3,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def _read_git_status(cwd: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "--no-optional-locks", "status", "--short", "--branch"],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=5,
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                lines = output.splitlines()
                if len(lines) > _GIT_STATUS_MAX_LINES:
                    return (
                        "\n".join(lines[:_GIT_STATUS_MAX_LINES])
                        + f"\n... ({len(lines) - _GIT_STATUS_MAX_LINES} more files)"
                    )
                return output
        except (OSError, subprocess.TimeoutExpired):
            pass
        return None

    @staticmethod
    def _read_git_log(cwd: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", f"-{_GIT_LOG_COUNT}"],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        return None
