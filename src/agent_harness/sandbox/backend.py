"""Sandbox backend base class and local (passthrough) implementation."""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecuteResult:
    """Result of a sandboxed command execution."""

    exit_code: int | None
    stdout: str
    stderr: str = ""


class SandboxBackend(ABC):
    """Base class for sandbox backends.

    Built-in: LocalBackend (passthrough), DockerBackend (container isolation).
    Extend this class to integrate third-party sandbox providers.
    """

    async def start(self, workspace: str | None = None) -> None:
        """Initialize the backend (e.g. create Docker container)."""

    @abstractmethod
    async def execute(
        self,
        command: str,
        *,
        timeout: float = 30.0,
        workdir: str | None = None,
    ) -> ExecuteResult:
        """Execute a command in the sandbox."""

    async def stop(self) -> None:
        """Clean up backend resources (e.g. destroy container)."""


class LocalBackend(SandboxBackend):
    """Passthrough backend — executes directly on the host, no isolation.

    Output is returned as raw stdout/stderr without truncation.
    Callers (terminal_tool) handle merging and truncation.
    """

    async def execute(
        self,
        command: str,
        *,
        timeout: float = 30.0,
        workdir: str | None = None,
    ) -> ExecuteResult:
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            return ExecuteResult(
                exit_code=None, stdout=f"Error: execution timed out after {timeout}s",
            )
        except asyncio.CancelledError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            raise
        except Exception as exc:  # noqa: BLE001
            return ExecuteResult(exit_code=None, stdout=f"Error: failed to execute command: {exc}")

        stdout_text = stdout_bytes.decode(errors="replace")
        stderr_text = stderr_bytes.decode(errors="replace")

        return ExecuteResult(
            exit_code=proc.returncode,
            stdout=stdout_text,
            stderr=stderr_text,
        )
