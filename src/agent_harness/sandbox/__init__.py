"""Sandbox execution backends for terminal tool isolation."""
from __future__ import annotations

from typing import TYPE_CHECKING

from agent_harness.sandbox.backend import ExecuteResult, LocalBackend, SandboxBackend
from agent_harness.sandbox.manager import SandboxManager

if TYPE_CHECKING:
    from agent_harness.core.config import HarnessConfig


def resolve_sandbox(
    sandbox: SandboxBackend | None,
    config: HarnessConfig,
) -> SandboxManager:
    """Resolve sandbox backend from user injection or config.

    Always returns a SandboxManager (never None):
    - Custom backend provided -> SandboxManager(backend)
    - Config enabled         -> SandboxManager.from_config() -> DockerBackend
    - Config disabled        -> SandboxManager.from_config() -> LocalBackend
    """
    if sandbox is not None:
        return SandboxManager(sandbox)
    return SandboxManager.from_config(config.tool.sandbox)


__all__ = [
    "ExecuteResult",
    "LocalBackend",
    "SandboxBackend",
    "SandboxManager",
    "resolve_sandbox",
]
