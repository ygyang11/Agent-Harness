"""CLI-specific UI configuration.

Framework-level configuration lives in ``agent_harness.core.config.HarnessConfig``.
``CliConfig`` only holds UI-layer settings and is not user-mutable in Phase 1.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CliConfig:
    theme_name: str = "flexoki_dark"
    keybindings: str = "default"


def default_config() -> CliConfig:
    return CliConfig()
