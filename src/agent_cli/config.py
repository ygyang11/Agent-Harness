"""CLI startup config discovery + first-run bootstrap."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import agent_cli


@dataclass(frozen=True, slots=True)
class ConfigLoadResult:
    path: Path | None
    bootstrapped: bool
    effort: str | None = None


def load_config() -> ConfigLoadResult:
    from agent_harness.core.config import HarnessConfig

    project_cfg = Path.cwd() / "config.yaml"
    if project_cfg.exists():
        HarnessConfig.load(project_cfg)
        return ConfigLoadResult(
            path=project_cfg, bootstrapped=False, effort=_effort(),
        )

    user_cfg = Path.home() / ".agent-harness" / "config.yaml"
    bootstrapped = False
    if not user_cfg.exists():
        bootstrapped = _bootstrap_user_config(user_cfg)

    if user_cfg.exists():
        HarnessConfig.load(user_cfg)
        return ConfigLoadResult(
            path=user_cfg, bootstrapped=bootstrapped, effort=_effort(),
        )

    HarnessConfig.load(None, env_override=True)
    return ConfigLoadResult(path=None, bootstrapped=False, effort=_effort())


def _effort() -> str | None:
    from agent_harness.core.config import HarnessConfig
    return HarnessConfig.get().llm.reasoning_effort


def _bootstrap_user_config(dest: Path) -> bool:
    template = Path(agent_cli.__file__).resolve().parents[2] / "config_example.yaml"
    if not template.exists():
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError:
        return False
    return True
