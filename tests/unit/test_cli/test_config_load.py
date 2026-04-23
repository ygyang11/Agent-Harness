from pathlib import Path
from unittest.mock import patch

import pytest

from agent_cli.config import ConfigLoadResult, load_config


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    return home


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


def test_project_config_wins(
    isolated_home: Path, isolated_cwd: Path,
) -> None:
    project = isolated_cwd / "config.yaml"
    project.write_text("llm:\n  provider: openai\n")
    with patch("agent_harness.core.config.HarnessConfig.load") as mock_load:
        result = load_config()
    assert result == ConfigLoadResult(path=project, bootstrapped=False)
    mock_load.assert_called_once_with(project)


def test_bootstrap_from_repo_template_when_user_config_missing(
    isolated_home: Path, isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_repo = isolated_cwd.parent / "repo"
    fake_pkg_dir = fake_repo / "src" / "agent_cli"
    fake_pkg_dir.mkdir(parents=True)
    (fake_repo / "config_example.yaml").write_text("# example\nllm:\n  provider: anthropic\n")
    fake_init = fake_pkg_dir / "__init__.py"
    fake_init.write_text("")

    import agent_cli as _agent_cli
    monkeypatch.setattr(_agent_cli, "__file__", str(fake_init))

    with patch("agent_harness.core.config.HarnessConfig.load") as mock_load:
        result = load_config()

    user_cfg = isolated_home / ".agent-harness" / "config.yaml"
    assert result == ConfigLoadResult(path=user_cfg, bootstrapped=True)
    assert user_cfg.exists()
    assert "anthropic" in user_cfg.read_text()
    mock_load.assert_called_once_with(user_cfg)


def test_fallback_when_template_missing(
    isolated_home: Path, isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pkg = isolated_cwd.parent / "pkg" / "agent_cli" / "__init__.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("")

    import agent_cli as _agent_cli
    monkeypatch.setattr(_agent_cli, "__file__", str(fake_pkg))

    with patch("agent_harness.core.config.HarnessConfig.load") as mock_load:
        result = load_config()

    assert result == ConfigLoadResult(path=None, bootstrapped=False)
    mock_load.assert_called_once_with(None, env_override=True)
