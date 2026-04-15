"""Tests for SandboxManager and resolve_sandbox."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from agent_harness.core.config import SandboxConfig
from agent_harness.sandbox import resolve_sandbox
from agent_harness.sandbox.backend import LocalBackend
from agent_harness.sandbox.manager import SandboxManager


class TestManagerFromConfig:
    async def test_disabled_uses_local(self) -> None:
        mgr = SandboxManager.from_config(SandboxConfig(enabled=False))
        await mgr.start()
        assert isinstance(mgr.backend, LocalBackend)
        assert not mgr.is_sandboxed
        await mgr.stop()

    async def test_enabled_uses_docker(self) -> None:
        with patch("agent_harness.sandbox.docker.DockerBackend") as MockDocker:
            mock_instance = MagicMock()
            MockDocker.return_value = mock_instance
            SandboxManager.from_config(SandboxConfig(enabled=True))
        MockDocker.assert_called_once()

    async def test_custom_backend_constructor(self) -> None:
        custom = AsyncMock()
        custom.start = AsyncMock()
        mgr = SandboxManager(custom)
        await mgr.start()
        assert mgr.backend is custom
        custom.start.assert_called_once()
        await mgr.stop()

    async def test_start_idempotent(self) -> None:
        mgr = SandboxManager.from_config(SandboxConfig(enabled=False))
        await mgr.start()
        backend1 = mgr.backend
        await mgr.start()
        assert mgr.backend is backend1
        await mgr.stop()


class TestManagerLifecycle:
    async def test_stop_resets_started(self) -> None:
        mgr = SandboxManager.from_config(SandboxConfig(enabled=False))
        await mgr.start()
        await mgr.stop()
        assert not mgr._started

    async def test_execute_lazy_starts(self) -> None:
        mgr = SandboxManager.from_config(SandboxConfig(enabled=False))
        assert not mgr._started
        result = await mgr.execute("echo lazy")
        assert mgr._started
        assert "lazy" in result.stdout
        await mgr.stop()

    async def test_execute_delegates_to_backend(self) -> None:
        mgr = SandboxManager.from_config(SandboxConfig(enabled=False))
        result = await mgr.execute("echo test123")
        assert "test123" in result.stdout
        await mgr.stop()

    async def test_is_sandboxed_false_for_local(self) -> None:
        mgr = SandboxManager.from_config(SandboxConfig(enabled=False))
        await mgr.start()
        assert not mgr.is_sandboxed
        await mgr.stop()


class TestResolveSandbox:
    def test_custom_backend_takes_priority(self) -> None:
        custom = MagicMock()
        config = MagicMock()
        config.tool.sandbox = SandboxConfig(enabled=True)
        mgr = resolve_sandbox(custom, config)
        assert mgr.backend is custom

    def test_config_disabled_uses_local(self) -> None:
        config = MagicMock()
        config.tool.sandbox = SandboxConfig(enabled=False)
        mgr = resolve_sandbox(None, config)
        assert isinstance(mgr.backend, LocalBackend)

    def test_always_returns_manager(self) -> None:
        config = MagicMock()
        config.tool.sandbox = SandboxConfig(enabled=False)
        mgr = resolve_sandbox(None, config)
        assert isinstance(mgr, SandboxManager)
