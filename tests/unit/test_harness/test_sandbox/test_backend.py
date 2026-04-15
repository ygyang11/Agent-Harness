"""Tests for LocalBackend — passthrough execution."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_harness.sandbox.backend import ExecuteResult, LocalBackend


class TestExecuteResult:
    def test_fields(self) -> None:
        r = ExecuteResult(exit_code=0, stdout="hello")
        assert r.exit_code == 0
        assert r.stdout == "hello"
        assert r.stderr == ""

    def test_none_exit_code(self) -> None:
        r = ExecuteResult(exit_code=None, stdout="timeout")
        assert r.exit_code is None

    def test_with_stderr(self) -> None:
        r = ExecuteResult(exit_code=1, stdout="out", stderr="err")
        assert r.stderr == "err"


class TestLocalBackend:
    @pytest.fixture
    def backend(self) -> LocalBackend:
        return LocalBackend()

    async def test_simple_command(self, backend: LocalBackend) -> None:
        result = await backend.execute("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout

    async def test_exit_code(self, backend: LocalBackend) -> None:
        result = await backend.execute("exit 42")
        assert result.exit_code == 42

    async def test_timeout(self, backend: LocalBackend) -> None:
        result = await backend.execute("sleep 10", timeout=0.1)
        assert result.exit_code is None
        assert "timed out" in result.stdout

    async def test_stderr_separate(self, backend: LocalBackend) -> None:
        result = await backend.execute("echo err >&2")
        assert "err" in result.stderr
        assert result.stdout == ""

    async def test_stdout_and_stderr(self, backend: LocalBackend) -> None:
        result = await backend.execute("echo out && echo err >&2")
        assert "out" in result.stdout
        assert "err" in result.stderr

    async def test_workdir(self, backend: LocalBackend, tmp_path: Path) -> None:
        result = await backend.execute("pwd", workdir=str(tmp_path))
        assert str(tmp_path) in result.stdout

    async def test_start_stop_noop(self, backend: LocalBackend) -> None:
        await backend.start()
        await backend.stop()

    async def test_empty_output(self, backend: LocalBackend) -> None:
        result = await backend.execute("true")
        assert result.exit_code == 0
        assert result.stdout == ""

    async def test_command_not_found(self, backend: LocalBackend) -> None:
        result = await backend.execute("nonexistent_command_xyz_999")
        assert result.exit_code != 0

    async def test_cancelled_kills_process(self, backend: LocalBackend) -> None:
        task = asyncio.create_task(backend.execute("sleep 100", timeout=60))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
