"""Unit tests for BackgroundTaskManager."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_harness.background import BackgroundTaskManager


async def _immediate(value: str = "done") -> tuple[str, str]:
    return value, f"Result: {value}"


async def _slow(seconds: float = 10) -> tuple[str, str]:
    await asyncio.sleep(seconds)
    return "done", "Result: done"


async def _failing() -> tuple[str, str]:
    raise RuntimeError("boom")


@pytest.fixture
def manager(tmp_path: Path) -> BackgroundTaskManager:
    m = BackgroundTaskManager()
    m._output_dir = str(tmp_path / "bg_output")
    return m


# -- Lifecycle --


class TestSpawn:
    async def test_returns_task_id(self, manager: BackgroundTaskManager) -> None:
        tid = manager.spawn("terminal", "test", _immediate())
        assert tid.startswith("bg_")

    async def test_task_runs_and_completes(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("terminal", "test", _immediate("hello"))
        await asyncio.sleep(0.05)
        completed = manager.collect_completed()
        assert len(completed) == 1
        assert completed[0].status == "completed"
        assert completed[0].result is not None
        assert completed[0].result.summary == "Result: hello"

    async def test_multiple_independent(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("a", "t1", _immediate("one"))
        manager.spawn("b", "t2", _immediate("two"))
        await asyncio.sleep(0.05)
        completed = manager.collect_completed()
        assert len(completed) == 2

    async def test_task_id_increments(self, manager: BackgroundTaskManager) -> None:
        id1 = manager.spawn("a", "t1", _immediate())
        id2 = manager.spawn("b", "t2", _immediate())
        assert id1 != id2
        n1 = int(id1.split("_")[1])
        n2 = int(id2.split("_")[1])
        assert n2 == n1 + 1


# -- Collect --


class TestCollect:
    async def test_returns_once(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("terminal", "test", _immediate())
        await asyncio.sleep(0.05)
        assert len(manager.collect_completed()) == 1
        assert len(manager.collect_completed()) == 0

    async def test_empty_when_running(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("terminal", "test", _slow())
        assert len(manager.collect_completed()) == 0

    async def test_skips_cancelled(self, manager: BackgroundTaskManager) -> None:
        tid = manager.spawn("terminal", "test", _slow())
        manager.cancel(tid)
        await asyncio.sleep(0.05)
        assert len(manager.collect_completed()) == 0

    async def test_includes_failed(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("terminal", "test", _failing())
        await asyncio.sleep(0.05)
        completed = manager.collect_completed()
        assert len(completed) == 1
        assert completed[0].status == "failed"

    async def test_multiple(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("a", "t1", _immediate("one"))
        manager.spawn("b", "t2", _immediate("two"))
        manager.spawn("c", "t3", _immediate("three"))
        await asyncio.sleep(0.05)
        assert len(manager.collect_completed()) == 3


# -- Running --


class TestRunning:
    async def test_has_running_true(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("terminal", "test", _slow())
        assert manager.has_running() is True

    async def test_has_running_false_when_done(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("terminal", "test", _immediate())
        await asyncio.sleep(0.05)
        assert manager.has_running() is False

    async def test_has_running_false_after_cancel(self, manager: BackgroundTaskManager) -> None:
        tid = manager.spawn("terminal", "test", _slow())
        manager.cancel(tid)
        assert manager.has_running() is False


# -- Wait --


class TestWaitNext:
    async def test_returns_completed(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("terminal", "test", _immediate())
        task = await manager.wait_next()
        assert task is not None
        assert task.status == "completed"

    async def test_no_running_returns_none(self, manager: BackgroundTaskManager) -> None:
        result = await manager.wait_next()
        assert result is None

    async def test_multiple_picks_first(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("a", "fast", _immediate())
        manager.spawn("b", "slow", _slow())
        task = await manager.wait_next()
        assert task is not None
        assert task.tool_name == "a"


# -- Cancel --


class TestCancel:
    async def test_running_returns_true(self, manager: BackgroundTaskManager) -> None:
        tid = manager.spawn("terminal", "test", _slow())
        assert manager.cancel(tid) is True

    async def test_sets_status(self, manager: BackgroundTaskManager) -> None:
        tid = manager.spawn("terminal", "test", _slow())
        manager.cancel(tid)
        assert manager.get_task(tid).status == "cancelled"

    async def test_nonexistent_returns_false(self, manager: BackgroundTaskManager) -> None:
        assert manager.cancel("bg_999") is False

    async def test_already_completed(self, manager: BackgroundTaskManager) -> None:
        tid = manager.spawn("terminal", "test", _immediate())
        await asyncio.sleep(0.05)
        assert manager.cancel(tid) is False

    async def test_already_cancelled(self, manager: BackgroundTaskManager) -> None:
        tid = manager.spawn("terminal", "test", _slow())
        manager.cancel(tid)
        assert manager.cancel(tid) is False

    async def test_cancel_all(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("a", "t1", _slow())
        manager.spawn("b", "t2", _slow())
        assert manager.cancel_all() == 2

    async def test_cancel_all_none_running(self, manager: BackgroundTaskManager) -> None:
        assert manager.cancel_all() == 0


# -- Failed --


class TestFailed:
    async def test_status_and_error(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("terminal", "test", _failing())
        await asyncio.sleep(0.05)
        task = manager.get_task(manager.get_all()[0].task_id)
        assert task.status == "failed"
        assert "boom" in (task.error or "")

    async def test_no_asyncio_warning(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("terminal", "test", _failing())
        await asyncio.sleep(0.05)
        # If exception leaked, asyncio would log a warning.
        # No assertion needed — test passes if no warning/error.

    async def test_has_result(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("terminal", "test", _failing())
        await asyncio.sleep(0.05)
        task = manager.get_all()[0]
        assert task.result is not None
        assert "Error:" in task.result.summary


# -- Output --


class TestOutput:
    async def test_creates_file(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("terminal", "test", _immediate("file content"))
        await asyncio.sleep(0.05)
        task = manager.get_all()[0]
        assert task.result.output_path is not None
        assert Path(task.result.output_path).read_text() == "file content"

    async def test_path_in_result(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("terminal", "test", _immediate("data"))
        await asyncio.sleep(0.05)
        task = manager.get_all()[0]
        assert task.result.output_path.endswith(".txt")

    async def test_dir_created_if_missing(self, manager: BackgroundTaskManager) -> None:
        manager._output_dir = str(Path(manager._output_dir) / "nested" / "deep")
        manager.spawn("terminal", "test", _immediate("x"))
        await asyncio.sleep(0.05)
        task = manager.get_all()[0]
        assert Path(task.result.output_path).exists()

    def test_cleanup_on_init(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "cleanup_test"
        out_dir.mkdir()
        (out_dir / "old.txt").write_text("stale")
        m = BackgroundTaskManager()
        m._output_dir = str(out_dir)
        m._cleanup_output_dir()
        assert not out_dir.exists()


# -- Session --


class TestSession:
    def test_bind_session_changes_dir(self, manager: BackgroundTaskManager) -> None:
        manager.bind_session("sess_abc")
        assert "sess_abc" in manager._output_dir

    def test_default_isolation_by_pid(self) -> None:
        import os

        m = BackgroundTaskManager()
        assert str(os.getpid()) in m._output_dir

    def test_bind_same_session_noop(self, manager: BackgroundTaskManager) -> None:
        manager.bind_session("sess_abc")
        dir1 = manager._output_dir
        manager.bind_session("sess_abc")
        dir2 = manager._output_dir
        assert dir1 == dir2


# -- Summary --


class TestSummary:
    async def test_with_tasks(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("terminal", "pytest", _slow())
        summary = manager.get_running_summary()
        assert summary is not None
        assert "1 background task" in summary
        assert "pytest" in summary

    async def test_none_when_empty(self, manager: BackgroundTaskManager) -> None:
        assert manager.get_running_summary() is None

    async def test_excludes_completed(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("terminal", "done", _immediate())
        await asyncio.sleep(0.05)
        assert manager.get_running_summary() is None


# -- Get --


class TestGet:
    async def test_exists(self, manager: BackgroundTaskManager) -> None:
        tid = manager.spawn("terminal", "test", _immediate())
        assert manager.get_task(tid) is not None

    def test_nonexistent(self, manager: BackgroundTaskManager) -> None:
        assert manager.get_task("bg_999") is None

    async def test_get_all(self, manager: BackgroundTaskManager) -> None:
        manager.spawn("a", "t1", _immediate())
        tid = manager.spawn("b", "t2", _slow())
        manager.cancel(tid)
        await asyncio.sleep(0.05)
        all_tasks = manager.get_all()
        assert len(all_tasks) == 2
        statuses = {t.status for t in all_tasks}
        assert "completed" in statuses
        assert "cancelled" in statuses
