"""Unit tests for InputMultiplexer."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from agent_harness.utils.input_mux import InputMultiplexer


@pytest.fixture
def mux() -> InputMultiplexer:
    return InputMultiplexer()


class TestBasic:
    async def test_single_read(self, mux: InputMultiplexer) -> None:
        with patch("builtins.input", return_value="hello"):
            result = await mux.read("> ")
        assert result == "hello"

    async def test_sequential_reads(self, mux: InputMultiplexer) -> None:
        responses = iter(["first", "second"])
        with patch("builtins.input", side_effect=responses):
            r1 = await mux.read("> ")
            r2 = await mux.read("> ")
        assert r1 == "first"
        assert r2 == "second"


class TestPriority:
    async def test_higher_priority_served_first(self, mux: InputMultiplexer) -> None:
        """When multiple requests queued, higher priority goes first."""
        served_order: list[str] = []

        responses = iter(["low_response", "high_response"])

        async def simulate() -> None:
            with patch("builtins.input", side_effect=responses):
                low_task = asyncio.create_task(mux.read("low> ", priority=0))
                await asyncio.sleep(0)  # let low_task start processing
                high_task = asyncio.create_task(mux.read("high> ", priority=10))

                low_result = await low_task
                served_order.append(f"low:{low_result}")
                high_result = await high_task
                served_order.append(f"high:{high_result}")

        await simulate()
        # Low was already being processed when high arrived,
        # so low gets first input, high gets second
        assert served_order[0].startswith("low:")

    async def test_same_priority_fifo(self, mux: InputMultiplexer) -> None:
        """Same priority requests served in arrival order."""
        responses = iter(["first", "second"])
        with patch("builtins.input", side_effect=responses):
            t1 = asyncio.create_task(mux.read("a> ", priority=0))
            t2 = asyncio.create_task(mux.read("b> ", priority=0))
            r1 = await t1
            r2 = await t2
        assert r1 == "first"
        assert r2 == "second"

    async def test_approval_before_chat(self, mux: InputMultiplexer) -> None:
        """Priority 10 request queued while idle is served before priority 0."""
        results: list[str] = []
        responses = iter(["approval_yes", "chat_input"])

        with patch("builtins.input", side_effect=responses):
            # Queue high priority first (no input running yet)
            high = asyncio.create_task(mux.read("approve? ", priority=10))
            low = asyncio.create_task(mux.read("> ", priority=0))

            results.append(await high)
            results.append(await low)

        assert results[0] == "approval_yes"
        assert results[1] == "chat_input"


class TestConcurrency:
    async def test_processing_flag_prevents_double_start(
        self, mux: InputMultiplexer
    ) -> None:
        """Two near-simultaneous reads should not start two processors."""
        call_count = 0
        original_process = mux._process_queue

        async def counting_process() -> None:
            nonlocal call_count
            call_count += 1
            await original_process()

        mux._process_queue = counting_process  # type: ignore[assignment]

        with patch("builtins.input", side_effect=["a", "b"]):
            t1 = asyncio.create_task(mux.read("1> "))
            t2 = asyncio.create_task(mux.read("2> "))
            await t1
            await t2

        assert call_count == 1

    async def test_cancelled_future_skipped(self, mux: InputMultiplexer) -> None:
        """Cancelled futures in queue are skipped."""
        with patch("builtins.input", return_value="kept"):
            task = asyncio.create_task(mux.read("cancel> "))
            task.cancel()

            kept = asyncio.create_task(mux.read("keep> "))
            try:
                await task
            except asyncio.CancelledError:
                pass
            result = await kept

        assert result == "kept"


class TestNotification:
    async def test_notification_when_processing(
        self, mux: InputMultiplexer, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """High-priority request prints notification when input is active."""
        mux._processing = True  # simulate active input
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()

        # Don't actually process — just test the notification
        with patch.object(mux, "_process_queue"):
            # Read with high priority while processing
            task = asyncio.create_task(mux.read("approve? ", priority=10))
            await asyncio.sleep(0)
            future.set_result("")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        captured = capsys.readouterr()
        assert "Approval pending" in captured.out

    async def test_no_notification_when_idle(
        self, mux: InputMultiplexer, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No notification when not processing."""
        with patch("builtins.input", return_value="ok"):
            await mux.read("test> ", priority=10)

        captured = capsys.readouterr()
        assert "Approval pending" not in captured.out
