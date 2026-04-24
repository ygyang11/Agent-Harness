"""Tests for retry semantics in BaseLLM (rate limiter + on_retry hook)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent_harness.core.config import LLMConfig
from agent_harness.core.errors import LLMConnectionError, LLMRateLimitError
from agent_harness.core.message import Message, MessageChunk
from agent_harness.llm.base import BaseLLM, RateLimiter
from agent_harness.llm.types import (
    FinishReason,
    LLMResponse,
    LLMRetryInfo,
    StreamDelta,
    Usage,
)
from agent_harness.tool.base import ToolSchema


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_succeeds_within_limit(self) -> None:
        limiter = RateLimiter(max_requests=10, window_seconds=60.0)
        await limiter.acquire()
        await limiter.acquire()

    @pytest.mark.asyncio
    async def test_acquire_tracks_timestamps(self) -> None:
        limiter = RateLimiter(max_requests=5, window_seconds=60.0)
        for _ in range(3):
            await limiter.acquire()
        assert len(limiter._timestamps) == 3


def _config(max_retries: int = 3, retry_delay: float = 0.0) -> LLMConfig:
    return LLMConfig(
        provider="anthropic",
        model="test",
        api_key="key",
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


class _StubLLM(BaseLLM):
    """Minimal stub: stream/generate driven by pre-set scripts."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        stream_script: list[Any] | None = None,
        generate_script: list[Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._stream_script = list(stream_script or [])
        self._generate_script = list(generate_script or [])

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        item = self._generate_script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamDelta]:
        item = self._stream_script.pop(0)
        if isinstance(item, Exception):
            raise item
        for delta in item:
            yield delta


def _delta(content: str | None = None, finish: FinishReason | None = None) -> StreamDelta:
    return StreamDelta(
        chunk=MessageChunk(delta_content=content),
        finish_reason=finish,
    )


class TestStreamWithEventsRetry:
    @pytest.mark.asyncio
    async def test_on_retry_fires_once_with_stream_kind(self) -> None:
        success_stream = [_delta("hi"), _delta(finish=FinishReason.STOP)]
        llm = _StubLLM(
            _config(),
            stream_script=[LLMConnectionError("boom"), success_stream],
        )
        captured: list[LLMRetryInfo] = []

        async def on_retry(info: LLMRetryInfo) -> None:
            captured.append(info)

        response = await llm.stream_with_events(
            [Message.user("hi")], on_retry=on_retry,
        )

        assert response.message.content == "hi"
        assert len(captured) == 1
        assert captured[0].kind == "stream"
        assert captured[0].attempt == 1
        assert captured[0].max_retries == 3
        assert isinstance(captured[0].error, LLMConnectionError)

    @pytest.mark.asyncio
    async def test_rate_limit_retry_honors_retry_after(self) -> None:
        rate_limit = LLMRateLimitError("slow down")
        rate_limit.retry_after = 5.0
        success_stream = [_delta("ok"), _delta(finish=FinishReason.STOP)]
        llm = _StubLLM(
            _config(retry_delay=0.0),
            stream_script=[rate_limit, success_stream],
        )
        captured: list[LLMRetryInfo] = []

        async def on_retry(info: LLMRetryInfo) -> None:
            captured.append(info)
            # Abort the actual sleep by raising something that is caught and logged
            raise RuntimeError("skip-sleep-by-raising-inside-hook")

        # With hook isolation, the hook's raise is swallowed; sleep still runs.
        # Use retry_delay=0.0 above so retry_after path still dominates wait calc.
        # We only assert wait comes from retry_after path (5.0) here.
        # The actual sleep is awaited via asyncio.sleep (0 delay would mean short).
        # To keep the test fast we monkeypatch asyncio.sleep externally in the
        # next test; here we tolerate the tiny wait if exponent overwhelmed.
        # Focus: captured[0].wait should be >= 5.0.
        import asyncio as _asyncio

        original_sleep = _asyncio.sleep

        async def _fast_sleep(_seconds: float) -> None:
            await original_sleep(0)

        _asyncio.sleep = _fast_sleep  # type: ignore[assignment]
        try:
            response = await llm.stream_with_events(
                [Message.user("hi")], on_retry=on_retry,
            )
        finally:
            _asyncio.sleep = original_sleep  # type: ignore[assignment]

        assert response.message.content == "ok"
        assert len(captured) == 1
        assert captured[0].wait >= 5.0

    @pytest.mark.asyncio
    async def test_on_retry_exception_does_not_break_retry_loop(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging as _logging

        success_stream = [_delta("ok"), _delta(finish=FinishReason.STOP)]
        llm = _StubLLM(
            _config(),
            stream_script=[LLMConnectionError("boom"), success_stream],
        )

        async def bad_on_retry(_info: LLMRetryInfo) -> None:
            raise RuntimeError("hook is broken")

        # agent_harness logger sets propagate=False in setup_logging; attach a
        # dedicated handler so caplog can capture.
        target = _logging.getLogger("agent_harness.llm.base")
        target.addHandler(caplog.handler)
        try:
            with caplog.at_level("ERROR", logger="agent_harness.llm.base"):
                response = await llm.stream_with_events(
                    [Message.user("hi")], on_retry=bad_on_retry,
                )
        finally:
            target.removeHandler(caplog.handler)

        assert response.message.content == "ok"
        assert any(
            "on_retry callback failed" in r.message for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_no_on_retry_when_attempt_succeeds_first(self) -> None:
        success_stream = [_delta("ok"), _delta(finish=FinishReason.STOP)]
        llm = _StubLLM(_config(), stream_script=[success_stream])
        captured: list[LLMRetryInfo] = []

        async def on_retry(info: LLMRetryInfo) -> None:
            captured.append(info)

        await llm.stream_with_events([Message.user("hi")], on_retry=on_retry)

        assert captured == []

    @pytest.mark.asyncio
    async def test_no_on_retry_on_final_exhausted_attempt(self) -> None:
        llm = _StubLLM(
            _config(max_retries=2),
            stream_script=[
                LLMConnectionError("1"),
                LLMConnectionError("2"),
                LLMConnectionError("3"),
            ],
        )
        captured: list[LLMRetryInfo] = []

        async def on_retry(info: LLMRetryInfo) -> None:
            captured.append(info)

        with pytest.raises(LLMConnectionError):
            await llm.stream_with_events([Message.user("hi")], on_retry=on_retry)

        # Hook fires before sleep for attempts 0 and 1 (two retries), not for the
        # final exhausted attempt (which re-raises).
        assert len(captured) == 2
        assert [c.attempt for c in captured] == [1, 2]


class TestGenerateWithEventsRetry:
    @pytest.mark.asyncio
    async def test_on_retry_fires_with_generate_kind(self) -> None:
        success = LLMResponse(
            message=Message.assistant("ok"),
            usage=Usage(),
            finish_reason=FinishReason.STOP,
            model="test",
        )
        llm = _StubLLM(
            _config(),
            generate_script=[LLMConnectionError("boom"), success],
        )
        captured: list[LLMRetryInfo] = []

        async def on_retry(info: LLMRetryInfo) -> None:
            captured.append(info)

        response = await llm.generate_with_events(
            [Message.user("hi")], on_retry=on_retry,
        )

        assert response.message.content == "ok"
        assert len(captured) == 1
        assert captured[0].kind == "generate"
        assert captured[0].attempt == 1

    @pytest.mark.asyncio
    async def test_on_retry_exception_isolation_in_generate(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging as _logging

        success = LLMResponse(
            message=Message.assistant("ok"),
            usage=Usage(),
            finish_reason=FinishReason.STOP,
            model="test",
        )
        llm = _StubLLM(
            _config(),
            generate_script=[LLMConnectionError("boom"), success],
        )

        async def bad_on_retry(_info: LLMRetryInfo) -> None:
            raise RuntimeError("nope")

        target = _logging.getLogger("agent_harness.llm.base")
        target.addHandler(caplog.handler)
        try:
            with caplog.at_level("ERROR", logger="agent_harness.llm.base"):
                response = await llm.generate_with_events(
                    [Message.user("hi")], on_retry=bad_on_retry,
                )
        finally:
            target.removeHandler(caplog.handler)

        assert response.message.content == "ok"
        assert any(
            "on_retry callback failed" in r.message for r in caplog.records
        )
