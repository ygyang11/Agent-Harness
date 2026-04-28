"""Tests for HTTP retry helpers."""
from __future__ import annotations

import sys

import pytest

from agent_harness.utils import http_retry as http_retry_module
from agent_harness.utils.http_retry import HttpTextResponse


class _FakeTimeout:
    def __init__(self, total: int) -> None:
        self.total = total


class _FakeResponse:
    def __init__(self, status: int, body: str, headers: dict[str, str]) -> None:
        self.status = status
        self._body = body
        self.headers = headers

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    async def text(self) -> str:
        return self._body


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def request(self, method: str, url: str, **kwargs: object) -> _FakeResponse:
        _ = (method, url, kwargs)
        return self._response


class _FakeAiohttpModule:
    class ClientError(Exception):
        pass

    ClientTimeout = _FakeTimeout

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def ClientSession(self) -> _FakeSession:  # noqa: N802
        return _FakeSession(self._response)


class TestHttpTextRetry:
    @pytest.mark.asyncio
    async def test_http_get_text_with_retry_returns_headers_and_body(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setitem(
            sys.modules,
            "aiohttp",
            _FakeAiohttpModule(
                _FakeResponse(
                    status=200,
                    body="hello",
                    headers={"Content-Type": "text/plain", "X-Test": "1"},
                )
            ),
        )

        response = await http_retry_module.http_get_text_with_retry("https://example.com")
        assert response == HttpTextResponse(
            status=200,
            headers={"Content-Type": "text/plain", "X-Test": "1"},
            body="hello",
        )

    @pytest.mark.asyncio
    async def test_http_get_with_retry_keeps_legacy_tuple_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def _fake_http_get_text_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: object = None,
        ) -> HttpTextResponse:
            _ = (url, headers, timeout, retry)
            return HttpTextResponse(status=204, headers={"X-Test": "1"}, body="")

        monkeypatch.setattr(
            http_retry_module,
            "http_get_text_with_retry",
            _fake_http_get_text_with_retry,
        )

        status, body = await http_retry_module.http_get_with_retry("https://example.com")
        assert (status, body) == (204, "")
