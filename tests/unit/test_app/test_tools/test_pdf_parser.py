"""Tests for pdf_parser tool."""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from agent_harness.core.config import PdfConfig
from agent_app.tools.pdf_parser import (
    _download_mineru_markdown,
    _download_paddleocr_markdown,
    _get_json_with_retry,
    _post_json_with_retry,
    pdf_parser,
)


class TestPdfParserValidation:
    @pytest.mark.asyncio
    async def test_empty_url_returns_error(self) -> None:
        result = await pdf_parser.execute(url="")
        assert result.startswith("Error:")
        assert "empty" in result

    @pytest.mark.asyncio
    async def test_whitespace_url_returns_error(self) -> None:
        result = await pdf_parser.execute(url="   ")
        assert result.startswith("Error:")
        assert "empty" in result

    @pytest.mark.asyncio
    async def test_unknown_provider_returns_error(self) -> None:
        fake_cfg = PdfConfig(provider="unknown")
        with patch(
            "agent_app.tools.pdf_parser.resolve_pdf_config",
            return_value=fake_cfg,
        ):
            result = await pdf_parser.execute(url="https://example.com/doc.pdf")
        assert "Unknown PDF provider" in result

    @pytest.mark.asyncio
    async def test_paddleocr_no_api_key_returns_error(self) -> None:
        fake_cfg = PdfConfig(provider="paddleocr", paddleocr_api_key=None)
        with patch(
            "agent_app.tools.pdf_parser.resolve_pdf_config",
            return_value=fake_cfg,
        ):
            result = await pdf_parser.execute(url="https://example.com/doc.pdf")
        assert "PADDLEOCR_API_KEY not set" in result


class TestPdfConfigDefaults:
    def test_default_provider_is_mineru(self) -> None:
        cfg = PdfConfig()
        assert cfg.provider == "mineru"

    def test_blank_api_key_becomes_none(self) -> None:
        cfg = PdfConfig(mineru_api_key="  ")
        assert cfg.mineru_api_key is None

    def test_blank_paddleocr_key_becomes_none(self) -> None:
        cfg = PdfConfig(paddleocr_api_key="")
        assert cfg.paddleocr_api_key is None


class TestPdfParserRetryWiring:
    @pytest.mark.asyncio
    async def test_get_json_with_retry_uses_retry_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pdf_parser_module = sys.modules[_get_json_with_retry.__module__]
        captured: dict[str, object] = {}

        async def _fake_http_get_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: object,
        ) -> tuple[int, str]:
            captured["url"] = url
            captured["headers"] = headers
            captured["timeout"] = timeout
            captured["retry"] = retry
            return 200, '{"ok": true}'

        monkeypatch.setattr(pdf_parser_module, "http_get_with_retry", _fake_http_get_with_retry)
        status, data, body = await _get_json_with_retry(
            "https://example.com/status",
            headers={"x-test": "1"},
            timeout=9,
        )

        assert status == 200
        assert data == {"ok": True}
        assert body == '{"ok": true}'
        assert captured["url"] == "https://example.com/status"
        assert captured["headers"] == {"x-test": "1"}
        assert captured["timeout"] == 9
        retry = captured["retry"]
        assert isinstance(retry, pdf_parser_module.HttpRetryConfig)
        assert retry.max_attempts == pdf_parser_module._CFG.request_max_attempts
        assert retry.base_delay == pdf_parser_module._CFG.request_base_delay

    @pytest.mark.asyncio
    async def test_post_json_with_retry_uses_retry_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pdf_parser_module = sys.modules[_get_json_with_retry.__module__]
        captured: dict[str, object] = {}

        async def _fake_http_post_json_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            json_body: object | None = None,
            timeout: int = 30,
            retry: object,
        ) -> tuple[int, str]:
            captured["url"] = url
            captured["headers"] = headers
            captured["json_body"] = json_body
            captured["timeout"] = timeout
            captured["retry"] = retry
            return 200, '{"task_id": "t1"}'

        monkeypatch.setattr(
            pdf_parser_module,
            "http_post_json_with_retry",
            _fake_http_post_json_with_retry,
        )
        status, data, body = await _post_json_with_retry(
            "https://example.com/submit",
            headers={"Authorization": "bearer k"},
            json_body={"url": "https://example.com/a.pdf"},
            timeout=15,
        )

        assert status == 200
        assert data == {"task_id": "t1"}
        assert body == '{"task_id": "t1"}'
        assert captured["url"] == "https://example.com/submit"
        assert captured["headers"] == {"Authorization": "bearer k"}
        assert captured["json_body"] == {"url": "https://example.com/a.pdf"}
        assert captured["timeout"] == 15
        retry = captured["retry"]
        assert isinstance(retry, pdf_parser_module.HttpRetryConfig)
        assert retry.max_attempts == pdf_parser_module._CFG.request_max_attempts
        assert retry.base_delay == pdf_parser_module._CFG.request_base_delay

    @pytest.mark.asyncio
    async def test_download_mineru_markdown_uses_retry_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pdf_parser_module = sys.modules[_get_json_with_retry.__module__]
        captured: dict[str, object] = {}

        async def _fake_http_get_bytes_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: object,
        ) -> tuple[int, bytes]:
            captured["url"] = url
            captured["headers"] = headers
            captured["timeout"] = timeout
            captured["retry"] = retry
            return 503, b"service unavailable"

        monkeypatch.setattr(
            pdf_parser_module,
            "http_get_bytes_with_retry",
            _fake_http_get_bytes_with_retry,
        )
        result = await _download_mineru_markdown("https://example.com/result.zip")

        assert result == "Error: failed to download PDF parsing result (HTTP 503)"
        assert captured["url"] == "https://example.com/result.zip"
        assert captured["headers"] is None
        assert captured["timeout"] == 30
        retry = captured["retry"]
        assert isinstance(retry, pdf_parser_module.HttpRetryConfig)
        assert retry.max_attempts == pdf_parser_module._CFG.request_max_attempts
        assert retry.base_delay == pdf_parser_module._CFG.request_base_delay

    @pytest.mark.asyncio
    async def test_download_paddleocr_markdown_uses_retry_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pdf_parser_module = sys.modules[_get_json_with_retry.__module__]
        captured: dict[str, object] = {}
        jsonl = '{"result":{"layoutParsingResults":[{"markdown":{"text":"Page 1"}}]}}\n'

        async def _fake_http_get_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: object,
        ) -> tuple[int, str]:
            captured["url"] = url
            captured["headers"] = headers
            captured["timeout"] = timeout
            captured["retry"] = retry
            return 200, jsonl

        monkeypatch.setattr(pdf_parser_module, "http_get_with_retry", _fake_http_get_with_retry)
        result = await _download_paddleocr_markdown("https://example.com/out.jsonl")

        assert result == "Page 1"
        assert captured["url"] == "https://example.com/out.jsonl"
        assert captured["headers"] is None
        assert captured["timeout"] == 30
        retry = captured["retry"]
        assert isinstance(retry, pdf_parser_module.HttpRetryConfig)
        assert retry.max_attempts == pdf_parser_module._CFG.request_max_attempts
        assert retry.base_delay == pdf_parser_module._CFG.request_base_delay
