"""Tests for pdf_parser tool."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from agent_harness.core.config import PdfConfig
from agent_harness.tool.builtin.pdf_parser import pdf_parser


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
            "agent_harness.tool.builtin.pdf_parser.resolve_pdf_config",
            return_value=fake_cfg,
        ):
            result = await pdf_parser.execute(url="https://example.com/doc.pdf")
        assert "Unknown PDF provider" in result

    @pytest.mark.asyncio
    async def test_paddleocr_no_api_key_returns_error(self) -> None:
        fake_cfg = PdfConfig(provider="paddleocr", paddleocr_api_key=None)
        with patch(
            "agent_harness.tool.builtin.pdf_parser.resolve_pdf_config",
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
