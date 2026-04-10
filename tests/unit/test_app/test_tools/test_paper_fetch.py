"""Tests for the paper_fetch builtin tool."""
from __future__ import annotations

import json
import sys

import pytest

from agent_app.tools.paper.paper_fetch import (
    _fetch_arxiv_metadata,
    _format_metadata,
    paper_fetch,
)


class TestFormatMetadata:
    def test_full_arxiv_metadata(self) -> None:
        paper = {
            "title": "Attention Is All You Need",
            "arxiv_id": "1706.03762",
            "authors": ["Vaswani", "Shazeer", "Parmar"],
            "published": "2017-06-12",
            "abstract": "The dominant sequence transduction models...",
            "categories": ["cs.CL", "cs.AI"],
            "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
            "abs_url": "https://arxiv.org/abs/1706.03762",
        }
        result = _format_metadata(paper)
        assert "# Attention Is All You Need" in result
        assert "1706.03762" in result
        assert "Vaswani" in result
        assert "## Abstract" in result
        assert "paper_fetch" in result

    def test_full_s2_metadata(self) -> None:
        paper = {
            "title": "Test Paper",
            "s2_id": "abc123",
            "doi": "10.1234/xxx",
            "authors": ["Alice"],
            "year": 2023,
            "venue": "NeurIPS",
            "citation_count": 50,
            "reference_count": 30,
            "fields_of_study": ["Computer Science"],
            "publication_types": ["Conference"],
            "abstract": "We propose...",
            "tldr": "A short summary.",
            "pdf_url": "https://example.com/paper.pdf",
        }
        result = _format_metadata(paper)
        assert "# Test Paper" in result
        assert "NeurIPS" in result
        assert "Citations" in result
        assert "References" in result
        assert "Fields of Study" in result
        assert "## TL;DR" in result

    def test_minimal_metadata(self) -> None:
        paper = {"title": "Test"}
        result = _format_metadata(paper)
        assert "# Test" in result

    def test_metadata_actionable_guidance(self) -> None:
        paper = {"title": "X", "arxiv_id": "2301.07041"}
        result = _format_metadata(paper)
        assert 'mode="full"' in result
        assert "paper_fetch" in result

    def test_metadata_no_guidance_without_id(self) -> None:
        paper = {"title": "X"}
        result = _format_metadata(paper)
        assert "paper_fetch" not in result

    def test_publication_date_preferred_over_year(self) -> None:
        paper = {"title": "X", "publication_date": "2023-06-15", "year": 2023}
        result = _format_metadata(paper)
        assert "2023-06-15" in result
        assert "Year" not in result


class TestPaperFetchTool:
    async def test_empty_id(self) -> None:
        result = await paper_fetch.execute(paper_id="")
        assert "Error" in result

    async def test_unknown_mode(self) -> None:
        result = await paper_fetch.execute(paper_id="test", mode="unknown")
        assert "Error" in result
        assert "metadata" in result

    async def test_unknown_source(self) -> None:
        result = await paper_fetch.execute(
            paper_id="test", mode="metadata", source="unknown"
        )
        assert "Error" in result

    async def test_unknown_source_full_mode(self) -> None:
        result = await paper_fetch.execute(
            paper_id="test", mode="full", source="unknown"
        )
        assert "Error" in result
        assert "semantic_scholar" in result

    def test_schema_params(self) -> None:
        schema = paper_fetch.get_schema()
        assert schema.name == "paper_fetch"
        assert (
            schema.description
            == "Fetch a specific paper by ID, returning full text or metadata."
        )
        props = schema.parameters["properties"]
        assert "paper_id" in props
        assert "mode" in props
        assert "source" in props
        assert props["mode"]["enum"] == ["metadata", "full"]
        assert props["source"]["enum"] == ["arxiv", "semantic_scholar"]


class _FakeTimeout:
    def __init__(self, total: int) -> None:
        self.total = total


class _FakeResponse:
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    async def text(self) -> str:
        return self._body

    async def read(self) -> bytes:
        return self._body.encode("utf-8")


class _FakeSession:
    def __init__(
        self,
        statuses: list[int],
        calls: list[int],
        bodies: list[str] | None = None,
    ) -> None:
        self._statuses = statuses
        self._calls = calls
        self._bodies = bodies or []

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def request(
        self,
        method: str,
        url: str,
        **kwargs: object,
    ) -> _FakeResponse:
        _ = (method, url, kwargs)
        self._calls.append(1)
        idx = min(len(self._calls) - 1, len(self._statuses) - 1)
        status = self._statuses[idx]
        body = self._bodies[min(idx, len(self._bodies) - 1)] if self._bodies else f"status={status}"
        return _FakeResponse(status, body)


class _FakeAiohttpModule:
    class ClientError(Exception):
        pass

    ClientTimeout = _FakeTimeout

    def __init__(
        self,
        statuses: list[int],
        calls: list[int],
        bodies: list[str] | None = None,
    ) -> None:
        self._statuses = statuses
        self._calls = calls
        self._bodies = bodies

    def ClientSession(self) -> _FakeSession:  # noqa: N802
        return _FakeSession(self._statuses, self._calls, self._bodies)


class TestFetchArxivMetadata:
    @pytest.mark.asyncio
    async def test_fetch_xml_exception_returns_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_fetch_module = sys.modules[_fetch_arxiv_metadata.__module__]
        paper_search_module = sys.modules["agent_app.tools.paper.paper_search"]

        async def _fail_fetch_xml(url: str) -> object:
            _ = url
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(paper_search_module, "_fetch_xml", _fail_fetch_xml)
        result = await paper_fetch_module._fetch_arxiv_metadata("2301.07041")
        assert result.startswith("Error: arXiv request failed:")

    @pytest.mark.asyncio
    async def test_fetch_xml_exception_does_not_duplicate_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_fetch_module = sys.modules[_fetch_arxiv_metadata.__module__]
        paper_search_module = sys.modules["agent_app.tools.paper.paper_search"]

        async def _fail_fetch_xml(url: str) -> object:
            _ = url
            raise RuntimeError("arXiv request failed: simulated failure")

        monkeypatch.setattr(paper_search_module, "_fetch_xml", _fail_fetch_xml)
        result = await paper_fetch_module._fetch_arxiv_metadata("2301.07041")
        assert result == "Error: arXiv request failed: simulated failure"


class TestPaperFetchSemanticScholarParsing:
    @pytest.mark.asyncio
    async def test_metadata_invalid_json_returns_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_fetch_module = sys.modules[_fetch_arxiv_metadata.__module__]
        call_markers: list[int] = []
        fake_aiohttp = _FakeAiohttpModule([200], call_markers, bodies=["not-json"])
        monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
        monkeypatch.setattr(
            paper_fetch_module,
            "_CFG",
            paper_fetch_module.PaperFetchConfig(max_full_tokens=15_000, html_fetch_timeout=30),
        )

        result = await paper_fetch.execute(
            paper_id="DOI:10.1000/test",
            mode="metadata",
            source="semantic_scholar",
        )
        assert result.startswith("Error: failed to parse Semantic Scholar response:")

    @pytest.mark.asyncio
    async def test_metadata_non_object_json_returns_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_fetch_module = sys.modules[_fetch_arxiv_metadata.__module__]
        call_markers: list[int] = []
        fake_aiohttp = _FakeAiohttpModule(
            [200], call_markers, bodies=[json.dumps([1, 2, 3])]
        )
        monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
        monkeypatch.setattr(
            paper_fetch_module,
            "_CFG",
            paper_fetch_module.PaperFetchConfig(max_full_tokens=15_000, html_fetch_timeout=30),
        )

        result = await paper_fetch.execute(
            paper_id="DOI:10.1000/test",
            mode="metadata",
            source="semantic_scholar",
        )
        assert result == "Error: unexpected Semantic Scholar response format"


class TestPaperFetchFullPathRetries:
    @pytest.mark.asyncio
    async def test_try_arxiv_html_returns_none_on_retry_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_fetch_module = sys.modules[_fetch_arxiv_metadata.__module__]

        async def _fake_http_get_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: object = None,
        ) -> tuple[int, str]:
            _ = (headers, timeout, retry)
            assert "arxiv.org/html/" in url
            raise RuntimeError("transient failure")

        monkeypatch.setattr(paper_fetch_module, "http_get_with_retry", _fake_http_get_with_retry)
        result = await paper_fetch_module._try_arxiv_html("2301.07041")
        assert result is None

    @pytest.mark.asyncio
    async def test_try_unpaywall_uses_retry_result_and_parses_json(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_fetch_module = sys.modules[_fetch_arxiv_metadata.__module__]

        async def _fake_http_get_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: object = None,
        ) -> tuple[int, str]:
            _ = (headers, timeout, retry)
            assert "api.unpaywall.org" in url
            return 200, json.dumps({"best_oa_location": {"url_for_pdf": "https://oa.example/p.pdf"}})

        monkeypatch.setattr(paper_fetch_module, "http_get_with_retry", _fake_http_get_with_retry)
        result = await paper_fetch_module._try_unpaywall("10.1000/test")
        assert result == "https://oa.example/p.pdf"

    @pytest.mark.asyncio
    async def test_try_unpaywall_invalid_json_returns_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paper_fetch_module = sys.modules[_fetch_arxiv_metadata.__module__]

        async def _fake_http_get_with_retry(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            timeout: int = 30,
            retry: object = None,
        ) -> tuple[int, str]:
            _ = (url, headers, timeout, retry)
            return 200, "not-json"

        monkeypatch.setattr(paper_fetch_module, "http_get_with_retry", _fake_http_get_with_retry)
        result = await paper_fetch_module._try_unpaywall("10.1000/test")
        assert result.startswith("Error: failed to parse Unpaywall response for DOI 10.1000/test:")
