"""PDF parsing tool for extracting text from PDF documents."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp

from agent_harness.core.config import resolve_pdf_config
from agent_harness.tool.decorator import tool
from agent_harness.utils.token_counter import truncate_text_by_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfParserConfig:
    """Configuration for the pdf_parser tool."""

    max_output_tokens: int = 15_000
    poll_interval: float = 3.0
    max_poll_attempts: int = 100


_CFG = PdfParserConfig()

_MINERU_BASE = "https://mineru.net"
_PADDLEOCR_JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"


# ---------------------------------------------------------------------------
# MinerU providers
# ---------------------------------------------------------------------------


async def _parse_mineru(url: str, api_key: str) -> str:
    """Parse PDF via MinerU precise API."""
    import aiohttp  # noqa: PLC0415

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_MINERU_BASE}/api/v4/extract/task",
            headers=headers,
            json={"url": url, "model_version": "vlm", "enable_formula": True, "enable_table": True},
        ) as resp:
            result = await resp.json()
            if result.get("code") != 0:
                return f"Error: PDF parsing task submission failed: {result.get('msg', 'unknown error')}"
            task_id: str = result["data"]["task_id"]

        for _ in range(_CFG.max_poll_attempts):
            await asyncio.sleep(_CFG.poll_interval)
            async with session.get(
                f"{_MINERU_BASE}/api/v4/extract/task/{task_id}",
                headers=headers,
            ) as resp:
                result = await resp.json()
                state = result.get("data", {}).get("state", "")
                if state == "done":
                    zip_url: str = result["data"].get("full_zip_url", "")
                    if not zip_url:
                        return "Error: PDF parsing service returned no result URL"
                    return await _download_mineru_markdown(session, zip_url)
                if state == "failed":
                    err = result.get("data", {}).get("err_msg", "unknown error")
                    return f"Error: PDF parsing failed: {err}"

        return "Error: PDF parsing timed out"


async def _download_mineru_markdown(session: aiohttp.ClientSession, zip_url: str) -> str:
    """Download ZIP from MinerU and extract the markdown content."""
    import io  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    async with session.get(zip_url) as resp:
        if resp.status != 200:
            return f"Error: failed to download PDF parsing result (HTTP {resp.status})"
        data = await resp.read()

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            md_files = [n for n in zf.namelist() if n.endswith(".md")]
            if not md_files:
                return "Error: no markdown file found in PDF parsing result"
            target = next((n for n in md_files if "full.md" in n), md_files[0])
            return zf.read(target).decode("utf-8", errors="replace")
    except zipfile.BadZipFile:
        return "Error: PDF parsing service returned invalid result"


async def _parse_mineru_lightweight(url: str) -> str:
    """Parse PDF via MinerU lightweight agent API (no token required)."""
    import aiohttp  # noqa: PLC0415

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{_MINERU_BASE}/api/v1/agent/parse/url",
            json={"url": url},
        ) as resp:
            result = await resp.json()
            task_id: str = result.get("data", {}).get("task_id", "")
            if not task_id:
                return f"Error: PDF parsing task submission failed: {result.get('msg', '')}"

        for _ in range(_CFG.max_poll_attempts):
            await asyncio.sleep(_CFG.poll_interval)
            async with session.get(
                f"{_MINERU_BASE}/api/v1/agent/parse/{task_id}",
            ) as resp:
                result = await resp.json()
                state = result.get("data", {}).get("state", "")
                if state == "done":
                    md_url: str = result["data"].get("markdown_url", "")
                    if not md_url:
                        return "Error: PDF parsing service returned no result URL"
                    async with session.get(md_url) as md_resp:
                        return await md_resp.text()
                if state == "failed":
                    err = result.get("data", {}).get("err_msg", "unknown error")
                    return f"Error: PDF parsing failed: {err}"

        return "Error: PDF parsing timed out"


# ---------------------------------------------------------------------------
# PaddleOCR provider
# ---------------------------------------------------------------------------


async def _parse_paddleocr_with_model(url: str, api_key: str, model: str) -> str:
    """Parse PDF via PaddleOCR async API with the specified model."""
    import aiohttp  # noqa: PLC0415

    headers = {"Authorization": f"bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "fileUrl": url,
        "model": model,
        "optionalPayload": {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        },
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            _PADDLEOCR_JOB_URL, headers=headers, json=payload,
        ) as resp:
            if resp.status == 429:
                return "Error: rate limit exceeded"
            if resp.status != 200:
                body = await resp.text()
                return f"Error: PDF parsing task submission failed (HTTP {resp.status}): {body[:200]}"
            result = await resp.json()
            job_id: str = result.get("data", {}).get("jobId", "")
            if not job_id:
                return "Error: PDF parsing service returned no job ID"

        for _ in range(_CFG.max_poll_attempts):
            await asyncio.sleep(_CFG.poll_interval)
            async with session.get(
                f"{_PADDLEOCR_JOB_URL}/{job_id}",
                headers={"Authorization": f"bearer {api_key}"},
            ) as resp:
                if resp.status == 429:
                    return "Error: rate limit exceeded"
                result = await resp.json()
                state = result.get("data", {}).get("state", "")
                if state == "done":
                    json_url: str = result["data"].get("resultUrl", {}).get("jsonUrl", "")
                    if not json_url:
                        return "Error: PDF parsing service returned no result URL"
                    return await _download_paddleocr_markdown(session, json_url)
                if state == "failed":
                    err = result.get("data", {}).get("errorMsg", "unknown error")
                    return f"Error: PDF parsing failed: {err}"

        return "Error: PDF parsing timed out"


async def _download_paddleocr_markdown(session: aiohttp.ClientSession, json_url: str) -> str:
    """Download JSONL result from PaddleOCR and extract markdown text."""
    async with session.get(json_url) as resp:
        if resp.status != 200:
            return f"Error: failed to download PDF parsing result (HTTP {resp.status})"
        text = await resp.text()

    pages: list[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        for res in data.get("result", {}).get("layoutParsingResults", []):
            md_text: str = res.get("markdown", {}).get("text", "")
            if md_text.strip():
                pages.append(md_text.strip())

    if not pages:
        return "Error: PDF contains no extractable text"
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------


@tool
async def pdf_parser(url: str) -> str:
    """Extract text from a PDF document at the given URL.

    Submits the PDF URL to a cloud parsing service and returns
    the extracted content as markdown text. Supports complex
    layouts, tables, and formulas.

    Args:
        url: URL of the PDF document to parse.

    Returns:
        Extracted text in markdown format, truncated to token budget.
        Errors are prefixed with ``Error:``.
    """
    if not url.strip():
        return "Error: URL cannot be empty"

    cfg = resolve_pdf_config(None)
    provider = cfg.provider

    if provider == "mineru":
        api_key = cfg.mineru_api_key or ""
        if api_key:
            raw = await _parse_mineru(url, api_key)
            if raw.startswith("Error:"):
                logger.warning("MinerU precise API failed, falling back to lightweight: %s", raw)
                raw = await _parse_mineru_lightweight(url)
        else:
            raw = await _parse_mineru_lightweight(url)
    elif provider == "paddleocr":
        api_key = cfg.paddleocr_api_key or ""
        if not api_key:
            return (
                "PDF parsing not configured: PADDLEOCR_API_KEY not set. "
                "Set the environment variable or configure in config.yaml."
            )
        raw = await _parse_paddleocr_with_model(url, api_key, "PaddleOCR-VL-1.5")
        if raw.startswith("Error:"):
            logger.warning("PaddleOCR VL-1.5 failed, falling back to VL: %s", raw)
            raw = await _parse_paddleocr_with_model(url, api_key, "PaddleOCR-VL")
    else:
        return f"Unknown PDF provider: {provider!r}. Use 'mineru' or 'paddleocr'."

    if raw.startswith("Error:"):
        return raw

    return truncate_text_by_tokens(
        raw,
        max_tokens=_CFG.max_output_tokens,
        suffix="\n... (truncated)",
    )
