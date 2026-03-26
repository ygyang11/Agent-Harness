"""PDF parsing tool for extracting text from PDF documents."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from agent_harness.core.config import resolve_pdf_config
from agent_harness.tool.decorator import tool
from agent_harness.utils.http_retry import (
    HttpRetryConfig,
    http_get_bytes_with_retry,
    http_get_with_retry,
    http_post_json_with_retry,
)
from agent_harness.utils.token_counter import truncate_text_by_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfParserConfig:
    """Configuration for the pdf_parser tool."""

    max_output_tokens: int = 15_000
    poll_interval: float = 3.0
    max_poll_attempts: int = 100
    request_max_attempts: int = 3
    request_base_delay: float = 1.0


_CFG = PdfParserConfig()

_MINERU_BASE = "https://mineru.net"
_PADDLEOCR_JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"


def _retry_policy() -> HttpRetryConfig:
    return HttpRetryConfig(
        max_attempts=max(1, _CFG.request_max_attempts),
        base_delay=_CFG.request_base_delay,
    )


async def _get_json_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> tuple[int, dict[str, object] | None, str]:
    status, body = await http_get_with_retry(
        url,
        headers=headers,
        timeout=timeout,
        retry=_retry_policy(),
    )
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = None
    return status, data, body


async def _post_json_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: object | None = None,
    timeout: int = 30,
) -> tuple[int, dict[str, object] | None, str]:
    status, body = await http_post_json_with_retry(
        url,
        headers=headers,
        json_body=json_body,
        timeout=timeout,
        retry=_retry_policy(),
    )
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = None
    return status, data, body


# ---------------------------------------------------------------------------
# MinerU providers
# ---------------------------------------------------------------------------


async def _parse_mineru(url: str, api_key: str) -> str:
    """Parse PDF via MinerU precise API."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        status, result, body = await _post_json_with_retry(
            f"{_MINERU_BASE}/api/v4/extract/task",
            headers=headers,
            json_body={
                "url": url,
                "model_version": "vlm",
                "enable_formula": True,
                "enable_table": True,
            },
        )
    except Exception as exc:
        return f"Error: PDF parsing task submission failed: {exc}"
    if status != 200:
        return f"Error: PDF parsing task submission failed (HTTP {status}): {body[:200]}"
    if result is None:
        return "Error: PDF parsing task submission returned invalid JSON"

    if result.get("code") != 0:
        return f"Error: PDF parsing task submission failed: {result.get('msg', 'unknown error')}"
    data_obj = result.get("data")
    if not isinstance(data_obj, dict):
        return "Error: PDF parsing service returned invalid submission payload"
    task_id_obj = data_obj.get("task_id")
    task_id = task_id_obj if isinstance(task_id_obj, str) else ""
    if not task_id:
        return "Error: PDF parsing service returned no task ID"

    for _ in range(_CFG.max_poll_attempts):
        await asyncio.sleep(_CFG.poll_interval)
        try:
            status, result, _body = await _get_json_with_retry(
                f"{_MINERU_BASE}/api/v4/extract/task/{task_id}",
                headers=headers,
            )
        except Exception as exc:
            return f"Error: PDF parsing status polling failed: {exc}"
        if status != 200:
            return f"Error: PDF parsing status polling failed (HTTP {status})"
        if result is None:
            return "Error: PDF parsing status polling returned invalid JSON"
        poll_data = result.get("data")
        if not isinstance(poll_data, dict):
            return "Error: PDF parsing status polling returned invalid payload"
        state_obj = poll_data.get("state")
        state = state_obj if isinstance(state_obj, str) else ""
        if state == "done":
            zip_url_obj = poll_data.get("full_zip_url")
            zip_url = zip_url_obj if isinstance(zip_url_obj, str) else ""
            if not zip_url:
                return "Error: PDF parsing service returned no result URL"
            return await _download_mineru_markdown(zip_url)
        if state == "failed":
            err_obj = poll_data.get("err_msg")
            err = err_obj if isinstance(err_obj, str) and err_obj else "unknown error"
            return f"Error: PDF parsing failed: {err}"

    return "Error: PDF parsing timed out"


_MAX_ZIP_ENTRY_SIZE = 50 * 1024 * 1024  # 50 MB


async def _download_mineru_markdown(zip_url: str) -> str:
    """Download ZIP from MinerU and extract the markdown content."""
    import io  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    try:
        status, data = await http_get_bytes_with_retry(
            zip_url,
            retry=_retry_policy(),
        )
    except Exception as exc:
        return f"Error: failed to download PDF parsing result: {exc}"
    if status != 200:
        return f"Error: failed to download PDF parsing result (HTTP {status})"

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            md_files: list[str] = []
            for name in zf.namelist():
                if ".." in name or name.startswith("/"):
                    logger.warning("Skipping suspicious ZIP entry: %s", name)
                    continue
                info = zf.getinfo(name)
                if info.file_size > _MAX_ZIP_ENTRY_SIZE:
                    logger.warning(
                        "Skipping oversized ZIP entry: %s (%d bytes)",
                        name, info.file_size,
                    )
                    continue
                if name.endswith(".md"):
                    md_files.append(name)

            if not md_files:
                return "Error: no markdown file found in PDF parsing result"
            target = next((n for n in md_files if "full.md" in n), md_files[0])
            return zf.read(target).decode("utf-8", errors="replace")
    except zipfile.BadZipFile:
        return "Error: PDF parsing service returned invalid result"


async def _parse_mineru_lightweight(url: str) -> str:
    """Parse PDF via MinerU lightweight agent API (no token required)."""
    try:
        status, result, body = await _post_json_with_retry(
            f"{_MINERU_BASE}/api/v1/agent/parse/url",
            json_body={"url": url},
        )
    except Exception as exc:
        return f"Error: PDF parsing task submission failed: {exc}"
    if status != 200:
        return f"Error: PDF parsing task submission failed (HTTP {status}): {body[:200]}"
    if result is None:
        return "Error: PDF parsing task submission returned invalid JSON"

    data_obj = result.get("data")
    if not isinstance(data_obj, dict):
        return "Error: PDF parsing service returned invalid submission payload"
    task_id_obj = data_obj.get("task_id")
    task_id = task_id_obj if isinstance(task_id_obj, str) else ""
    if not task_id:
        msg_obj = result.get("msg")
        msg = msg_obj if isinstance(msg_obj, str) else ""
        return f"Error: PDF parsing task submission failed: {msg}"

    for _ in range(_CFG.max_poll_attempts):
        await asyncio.sleep(_CFG.poll_interval)
        try:
            status, result, _body = await _get_json_with_retry(
                f"{_MINERU_BASE}/api/v1/agent/parse/{task_id}",
            )
        except Exception as exc:
            return f"Error: PDF parsing status polling failed: {exc}"
        if status != 200:
            return f"Error: PDF parsing status polling failed (HTTP {status})"
        if result is None:
            return "Error: PDF parsing status polling returned invalid JSON"
        poll_data = result.get("data")
        if not isinstance(poll_data, dict):
            return "Error: PDF parsing status polling returned invalid payload"
        state_obj = poll_data.get("state")
        state = state_obj if isinstance(state_obj, str) else ""
        if state == "done":
            md_url_obj = poll_data.get("markdown_url")
            md_url = md_url_obj if isinstance(md_url_obj, str) else ""
            if not md_url:
                return "Error: PDF parsing service returned no result URL"
            try:
                md_status, md_body = await http_get_with_retry(
                    md_url,
                    retry=_retry_policy(),
                )
            except Exception as exc:
                return f"Error: failed to download PDF parsing result: {exc}"
            if md_status != 200:
                return f"Error: failed to download PDF parsing result (HTTP {md_status})"
            return md_body
        if state == "failed":
            err_obj = poll_data.get("err_msg")
            err = err_obj if isinstance(err_obj, str) and err_obj else "unknown error"
            return f"Error: PDF parsing failed: {err}"

    return "Error: PDF parsing timed out"


# ---------------------------------------------------------------------------
# PaddleOCR provider
# ---------------------------------------------------------------------------


async def _parse_paddleocr_with_model(url: str, api_key: str, model: str) -> str:
    """Parse PDF via PaddleOCR async API with the specified model."""
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

    try:
        status, result, body = await _post_json_with_retry(
            _PADDLEOCR_JOB_URL,
            headers=headers,
            json_body=payload,
        )
    except Exception as exc:
        return f"Error: PDF parsing task submission failed: {exc}"
    if status == 429:
        return "Error: rate limit exceeded"
    if status != 200:
        return f"Error: PDF parsing task submission failed (HTTP {status}): {body[:200]}"
    if result is None:
        return "Error: PDF parsing task submission returned invalid JSON"

    data_obj = result.get("data")
    if not isinstance(data_obj, dict):
        return "Error: PDF parsing service returned invalid submission payload"
    job_id_obj = data_obj.get("jobId")
    job_id = job_id_obj if isinstance(job_id_obj, str) else ""
    if not job_id:
        return "Error: PDF parsing service returned no job ID"

    for _ in range(_CFG.max_poll_attempts):
        await asyncio.sleep(_CFG.poll_interval)
        try:
            status, result, _body = await _get_json_with_retry(
                f"{_PADDLEOCR_JOB_URL}/{job_id}",
                headers={"Authorization": f"bearer {api_key}"},
            )
        except Exception as exc:
            return f"Error: PDF parsing status polling failed: {exc}"
        if status == 429:
            return "Error: rate limit exceeded"
        if status != 200:
            return f"Error: PDF parsing status polling failed (HTTP {status})"
        if result is None:
            return "Error: PDF parsing status polling returned invalid JSON"
        poll_data = result.get("data")
        if not isinstance(poll_data, dict):
            return "Error: PDF parsing status polling returned invalid payload"
        state_obj = poll_data.get("state")
        state = state_obj if isinstance(state_obj, str) else ""
        if state == "done":
            result_url_obj = poll_data.get("resultUrl")
            result_url = result_url_obj if isinstance(result_url_obj, dict) else {}
            json_url_obj = result_url.get("jsonUrl")
            json_url = json_url_obj if isinstance(json_url_obj, str) else ""
            if not json_url:
                return "Error: PDF parsing service returned no result URL"
            return await _download_paddleocr_markdown(json_url)
        if state == "failed":
            err_obj = poll_data.get("errorMsg")
            err = err_obj if isinstance(err_obj, str) and err_obj else "unknown error"
            return f"Error: PDF parsing failed: {err}"

    return "Error: PDF parsing timed out"


async def _download_paddleocr_markdown(json_url: str) -> str:
    """Download JSONL result from PaddleOCR and extract markdown text."""
    try:
        status, text = await http_get_with_retry(
            json_url,
            retry=_retry_policy(),
        )
    except Exception as exc:
        return f"Error: failed to download PDF parsing result: {exc}"
    if status != 200:
        return f"Error: failed to download PDF parsing result (HTTP {status})"

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
