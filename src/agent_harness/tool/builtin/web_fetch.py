"""Web content fetching tool with automatic HTML text extraction."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

from agent_harness.tool.decorator import tool
from agent_harness.utils.token_counter import truncate_text_by_tokens


@dataclass(frozen=True)
class WebFetchConfig:
    """Configuration for the web_fetch tool."""

    max_response_tokens: int = 15_000
    default_timeout: int = 30
    allowed_schemes: frozenset[str] = frozenset({"http", "https"})


_CFG = WebFetchConfig()


def _validate_url(url: str) -> None:
    if not url.strip():
        raise ValueError("URL cannot be empty")
    parsed = urlparse(url)
    if parsed.scheme not in _CFG.allowed_schemes:
        raise ValueError(
            f"unsupported URL scheme: {parsed.scheme!r} "
            f"(allowed: {', '.join(sorted(_CFG.allowed_schemes))})"
        )
    if not parsed.netloc:
        raise ValueError("invalid URL: missing host")


class _TextExtractor(HTMLParser):
    """Extract visible text from HTML, skipping script/style blocks."""

    _SKIP_TAGS: frozenset[str] = frozenset({"script", "style", "noscript"})
    _BLOCK_TAGS: frozenset[str] = frozenset({
        "br", "p", "div", "li", "tr",
        "h1", "h2", "h3", "h4", "h5", "h6",
    })

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag.lower() in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        lines = (line.strip() for line in raw.splitlines())
        return "\n".join(line for line in lines if line)


def _extract_text_from_html(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.get_text()


def _format_response(body: str, content_type: str) -> str:
    ct = content_type.lower()
    if "application/json" in ct:
        try:
            parsed = json.loads(body)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            return body
    if "text/html" in ct:
        return _extract_text_from_html(body)
    return body


@tool
async def web_fetch(url: str, timeout: int = 30) -> str:
    """Fetch content from a URL and return readable text.

    Fetches the given URL via GET. HTML pages are converted to plain
    text automatically. JSON responses are pretty-printed. Only
    http and https URLs are allowed.

    Args:
        url: The URL to fetch (http or https only).
        timeout: Maximum request time in seconds (positive integer, default 30).

    Returns:
        Page content as readable text, truncated to token budget.
        Errors are prefixed with ``Error:``.
    """
    if not url.strip():
        return "Error: URL cannot be empty"

    if timeout <= 0:
        return "Error: timeout must be greater than 0"

    try:
        _validate_url(url)
    except ValueError as exc:
        return f"Error: {exc}"

    try:
        import aiohttp  # noqa: PLC0415
    except ImportError:
        return "Error: aiohttp is not installed. Run `pip install aiohttp`."

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.get(url) as resp:
                if resp.status >= 400:
                    return f"Error: HTTP {resp.status} for {url}"
                body = await resp.text()
                content_type = resp.headers.get("Content-Type", "")
                formatted = _format_response(body, content_type)
                return truncate_text_by_tokens(
                    formatted,
                    max_tokens=_CFG.max_response_tokens,
                    suffix="\n... (truncated)",
                )
    except asyncio.TimeoutError:
        return f"Error: request timed out after {timeout}s"
    except aiohttp.ClientError as exc:
        return f"Error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"
