"""Web tools — search and fetch web content."""

from agent_app.tools.web.web_fetch import web_fetch
from agent_app.tools.web.web_search import web_search
from agent_harness.tool.base import BaseTool

WEB_TOOLS: list[BaseTool] = [web_fetch, web_search]

__all__ = ["WEB_TOOLS", "web_fetch", "web_search"]
