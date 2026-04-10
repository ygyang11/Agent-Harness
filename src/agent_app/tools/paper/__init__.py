"""Academic paper tools — search and fetch papers."""

from agent_app.tools.paper.paper_fetch import paper_fetch
from agent_app.tools.paper.paper_search import paper_search
from agent_harness.tool.base import BaseTool

PAPER_TOOLS: list[BaseTool] = [paper_search, paper_fetch]

__all__ = ["PAPER_TOOLS", "paper_fetch", "paper_search"]
