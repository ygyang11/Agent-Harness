"""Built-in tools for the agent application."""

from agent_app.tools.filesystem import (
    FILESYSTEM_TOOLS,
    READONLY_FILESYSTEM_TOOLS,
    WRITABLE_FILESYSTEM_TOOLS,
    edit_file,
    glob_files,
    grep_files,
    list_dir,
    read_file,
    write_file,
)
from agent_app.tools.paper import PAPER_TOOLS, paper_fetch, paper_search
from agent_app.tools.pdf_parser import PDF_TOOLS, pdf_parser
from agent_app.tools.skill import SKILL_TOOLS, skill_tool
from agent_app.tools.sub_agent import SUB_AGENT_TOOLS, sub_agent
from agent_app.tools.take_notes import list_notes, read_notes, take_notes
from agent_app.tools.terminal import TERMINAL_TOOLS, terminal_tool
from agent_app.tools.todo_write import TODO_TOOLS, todo_write
from agent_app.tools.web import WEB_TOOLS, web_fetch, web_search
from agent_harness.tool.base import BaseTool

BUILTIN_TOOLS: list[BaseTool] = [
    *TERMINAL_TOOLS,
    *WEB_TOOLS,
    *PDF_TOOLS,
    *PAPER_TOOLS,
    take_notes,
    list_notes,
    read_notes,
    *SKILL_TOOLS,
    *TODO_TOOLS,
    *SUB_AGENT_TOOLS,
    *FILESYSTEM_TOOLS,
]

__all__ = [
    # Tool group constants
    "BUILTIN_TOOLS",
    "FILESYSTEM_TOOLS",
    "READONLY_FILESYSTEM_TOOLS",
    "WRITABLE_FILESYSTEM_TOOLS",
    "TERMINAL_TOOLS",
    "WEB_TOOLS",
    "PDF_TOOLS",
    "PAPER_TOOLS",
    "SKILL_TOOLS",
    "TODO_TOOLS",
    "SUB_AGENT_TOOLS",
    # Individual tools
    "terminal_tool",
    "web_fetch",
    "web_search",
    "pdf_parser",
    "paper_search",
    "paper_fetch",
    "take_notes",
    "list_notes",
    "read_notes",
    "skill_tool",
    "todo_write",
    "sub_agent",
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "glob_files",
    "grep_files",
]
