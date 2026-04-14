"""Background task management tool."""
from agent_app.tools.background.background_task import background_task
from agent_harness.tool.base import BaseTool

BACKGROUND_TOOLS: list[BaseTool] = [background_task]

__all__ = ["BACKGROUND_TOOLS", "background_task"]
