"""Skill tool — loads domain knowledge from SKILL.md files on demand."""

from agent_app.tools.skill.skill_tool import skill_tool
from agent_harness.tool.base import BaseTool

SKILL_TOOLS: list[BaseTool] = [skill_tool]

__all__ = ["SKILL_TOOLS", "skill_tool"]
