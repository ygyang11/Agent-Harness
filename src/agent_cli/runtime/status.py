"""Runtime-layer status collector for /status command."""
from __future__ import annotations

from dataclasses import dataclass

from agent_cli.runtime import background
from agent_harness.agent.base import BaseAgent


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    model: str
    approval_mode: str
    tool_count: int
    skill_count: int
    message_count: int
    token_count: int
    max_tokens: int
    todo_count: int
    bg_running: int
    bg_total: int


def collect(agent: BaseAgent) -> StatusSnapshot:
    cfg = agent.context.config
    stm = agent.context.short_term_memory
    bg_all = background.get_all(agent)
    return StatusSnapshot(
        model=agent.llm.model_name,
        approval_mode=cfg.approval.mode,
        tool_count=len(agent.tools),
        skill_count=_skill_count(agent),
        message_count=len(stm._messages),
        token_count=stm.token_count,
        max_tokens=stm.max_tokens,
        todo_count=_todo_count(agent),
        bg_running=sum(1 for t in bg_all if t.status == "running"),
        bg_total=len(bg_all),
    )


def _skill_count(agent: BaseAgent) -> int:
    for tool in agent.tools:
        if tool.name == "skill_tool":
            loader = getattr(tool, "loader", None)
            if loader is None:
                return 0
            try:
                return len(loader.list_names())
            except Exception:
                return 0
    return 0


def _todo_count(agent: BaseAgent) -> int:
    if not agent.tool_registry.has("todo_write"):
        return 0
    tool = agent.tool_registry.get("todo_write")
    todos = getattr(tool, "_todos", None) or []
    return len(todos)
