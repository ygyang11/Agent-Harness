"""Data models for plan-and-execute orchestration."""
from __future__ import annotations

from pydantic import BaseModel, Field

from agent_harness.utils.token_counter import truncate_text_by_tokens

_PLAN_RESULT_MAX_TOKENS = 26


class PlanStep(BaseModel):
    """A single step in the plan."""

    id: str
    description: str
    status: str = "pending"
    result: str | None = None


class Plan(BaseModel):
    """A structured execution plan."""

    goal: str = ""
    steps: list[PlanStep] = Field(default_factory=list)

    @property
    def current_step(self) -> PlanStep | None:
        """Return the next pending or in-progress step."""
        for step in self.steps:
            if step.status in ("pending", "in_progress"):
                return step
        return None

    @property
    def is_complete(self) -> bool:
        """True when all steps are done or failed."""
        return (
            all(s.status in ("done", "failed") for s in self.steps)
            and len(self.steps) > 0
        )

    @property
    def progress_summary(self) -> str:
        """Compact progress summary with truncated step results."""
        lines = [f"Goal: {self.goal}"]
        for s in self.steps:
            marker = {
                "pending": "○",
                "in_progress": "◉",
                "done": "✓",
                "failed": "✗",
            }.get(s.status, "?")
            if s.result:
                result_str = " -> " + truncate_text_by_tokens(
                    s.result,
                    max_tokens=_PLAN_RESULT_MAX_TOKENS,
                    suffix="...",
                )
            else:
                result_str = ""
            lines.append(f"  {marker} [{s.id}] {s.description}{result_str}")
        return "\n".join(lines)

    @property
    def detailed_progress(self) -> str:
        """Full progress report with untruncated step results.

        Unlike progress_summary which truncates each step result for compact
        display, this property preserves complete results so that the
        replanner can accurately judge step quality and goal achievement.
        """
        lines = [f"Goal: {self.goal}"]
        for s in self.steps:
            marker = {
                "pending": "○",
                "in_progress": "◉",
                "done": "✓",
                "failed": "✗",
            }.get(s.status, "?")
            lines.append(f"  {marker} [{s.id}] {s.description}")
            if s.result:
                lines.append(f"    Result: {s.result}")
        return "\n".join(lines)


class ReplanDecision(BaseModel):
    """Structured decision from the replanner agent."""

    goal_achieved: bool = False
    should_replan: bool = False
    reason: str = ""
    updated_steps: list[PlanStep] | None = None
    final_answer: str | None = None
