"""Loop detection for agent execution cycles."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from agent_harness.core.message import Message, ToolCall

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoopSignal:
    """Result of a loop check."""

    level: str  # "none", "soft", "hard", "break"
    streak: int


_SOFT_WARNING = (
    "You appear to be repeating the same tool call with the same arguments. "
    "The results are unlikely to change. Try to use different arguments or a "
    "different approach, or provide a final answer."
)

_HARD_WARNING = (
    "Loop detected: you have repeated the exact same tool call {streak} times "
    "consecutively with identical results. You MUST either:\n"
    "1. Try a fundamentally different approach\n"
    "2. Provide a final answer explaining what you've tried and why it failed\n"
    "Continuing the same action will cause termination."
)


def _fingerprint(tc: ToolCall) -> str:
    """Compute a deterministic fingerprint for a tool call."""
    return tc.name + "\0" + json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False)


class LoopDetector:
    """Detects consecutive identical tool call patterns.

    Checks the trailing streak of identical tool calls in step history.
    Thresholds are configurable; defaults: soft=3, hard=5, break=7.
    """

    def __init__(
        self,
        *,
        soft_threshold: int = 3,
        hard_threshold: int = 5,
        break_threshold: int = 7,
    ) -> None:
        self._soft = soft_threshold
        self._hard = hard_threshold
        self._break = break_threshold
        self._history: list[str] = []

    def record(self, tool_calls: list[ToolCall]) -> None:
        """Record tool calls from the latest step.

        If a step has multiple tool calls, fingerprint is the sorted
        concatenation of individual fingerprints (parallel calls as a unit).
        """
        if not tool_calls:
            return
        fps = sorted(_fingerprint(tc) for tc in tool_calls)
        combined = "\n".join(fps)
        self._history.append(combined)

    def _check(self) -> LoopSignal:
        """Check the current trailing streak and return a signal."""
        if len(self._history) < 2:
            return LoopSignal(level="none", streak=0)

        current = self._history[-1]
        streak = 1
        for fp in reversed(self._history[:-1]):
            if fp == current:
                streak += 1
            else:
                break

        if streak >= self._break:
            return LoopSignal(level="break", streak=streak)
        if streak >= self._hard:
            return LoopSignal(level="hard", streak=streak)
        if streak >= self._soft:
            return LoopSignal(level="soft", streak=streak)
        return LoopSignal(level="none", streak=streak)

    def build_warning_message(self, signal: LoopSignal) -> Message | None:
        """Build a system warning message for the given signal."""
        if signal.level == "soft":
            return Message.system(_SOFT_WARNING)
        if signal.level == "hard":
            return Message.system(_HARD_WARNING.format(streak=signal.streak))
        return None

    def reset(self) -> None:
        self._history.clear()
