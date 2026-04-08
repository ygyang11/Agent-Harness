"""Prompt section — a named, ordered fragment of system prompt content."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Standard section order constants
ORDER_INTRO = 100
ORDER_GUIDELINES = 200
ORDER_TOOLS = 300
ORDER_SKILLS = 400
ORDER_INSTRUCTIONS = 500
ORDER_CUSTOM = 600


@dataclass(frozen=True)
class PromptSection:
    """A named, ordered section of the system prompt.

    Attributes:
        name: Unique identifier for this section.
        order: Sort weight (lower = earlier in prompt).
        content: Static string or callable(context) -> str for dynamic content.
        enabled: Master switch. If False, section is always skipped.
        condition: Optional callable(context) -> bool. If provided and returns
            False, section is skipped even when enabled=True.
    """

    name: str
    order: int
    content: str | Callable[..., str] = ""
    enabled: bool = True
    condition: Callable[..., bool] | None = None

    def resolve(self, context: dict[str, Any] | None = None) -> str:
        """Resolve this section to a string.

        Returns empty string if disabled or condition fails.
        """
        if not self.enabled:
            return ""
        if self.condition is not None:
            try:
                if not self.condition(context or {}):
                    return ""
            except Exception:
                logger.warning("Section '%s' condition failed", self.name)
                return ""
        if callable(self.content):
            try:
                return self.content(context or {})
            except Exception:
                logger.warning("Section '%s' content callback failed", self.name)
                return ""
        return self.content
