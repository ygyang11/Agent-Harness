"""SystemPromptBuilder — structured, extensible system prompt assembly."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent_harness.prompt.section import PromptSection


class SystemPromptBuilder:
    """Assembles system prompt from ordered, named sections.

    Sections are sorted by order, resolved, and joined with double newlines.
    Empty sections are skipped. The builder is designed to be inherited by
    child agents via fork().
    """

    def __init__(self) -> None:
        self._sections: dict[str, PromptSection] = {}

    def register(self, section: PromptSection) -> SystemPromptBuilder:
        """Register or replace a section by name."""
        self._sections[section.name] = section
        return self

    def unregister(self, name: str) -> SystemPromptBuilder:
        """Remove a section by name. No-op if not found."""
        self._sections.pop(name, None)
        return self

    def has(self, name: str) -> bool:
        """Check if a section with the given name is registered."""
        return name in self._sections

    def build(self, context: dict[str, Any] | None = None) -> str:
        """Resolve all sections and assemble the system prompt."""
        ordered = sorted(self._sections.values(), key=lambda s: s.order)
        parts: list[str] = []
        for section in ordered:
            resolved = section.resolve(context)
            if resolved.strip():
                parts.append(resolved.strip())
        return "\n\n".join(parts)

    def fork(self) -> SystemPromptBuilder:
        """Create an independent copy for child agents.

        Child inherits all parent sections but can override/extend
        without affecting the parent.
        """
        child = SystemPromptBuilder()
        child._sections = deepcopy(self._sections)
        return child

    @property
    def section_names(self) -> list[str]:
        """List registered section names in order."""
        return [s.name for s in sorted(self._sections.values(), key=lambda s: s.order)]
