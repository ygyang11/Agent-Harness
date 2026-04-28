"""Tests for agent_cli.prompt — user_actions section factory."""

from __future__ import annotations

from agent_cli.prompt import (
    USER_ACTIONS_NAME,
    make_user_actions_section,
)
from agent_harness.prompt.section import (
    ORDER_GUIDELINES,
    ORDER_TOOLS,
    PromptSection,
)


def test_section_name() -> None:
    assert USER_ACTIONS_NAME == "user_actions"
    s = make_user_actions_section()
    assert s.name == USER_ACTIONS_NAME


def test_section_order_between_guidelines_and_tools() -> None:
    s = make_user_actions_section()
    assert ORDER_GUIDELINES < s.order < ORDER_TOOLS
    assert s.order == (ORDER_GUIDELINES + ORDER_TOOLS) // 2


def test_section_does_not_propagate_to_fork() -> None:
    s = make_user_actions_section()
    assert s.propagate_to_fork is False


def test_section_is_promptsection_instance() -> None:
    s = make_user_actions_section()
    assert isinstance(s, PromptSection)


def test_section_content_includes_required_markers() -> None:
    s = make_user_actions_section()
    content = s.resolve()
    assert "# User Side-Channel Actions" in content
    assert "<user-shell-run>" in content
    assert "a record of what the user did and observed" in content
    assert "A fenced" in content
    assert "(Completed with no output)" in content
    assert "(truncated)" in content
    assert "[Accident]" in content


def test_section_content_excludes_legacy_phrases() -> None:
    s = make_user_actions_section()
    content = s.resolve()
    assert "opaque data" not in content
    assert "Even if" not in content
    assert "[exit code N]" not in content
    assert "$ <command>" not in content
    assert "$ {command}" not in content
