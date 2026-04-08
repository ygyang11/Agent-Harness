"""Tests for standard section factories and create_default_builder."""

from __future__ import annotations

from agent_harness.prompt.sections import (
    DEFAULT_INTRO,
    create_default_builder,
    make_custom_section,
    make_guidelines_section,
    make_intro_section,
    make_skills_section,
    make_tools_section,
)


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


class TestMakeIntroSection:
    def test_static_content(self) -> None:
        s = make_intro_section("You are a researcher.")
        assert s.resolve() == "You are a researcher."

    def test_name_and_order(self) -> None:
        s = make_intro_section("X")
        assert s.name == "intro"
        assert s.order == 100


class TestMakeGuidelinesSection:
    def test_active_with_tools(self) -> None:
        s = make_guidelines_section()
        result = s.resolve({"tools": [_FakeTool("read_file")]})
        assert "## Doing Tasks" in result
        assert "Executing Actions with Care" in result

    def test_inactive_without_tools(self) -> None:
        s = make_guidelines_section()
        assert s.resolve({"tools": []}) == ""

    def test_inactive_no_context(self) -> None:
        s = make_guidelines_section()
        assert s.resolve({}) == ""


class TestMakeToolsSection:
    def test_filesystem_supplement(self) -> None:
        s = make_tools_section()
        result = s.resolve({"tools": [_FakeTool("read_file")]})
        assert "## File Operations" in result
        assert "read_file instead of cat/head/tail" in result

    def test_terminal_supplement(self) -> None:
        s = make_tools_section()
        result = s.resolve({"tools": [_FakeTool("terminal_tool")]})
        assert "## Terminal Commands" in result

    def test_both(self) -> None:
        s = make_tools_section()
        tools = [_FakeTool("read_file"), _FakeTool("terminal_tool")]
        result = s.resolve({"tools": tools})
        assert "## File Operations" in result
        assert "## Terminal Commands" in result

    def test_no_tools(self) -> None:
        s = make_tools_section()
        assert s.resolve({"tools": []}) == ""

    def test_unrelated_tool(self) -> None:
        s = make_tools_section()
        assert s.resolve({"tools": [_FakeTool("web_search")]}) == ""


class TestMakeSkillsSection:
    def test_inactive_without_skill_tool(self) -> None:
        s = make_skills_section()
        assert s.resolve({"tools": [_FakeTool("read_file")]}) == ""

    def test_active_with_skill_tool_and_catalog(self) -> None:
        class _FakeLoader:
            def get_catalog(self) -> str:
                return "- demo: A demo skill."

        s = make_skills_section()
        ctx = {
            "tools": [_FakeTool("skill_tool")],
            "skill_loader": _FakeLoader(),
        }
        result = s.resolve(ctx)
        assert "# Skills" in result
        assert "<available_skills>" in result
        assert "- demo: A demo skill." in result
        assert "Do not skip a matching skill" in result

    def test_empty_catalog(self) -> None:
        class _FakeLoader:
            def get_catalog(self) -> str:
                return "(no skills available)"

        s = make_skills_section()
        ctx = {
            "tools": [_FakeTool("skill_tool")],
            "skill_loader": _FakeLoader(),
        }
        assert s.resolve(ctx) == ""


class TestMakeCustomSection:
    def test_basic(self) -> None:
        s = make_custom_section("Custom content here.")
        assert s.resolve() == "Custom content here."
        assert s.name == "custom"
        assert s.order == 600


class TestDefaultIntro:
    def test_not_empty(self) -> None:
        assert len(DEFAULT_INTRO) > 100

    def test_mentions_tools(self) -> None:
        assert "tools" in DEFAULT_INTRO

    def test_not_coding_only(self) -> None:
        assert "research" in DEFAULT_INTRO
        assert "analysis" in DEFAULT_INTRO


class TestCreateDefaultBuilder:
    def test_empty_prompt_uses_default_intro(self) -> None:
        builder = create_default_builder("")
        result = builder.build({"tools": []})
        assert DEFAULT_INTRO.split("\n")[0] in result

    def test_custom_prompt_replaces_intro(self) -> None:
        builder = create_default_builder("You are a custom agent.")
        result = builder.build({"tools": []})
        assert "You are a custom agent." in result
        assert DEFAULT_INTRO.split("\n")[0] not in result

    def test_guidelines_with_tools(self) -> None:
        builder = create_default_builder("Agent.")
        tools = [_FakeTool("read_file")]
        result = builder.build({"tools": tools})
        assert "## Doing Tasks" in result

    def test_no_guidelines_without_tools(self) -> None:
        builder = create_default_builder("Agent.")
        result = builder.build({"tools": []})
        assert "## Doing Tasks" not in result

    def test_all_sections_registered(self) -> None:
        builder = create_default_builder("Agent.")
        names = builder.section_names
        assert "intro" in names
        assert "guidelines" in names
        assert "tools" in names
        assert "skills" in names
        assert "instructions" in names
