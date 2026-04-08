"""Tests for sub-agent prompt builder inheritance."""

from __future__ import annotations

from agent_harness.prompt.section import PromptSection
from agent_harness.prompt.sections import (
    create_default_builder,
    make_intro_section,
)


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


class TestPlannerInheritance:
    def test_planner_preserves_json_contract(self) -> None:
        """PlannerAgent keeps its own system_prompt when inheriting builder."""
        parent_builder = create_default_builder("Parent agent role")
        child_builder = parent_builder.fork()
        planner_prompt = "You are a planner. Output JSON only."
        child_builder.register(make_intro_section(planner_prompt))
        result = child_builder.build({"tools": []})
        assert "Output JSON only" in result
        assert "Parent agent role" not in result

    def test_planner_inherits_instructions_section(self) -> None:
        parent_builder = create_default_builder("Parent")
        child_builder = parent_builder.fork()
        child_builder.register(make_intro_section("Planner"))
        assert child_builder.has("instructions")

    def test_planner_no_guidelines_without_tools(self) -> None:
        """PlannerAgent (tools=[]) should NOT get guidelines."""
        parent_builder = create_default_builder("Parent with tools")
        child_builder = parent_builder.fork()
        child_builder.register(make_intro_section("Planner"))
        result = child_builder.build({"tools": []})
        assert "## Doing Tasks" not in result


class TestExecutorInheritance:
    def test_executor_inherits_tool_supplements(self) -> None:
        parent_builder = create_default_builder("Parent agent")
        child_builder = parent_builder.fork()
        child_builder.register(make_intro_section("Executor"))
        tools = [_FakeTool("read_file"), _FakeTool("terminal_tool")]
        result = child_builder.build({"tools": tools})
        assert "## File Operations" in result
        assert "## Terminal Commands" in result
        assert "## Doing Tasks" in result

    def test_executor_gets_guidelines_with_tools(self) -> None:
        parent_builder = create_default_builder("Parent")
        child_builder = parent_builder.fork()
        child_builder.register(make_intro_section("Executor"))
        result = child_builder.build({"tools": [_FakeTool("read_file")]})
        assert "## Doing Tasks" in result


class TestForkIndependence:
    def test_child_changes_dont_affect_parent(self) -> None:
        parent = create_default_builder("Parent")
        child = parent.fork()
        child.register(PromptSection(name="extra", order=999, content="Child-only"))
        parent_result = parent.build({"tools": []})
        assert "Child-only" not in parent_result

    def test_child_intro_override_doesnt_affect_parent(self) -> None:
        parent = create_default_builder("Parent")
        child = parent.fork()
        child.register(make_intro_section("Child"))
        assert "Parent" in parent.build({"tools": []})
        assert "Child" in child.build({"tools": []})

    def test_multi_level_fork(self) -> None:
        gp = create_default_builder("Grandparent")
        p = gp.fork()
        p.register(make_intro_section("Parent"))
        c = p.fork()
        c.register(make_intro_section("Child"))
        assert "Grandparent" in gp.build({"tools": []})
        assert "Parent" in p.build({"tools": []})
        assert "Child" in c.build({"tools": []})
