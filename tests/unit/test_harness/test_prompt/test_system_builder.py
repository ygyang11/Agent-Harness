"""Tests for SystemPromptBuilder."""

from __future__ import annotations

from agent_harness.prompt.section import PromptSection
from agent_harness.prompt.system_builder import SystemPromptBuilder


class TestBuild:
    def test_build_empty(self) -> None:
        assert SystemPromptBuilder().build() == ""

    def test_build_single_section(self) -> None:
        b = SystemPromptBuilder()
        b.register(PromptSection(name="intro", order=100, content="Hello"))
        assert b.build() == "Hello"

    def test_build_respects_order(self) -> None:
        b = SystemPromptBuilder()
        b.register(PromptSection(name="b", order=200, content="Second"))
        b.register(PromptSection(name="a", order=100, content="First"))
        assert b.build() == "First\n\nSecond"

    def test_empty_sections_skipped(self) -> None:
        b = SystemPromptBuilder()
        b.register(PromptSection(name="a", order=100, content="Hello"))
        b.register(PromptSection(name="b", order=200, content=""))
        b.register(PromptSection(name="c", order=300, content="World"))
        assert b.build() == "Hello\n\nWorld"

    def test_whitespace_only_sections_skipped(self) -> None:
        b = SystemPromptBuilder()
        b.register(PromptSection(name="a", order=100, content="Hello"))
        b.register(PromptSection(name="b", order=200, content="   \n  "))
        assert b.build() == "Hello"

    def test_strips_section_content(self) -> None:
        b = SystemPromptBuilder()
        b.register(PromptSection(name="a", order=100, content="\n  Hello  \n"))
        assert b.build() == "Hello"

    def test_context_passed_to_sections(self) -> None:
        b = SystemPromptBuilder()
        b.register(PromptSection(name="dyn", order=100, content=lambda ctx: f"cwd={ctx['cwd']}"))
        assert b.build({"cwd": "/tmp"}) == "cwd=/tmp"


class TestRegister:
    def test_overwrite_by_name(self) -> None:
        b = SystemPromptBuilder()
        b.register(PromptSection(name="x", order=100, content="Old"))
        b.register(PromptSection(name="x", order=100, content="New"))
        assert b.build() == "New"

    def test_unregister(self) -> None:
        b = SystemPromptBuilder()
        b.register(PromptSection(name="x", order=100, content="Hello"))
        b.unregister("x")
        assert b.build() == ""

    def test_unregister_nonexistent_is_noop(self) -> None:
        b = SystemPromptBuilder()
        b.unregister("nonexistent")

    def test_has(self) -> None:
        b = SystemPromptBuilder()
        assert not b.has("x")
        b.register(PromptSection(name="x", order=100, content="Hello"))
        assert b.has("x")

    def test_section_names(self) -> None:
        b = SystemPromptBuilder()
        b.register(PromptSection(name="b", order=200, content="B"))
        b.register(PromptSection(name="a", order=100, content="A"))
        assert b.section_names == ["a", "b"]

    def test_register_returns_self(self) -> None:
        b = SystemPromptBuilder()
        result = b.register(PromptSection(name="x", order=100, content="X"))
        assert result is b


class TestFork:
    def test_fork_independent(self) -> None:
        parent = SystemPromptBuilder()
        parent.register(PromptSection(name="a", order=100, content="Shared"))
        child = parent.fork()
        child.register(PromptSection(name="b", order=200, content="ChildOnly"))
        assert "ChildOnly" not in parent.build()
        assert "Shared" in child.build()
        assert "ChildOnly" in child.build()

    def test_fork_override_intro(self) -> None:
        parent = SystemPromptBuilder()
        parent.register(PromptSection(name="intro", order=100, content="Parent role"))
        parent.register(PromptSection(name="guidelines", order=200, content="Shared rules"))
        child = parent.fork()
        child.register(PromptSection(name="intro", order=100, content="Child role"))
        assert "Parent role" in parent.build()
        assert "Child role" in child.build()
        assert "Shared rules" in child.build()

    def test_parent_unaffected_by_child_unregister(self) -> None:
        parent = SystemPromptBuilder()
        parent.register(PromptSection(name="a", order=100, content="A"))
        child = parent.fork()
        child.unregister("a")
        assert parent.build() == "A"
        assert child.build() == ""

    def test_multi_level_fork(self) -> None:
        grandparent = SystemPromptBuilder()
        grandparent.register(PromptSection(name="a", order=100, content="GP"))
        parent = grandparent.fork()
        parent.register(PromptSection(name="b", order=200, content="P"))
        child = parent.fork()
        child.register(PromptSection(name="a", order=100, content="C"))
        assert grandparent.build() == "GP"
        assert parent.build() == "GP\n\nP"
        assert child.build() == "C\n\nP"

    def test_fork_skips_non_propagating_sections(self) -> None:
        parent = SystemPromptBuilder()
        parent.register(PromptSection(name="a", order=100, content="Shared"))
        parent.register(
            PromptSection(
                name="root_only", order=200, content="RootOnly",
                propagate_to_fork=False,
            )
        )
        child = parent.fork()
        assert "Shared" in parent.build()
        assert "RootOnly" in parent.build()
        assert "Shared" in child.build()
        assert "RootOnly" not in child.build()
        assert child.has("root_only") is False

    def test_fork_default_sections_propagate(self) -> None:
        parent = SystemPromptBuilder()
        parent.register(PromptSection(name="a", order=100, content="A"))
        parent.register(PromptSection(name="b", order=200, content="B"))
        child = parent.fork()
        assert "A" in child.build()
        assert "B" in child.build()
