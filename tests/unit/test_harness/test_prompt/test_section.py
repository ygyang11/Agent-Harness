"""Tests for PromptSection."""

from __future__ import annotations

from agent_harness.prompt.section import (
    ORDER_CUSTOM,
    ORDER_GUIDELINES,
    ORDER_INTRO,
    ORDER_TOOLS,
    PromptSection,
)


class TestResolve:
    def test_static_content(self) -> None:
        s = PromptSection(name="intro", order=ORDER_INTRO, content="Hello")
        assert s.resolve() == "Hello"

    def test_empty_content(self) -> None:
        s = PromptSection(name="x", order=100, content="")
        assert s.resolve() == ""

    def test_callable_content(self) -> None:
        s = PromptSection(name="env", order=100, content=lambda ctx: f"date: {ctx['date']}")
        assert s.resolve({"date": "2026-04-08"}) == "date: 2026-04-08"

    def test_callable_receives_empty_dict_when_no_context(self) -> None:
        s = PromptSection(name="x", order=100, content=lambda ctx: f"keys={len(ctx)}")
        assert s.resolve() == "keys=0"

    def test_disabled_returns_empty(self) -> None:
        s = PromptSection(name="x", order=100, content="Hello", enabled=False)
        assert s.resolve() == ""

    def test_condition_false_returns_empty(self) -> None:
        s = PromptSection(name="x", order=100, content="Hello", condition=lambda _: False)
        assert s.resolve() == ""

    def test_condition_true_returns_content(self) -> None:
        s = PromptSection(name="x", order=100, content="Hello", condition=lambda _: True)
        assert s.resolve() == "Hello"

    def test_condition_uses_context(self) -> None:
        s = PromptSection(
            name="x",
            order=100,
            content="Has tools",
            condition=lambda ctx: bool(ctx.get("tools")),
        )
        assert s.resolve({"tools": []}) == ""
        assert s.resolve({"tools": ["read_file"]}) == "Has tools"

    def test_callback_exception_returns_empty(self) -> None:
        s = PromptSection(name="x", order=100, content=lambda _: 1 / 0)  # type: ignore[return-value]
        assert s.resolve() == ""

    def test_condition_exception_returns_empty(self) -> None:
        s = PromptSection(
            name="x",
            order=100,
            content="Hello",
            condition=lambda _: 1 / 0,  # type: ignore[return-value]
        )
        assert s.resolve() == ""


class TestFrozen:
    def test_immutable(self) -> None:
        s = PromptSection(name="x", order=100, content="Hello")
        try:
            s.name = "y"  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestOrderConstants:
    def test_ordering(self) -> None:
        assert ORDER_INTRO < ORDER_GUIDELINES < ORDER_TOOLS < ORDER_CUSTOM


class TestPropagateToFork:
    def test_default_is_true(self) -> None:
        s = PromptSection(name="x", order=100, content="Hello")
        assert s.propagate_to_fork is True

    def test_explicit_false(self) -> None:
        s = PromptSection(
            name="x", order=100, content="Hello", propagate_to_fork=False,
        )
        assert s.propagate_to_fork is False
