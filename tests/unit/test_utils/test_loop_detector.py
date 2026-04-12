"""Unit tests for LoopDetector."""
from __future__ import annotations

from agent_harness.core.message import ToolCall
from agent_harness.utils.loop_detector import LoopDetector, LoopSignal, _fingerprint


def _tc(name: str, **kwargs: object) -> ToolCall:
    return ToolCall(name=name, arguments=kwargs)


# -- Fingerprint --


class TestFingerprint:
    def test_deterministic(self) -> None:
        tc = _tc("bash", cmd="ls")
        assert _fingerprint(tc) == _fingerprint(tc)

    def test_args_order_irrelevant(self) -> None:
        tc1 = ToolCall(name="f", arguments={"a": 1, "b": 2})
        tc2 = ToolCall(name="f", arguments={"b": 2, "a": 1})
        assert _fingerprint(tc1) == _fingerprint(tc2)

    def test_different_name(self) -> None:
        assert _fingerprint(_tc("a", x=1)) != _fingerprint(_tc("b", x=1))

    def test_different_args(self) -> None:
        assert _fingerprint(_tc("a", x=1)) != _fingerprint(_tc("a", x=2))

    def test_empty_args(self) -> None:
        fp = _fingerprint(_tc("f"))
        assert "f" in fp


# -- Streak counting --


class TestStreak:
    def test_no_history(self) -> None:
        d = LoopDetector()
        assert d._check().level == "none"
        assert d._check().streak == 0

    def test_single_step(self) -> None:
        d = LoopDetector()
        d.record([_tc("bash", cmd="ls")])
        assert d._check().level == "none"

    def test_two_different(self) -> None:
        d = LoopDetector()
        d.record([_tc("bash", cmd="ls")])
        d.record([_tc("bash", cmd="pwd")])
        assert d._check().streak == 1

    def test_two_identical(self) -> None:
        d = LoopDetector()
        d.record([_tc("bash", cmd="ls")])
        d.record([_tc("bash", cmd="ls")])
        sig = d._check()
        assert sig.streak == 2
        assert sig.level == "none"

    def test_streak_three(self) -> None:
        d = LoopDetector(soft_threshold=3)
        for _ in range(3):
            d.record([_tc("bash", cmd="ls")])
        sig = d._check()
        assert sig.streak == 3
        assert sig.level == "soft"

    def test_streak_reset_on_different_call(self) -> None:
        d = LoopDetector()
        d.record([_tc("bash", cmd="ls")])
        d.record([_tc("bash", cmd="ls")])
        d.record([_tc("read", path="a.py")])
        d.record([_tc("bash", cmd="ls")])
        assert d._check().streak == 1

    def test_empty_tool_calls_not_recorded(self) -> None:
        d = LoopDetector()
        d.record([_tc("f")])
        d.record([])
        d.record([_tc("f")])
        assert d._check().streak == 2


# -- Thresholds --


class TestThresholds:
    def test_soft(self) -> None:
        d = LoopDetector(soft_threshold=3)
        for _ in range(3):
            d.record([_tc("f")])
        assert d._check().level == "soft"

    def test_hard(self) -> None:
        d = LoopDetector(hard_threshold=5)
        for _ in range(5):
            d.record([_tc("f")])
        assert d._check().level == "hard"

    def test_break(self) -> None:
        d = LoopDetector(break_threshold=7)
        for _ in range(7):
            d.record([_tc("f")])
        assert d._check().level == "break"

    def test_custom_thresholds(self) -> None:
        d = LoopDetector(soft_threshold=2, hard_threshold=3, break_threshold=4)
        d.record([_tc("f")])
        d.record([_tc("f")])
        assert d._check().level == "soft"
        d.record([_tc("f")])
        assert d._check().level == "hard"
        d.record([_tc("f")])
        assert d._check().level == "break"

    def test_between_soft_and_hard(self) -> None:
        d = LoopDetector(soft_threshold=3, hard_threshold=5)
        for _ in range(4):
            d.record([_tc("f")])
        assert d._check().level == "soft"

    def test_between_hard_and_break(self) -> None:
        d = LoopDetector(hard_threshold=5, break_threshold=7)
        for _ in range(6):
            d.record([_tc("f")])
        assert d._check().level == "hard"


# -- Parallel calls --


class TestParallelCalls:
    def test_same_parallel_set(self) -> None:
        d = LoopDetector()
        calls = [_tc("a", x=1), _tc("b", x=2)]
        d.record(calls)
        d.record(calls)
        d.record(calls)
        assert d._check().streak == 3

    def test_different_order_same_set(self) -> None:
        d = LoopDetector()
        d.record([_tc("a"), _tc("b")])
        d.record([_tc("b"), _tc("a")])
        d.record([_tc("a"), _tc("b")])
        assert d._check().streak == 3

    def test_different_parallel_set(self) -> None:
        d = LoopDetector()
        d.record([_tc("a"), _tc("b")])
        d.record([_tc("a"), _tc("c")])
        assert d._check().streak == 1


# -- Warning messages --


class TestWarningMessage:
    def test_none_level(self) -> None:
        d = LoopDetector()
        assert d.build_warning_message(LoopSignal(level="none", streak=0)) is None

    def test_soft_level(self) -> None:
        d = LoopDetector()
        msg = d.build_warning_message(LoopSignal(level="soft", streak=3))
        assert msg is not None
        assert msg.role.value == "system"
        assert "repeating" in (msg.content or "").lower()

    def test_hard_includes_streak(self) -> None:
        d = LoopDetector()
        msg = d.build_warning_message(LoopSignal(level="hard", streak=5))
        assert msg is not None
        assert "5" in (msg.content or "")

    def test_break_returns_none(self) -> None:
        d = LoopDetector()
        assert d.build_warning_message(LoopSignal(level="break", streak=7)) is None


# -- Reset --


class TestReset:
    def test_reset_clears(self) -> None:
        d = LoopDetector()
        for _ in range(5):
            d.record([_tc("f")])
        d.reset()
        assert d._check().level == "none"
        assert d._check().streak == 0

    def test_reset_then_new_records(self) -> None:
        d = LoopDetector(soft_threshold=3)
        for _ in range(5):
            d.record([_tc("f")])
        d.reset()
        d.record([_tc("f")])
        d.record([_tc("f")])
        assert d._check().level == "none"
        assert d._check().streak == 2
