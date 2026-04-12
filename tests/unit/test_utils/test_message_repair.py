"""Unit tests for message repair utilities.

CRITICAL: patch_dangling_tool_calls is the last line of defense before messages
reach the LLM API. These tests verify not only that dangling/orphan issues are
fixed, but that the patch NEVER corrupts, reorders, or drops normal messages.
"""
from __future__ import annotations

from agent_harness.core.message import Message, Role, ToolCall
from agent_harness.utils.message_repair import (
    _DANGLING_CONTENT,
    patch_dangling_tool_calls,
)


def _assistant_with_calls(*names: str) -> tuple[Message, list[ToolCall]]:
    """Create an assistant message with tool calls, return (msg, calls)."""
    calls = [ToolCall(name=n, arguments={}) for n in names]
    msg = Message.assistant(content="thinking", tool_calls=calls)
    return msg, calls


def _tool_result(tc: ToolCall, content: str = "ok") -> Message:
    return Message.tool(tool_call_id=tc.id, content=content)


# -- Fast-path / no-op --


class TestNoop:
    def test_empty(self) -> None:
        result = patch_dangling_tool_calls([])
        assert result == []

    def test_no_tool_calls(self) -> None:
        msgs = [Message.user("hi"), Message.assistant("hello")]
        result = patch_dangling_tool_calls(msgs)
        assert result is msgs

    def test_all_resolved(self) -> None:
        msg, calls = _assistant_with_calls("bash")
        msgs = [msg, _tool_result(calls[0])]
        result = patch_dangling_tool_calls(msgs)
        assert result is msgs


# -- Dangling tool_calls --


class TestDangling:
    def test_single_dangling(self) -> None:
        msg, calls = _assistant_with_calls("bash")
        msgs = [Message.user("hi"), msg]
        result = patch_dangling_tool_calls(msgs)
        assert len(result) == 3
        assert result[2].role == Role.TOOL
        assert result[2].tool_result.tool_call_id == calls[0].id
        assert result[2].tool_result.is_error is True
        assert result[2].tool_result.content == _DANGLING_CONTENT

    def test_multiple_dangling_same_assistant(self) -> None:
        msg, calls = _assistant_with_calls("a", "b", "c")
        msgs = [msg]
        result = patch_dangling_tool_calls(msgs)
        assert len(result) == 4
        for i, tc in enumerate(calls):
            assert result[i + 1].tool_result.tool_call_id == tc.id
            assert result[i + 1].tool_result.is_error is True

    def test_partial_tail_missing(self) -> None:
        msg, calls = _assistant_with_calls("a", "b", "c")
        msgs = [msg, _tool_result(calls[0])]
        result = patch_dangling_tool_calls(msgs)
        assert len(result) == 4
        assert result[1].tool_result.tool_call_id == calls[0].id
        assert result[1].tool_result.is_error is False
        assert result[2].tool_result.tool_call_id == calls[1].id
        assert result[2].tool_result.is_error is True
        assert result[3].tool_result.tool_call_id == calls[2].id
        assert result[3].tool_result.is_error is True

    def test_synthetic_after_existing_results(self) -> None:
        msg, calls = _assistant_with_calls("a", "b", "c")
        msgs = [msg, _tool_result(calls[0]), _tool_result(calls[1])]
        result = patch_dangling_tool_calls(msgs)
        assert result[1].tool_result.is_error is False
        assert result[2].tool_result.is_error is False
        assert result[3].tool_result.is_error is True
        assert result[3].tool_result.tool_call_id == calls[2].id

    def test_synthetic_position_with_trailing_messages(self) -> None:
        msg, calls = _assistant_with_calls("bash")
        msgs = [Message.user("hi"), msg, Message.user("next")]
        result = patch_dangling_tool_calls(msgs)
        assert result[0].role == Role.USER
        assert result[1].role == Role.ASSISTANT
        assert result[2].role == Role.TOOL
        assert result[3].role == Role.USER


# -- Orphaned tool_results --


class TestOrphaned:
    def test_remove_orphaned(self) -> None:
        orphan = Message.tool(tool_call_id="nonexistent_id", content="stale")
        msgs = [Message.user("hi"), orphan]
        result = patch_dangling_tool_calls(msgs)
        assert len(result) == 1
        assert result[0].role == Role.USER

    def test_orphan_adjacent_to_assistant(self) -> None:
        msg, calls = _assistant_with_calls("bash")
        orphan = Message.tool(tool_call_id="ghost", content="stale")
        msgs = [msg, orphan]
        result = patch_dangling_tool_calls(msgs)
        assert len(result) == 2
        assert result[1].tool_result.tool_call_id == calls[0].id
        assert result[1].tool_result.is_error is True


# -- Immutability --


class TestImmutability:
    def test_does_not_mutate_input(self) -> None:
        msg, _calls = _assistant_with_calls("bash")
        msgs = [msg]
        original_len = len(msgs)
        patch_dangling_tool_calls(msgs)
        assert len(msgs) == original_len


# -- Message integrity (CRITICAL) --


class TestMessageIntegrity:
    def test_full_conversation_integrity(self) -> None:
        sys_msg = Message.system("You are helpful")
        user1 = Message.user("hello")
        ast1 = Message.assistant("hi there")
        ast2, calls2 = _assistant_with_calls("search", "read")
        tr2a = _tool_result(calls2[0], "found it")
        tr2b = _tool_result(calls2[1], "file contents")
        user2 = Message.user("now do X")
        ast3, calls3 = _assistant_with_calls("bash", "write")

        msgs = [sys_msg, user1, ast1, ast2, tr2a, tr2b, user2, ast3]
        result = patch_dangling_tool_calls(msgs)

        assert result[0] is sys_msg
        assert result[1] is user1
        assert result[2] is ast1
        assert result[3] is ast2
        assert result[4] is tr2a
        assert result[5] is tr2b
        assert result[6] is user2
        assert result[7] is ast3
        assert result[8].tool_result.tool_call_id == calls3[0].id
        assert result[9].tool_result.tool_call_id == calls3[1].id
        assert len(result) == 10

    def test_multiple_rounds_only_patch_broken(self) -> None:
        ast1, calls1 = _assistant_with_calls("a", "b")
        tr1a = _tool_result(calls1[0])
        tr1b = _tool_result(calls1[1])
        ast2, calls2 = _assistant_with_calls("c")

        msgs = [ast1, tr1a, tr1b, ast2]
        result = patch_dangling_tool_calls(msgs)

        assert result[0] is ast1
        assert result[1] is tr1a
        assert result[2] is tr1b
        assert result[3] is ast2
        assert result[4].tool_result.tool_call_id == calls2[0].id
        assert len(result) == 5

    def test_non_tool_messages_never_lost(self) -> None:
        sys_msg = Message.system("sys")
        user1 = Message.user("u1")
        ast_plain = Message.assistant("plain response")
        user2 = Message.user("u2")
        ast_broken, _calls = _assistant_with_calls("bash")
        user3 = Message.user("u3")

        msgs = [sys_msg, user1, ast_plain, user2, ast_broken, user3]
        result = patch_dangling_tool_calls(msgs)

        non_tool = [m for m in result if m.role != Role.TOOL]
        assert non_tool == [sys_msg, user1, ast_plain, user2, ast_broken, user3]

    def test_existing_results_order_preserved(self) -> None:
        ast, calls = _assistant_with_calls("a", "b", "c", "d")
        tr_a = _tool_result(calls[0], "result_a")
        tr_b = _tool_result(calls[1], "result_b")

        msgs = [ast, tr_a, tr_b]
        result = patch_dangling_tool_calls(msgs)

        assert result[1] is tr_a
        assert result[2] is tr_b
        assert result[3].tool_result.tool_call_id == calls[2].id
        assert result[4].tool_result.tool_call_id == calls[3].id

    def test_normal_conversation_passthrough(self) -> None:
        ast, calls = _assistant_with_calls("bash")
        msgs = [
            Message.system("sys"),
            Message.user("hi"),
            ast,
            _tool_result(calls[0]),
            Message.assistant("done"),
        ]
        result = patch_dangling_tool_calls(msgs)
        assert result is msgs
