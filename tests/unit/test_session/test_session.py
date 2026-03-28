"""Tests for session module: SessionState, InMemorySession, FileSession."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_harness.core.message import Message, ToolCall
from agent_harness.session.base import BaseSession, SessionState, resolve_session
from agent_harness.session.file_session import FileSession
from agent_harness.session.memory_session import InMemorySession


class TestSessionState:
    async def test_roundtrip_serialization(self) -> None:
        state = SessionState(
            session_id="test",
            messages=[Message.user("hello"), Message.assistant("hi")],
            working_memory_scratchpad={"goal": "test"},
            variables_agent={"key": "value"},
        )
        json_str = state.model_dump_json()
        restored = SessionState.model_validate_json(json_str)
        assert restored.session_id == "test"
        assert len(restored.messages) == 2
        assert restored.working_memory_scratchpad["goal"] == "test"
        assert restored.variables_agent["key"] == "value"

    async def test_roundtrip_with_tool_calls(self) -> None:
        tc = ToolCall(name="search", arguments={"q": "test"})
        msg = Message.assistant(content=None, tool_calls=[tc])
        state = SessionState(session_id="test", messages=[msg])
        json_str = state.model_dump_json()
        restored = SessionState.model_validate_json(json_str)
        assert restored.messages[0].tool_calls is not None
        assert restored.messages[0].tool_calls[0].name == "search"


class TestSessionIdValidation:
    def test_rejects_path_traversal(self) -> None:
        with pytest.raises(ValueError):
            InMemorySession("../../etc/passwd")

    def test_rejects_spaces(self) -> None:
        with pytest.raises(ValueError):
            InMemorySession("has spaces")

    def test_accepts_valid_id(self) -> None:
        InMemorySession("user_123-abc")


class TestInMemorySession:
    async def test_save_load_clear(self) -> None:
        session = InMemorySession("s1")
        assert await session.load_state() is None

        state = SessionState(session_id="s1", messages=[Message.user("hi")])
        await session.save_state(state)

        loaded = await session.load_state()
        assert loaded is not None
        assert loaded.messages[0].content == "hi"

        await session.clear()
        assert await session.load_state() is None


class TestFileSession:
    async def test_persistence_across_instances(self, tmp_path: Path) -> None:
        session = FileSession("s1", path=tmp_path)
        state = SessionState(session_id="s1", messages=[Message.user("hello")])
        await session.save_state(state)

        session2 = FileSession("s1", path=tmp_path)
        loaded = await session2.load_state()
        assert loaded is not None
        assert loaded.messages[0].content == "hello"

    async def test_corrupted_file_returns_none(self, tmp_path: Path) -> None:
        session = FileSession("s1", path=tmp_path)
        (tmp_path / "s1.json").write_text("not valid json", encoding="utf-8")
        assert await session.load_state() is None

    async def test_clear_removes_file(self, tmp_path: Path) -> None:
        session = FileSession("s1", path=tmp_path)
        state = SessionState(session_id="s1", messages=[Message.user("hi")])
        await session.save_state(state)
        assert (tmp_path / "s1.json").exists()
        await session.clear()
        assert not (tmp_path / "s1.json").exists()

    async def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        session = FileSession("nope", path=tmp_path)
        assert await session.load_state() is None


class TestResolveSession:
    def test_none_returns_none(self) -> None:
        assert resolve_session(None) is None

    def test_string_creates_file_session(self) -> None:
        result = resolve_session("my-session")
        assert isinstance(result, FileSession)
        assert result.session_id == "my-session"

    def test_base_session_passes_through(self) -> None:
        session = InMemorySession("s1")
        assert resolve_session(session) is session
