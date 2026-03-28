"""Session state model and abstract base for session backends."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agent_harness.core.message import Message

_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")


class SessionState(BaseModel):
    """Snapshot of all agent state that needs to survive process restarts."""

    session_id: str
    messages: list[Message] = Field(default_factory=list)
    working_memory_scratchpad: dict[str, Any] = Field(default_factory=dict)
    working_memory_history: list[Message] = Field(default_factory=list)
    variables_agent: dict[str, Any] = Field(default_factory=dict)
    variables_global: dict[str, Any] = Field(default_factory=dict)
    agent_state: str = "idle"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class BaseSession(ABC):
    """Abstract base for session persistence backends."""

    def __init__(self, session_id: str) -> None:
        if not _SAFE_ID_PATTERN.match(session_id):
            raise ValueError(
                f"session_id must match [a-zA-Z0-9_-], got: {session_id!r}"
            )
        self.session_id = session_id

    @abstractmethod
    async def load_state(self) -> SessionState | None: ...

    @abstractmethod
    async def save_state(self, state: SessionState) -> None: ...

    @abstractmethod
    async def clear(self) -> None: ...


def resolve_session(session: str | BaseSession | None) -> BaseSession | None:
    """Convert a session parameter to a BaseSession instance.

    Accepts str (auto-creates FileSession), BaseSession (pass-through), or None.
    """
    if session is None:
        return None
    if isinstance(session, str):
        from agent_harness.session.file_session import FileSession
        return FileSession(session)
    return session
