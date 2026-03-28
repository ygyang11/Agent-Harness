"""In-memory session backend for testing and single-process use."""
from __future__ import annotations

from agent_harness.session.base import BaseSession, SessionState


class InMemorySession(BaseSession):

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self._state: SessionState | None = None

    async def load_state(self) -> SessionState | None:
        return self._state

    async def save_state(self, state: SessionState) -> None:
        self._state = state

    async def clear(self) -> None:
        self._state = None
