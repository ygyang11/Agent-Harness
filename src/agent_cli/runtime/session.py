"""Runtime-layer wrappers around session state mutation."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from agent_harness.agent.base import BaseAgent
from agent_harness.core.message import Message
from agent_harness.llm import BaseLLM
from agent_harness.session.base import BaseSession

SaveSession = Callable[[], Awaitable[None]]


def reset_approval(agent: BaseAgent) -> None:
    if agent._approval is not None:
        agent._approval.reset_session()


def export_approval_grants(agent: BaseAgent) -> dict[str, Any] | None:
    if agent._approval is None:
        return None
    return agent._approval.export_session_grants()


async def stop_sandbox(agent: BaseAgent) -> None:
    await agent._sandbox.stop()


def get_messages(agent: BaseAgent) -> list[Message]:
    return list(agent.context.short_term_memory._messages)


def set_messages(agent: BaseAgent, messages: list[Message]) -> None:
    agent.context.short_term_memory._messages = messages


def update_compressor_model(
    agent: BaseAgent, new_model: str, new_llm: BaseLLM,
) -> None:
    compressor = agent.context.short_term_memory.compressor
    cfg = agent.context.config
    if compressor is not None and cfg.memory.compression.summary_model is None:
        compressor._model = new_model
        compressor._llm = new_llm


def session_created_at(agent: BaseAgent) -> datetime | None:
    return agent._session_created_at


def make_save_session(
    agent: BaseAgent, session_id: str, backend: BaseSession,
) -> SaveSession:
    """Build a no-arg async closure that snapshots + persists the current session."""
    async def _save() -> None:
        now = datetime.now()
        ss = agent.context.to_session_state(session_id, agent_name=agent.name)
        ss.created_at = session_created_at(agent) or now
        ss.updated_at = now
        tool_states = agent.tool_registry.save_states()
        if tool_states:
            ss.metadata["_tool_states"] = tool_states
        grants = export_approval_grants(agent)
        if grants:
            ss.metadata["_approval_grants"] = grants
        await backend.save_state(ss)

    return _save
