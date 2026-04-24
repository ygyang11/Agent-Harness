"""Tests for after_input_appended callback on BaseAgent.run / PlanAndExecuteAgent.run."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_harness.agent.base import BaseAgent
from agent_harness.agent.react import ReActAgent
from agent_harness.context.state import AgentState
from agent_harness.core.message import Message, Role
from tests.conftest import MockLLM


class TestBaseAgentCallback:
    @pytest.mark.asyncio
    async def test_no_callback_keeps_run_unchanged(self) -> None:
        llm = MockLLM()
        llm.add_text_response("done")
        agent = ReActAgent(name="test", llm=llm, max_steps=2)

        result = await agent.run("hello")

        assert result.output == "done"

    @pytest.mark.asyncio
    async def test_callback_fires_with_correct_args(self) -> None:
        llm = MockLLM()
        llm.add_text_response("done")
        agent = ReActAgent(name="test", llm=llm, max_steps=2)

        captured: dict[str, Any] = {}

        async def cb(a: BaseAgent, msg: Message, text: str) -> None:
            captured["agent"] = a
            captured["msg"] = msg
            captured["text"] = text

        await agent.run("hello", after_input_appended=cb)

        assert captured["agent"] is agent
        assert captured["msg"].role == Role.USER
        assert captured["msg"].content == "hello"
        assert captured["text"] == "hello"

    @pytest.mark.asyncio
    async def test_callback_runs_after_user_append_before_first_step(self) -> None:
        llm = MockLLM()
        llm.add_text_response("answer")
        agent = ReActAgent(name="test", llm=llm, max_steps=2)

        async def cb(a: BaseAgent, msg: Message, text: str) -> None:
            messages = await a.context.short_term_memory.get_context_messages()
            roles = [m.role for m in messages]
            assert Role.USER in roles, "user message must be appended before callback"
            assert llm.call_history == [], "LLM must not be called before callback"

        await agent.run("hi", after_input_appended=cb)

    @pytest.mark.asyncio
    async def test_callback_can_inject_messages_visible_to_llm(self) -> None:
        llm = MockLLM()
        llm.add_text_response("seen")
        agent = ReActAgent(name="test", llm=llm, max_steps=2)

        async def cb(a: BaseAgent, msg: Message, text: str) -> None:
            await a.context.short_term_memory.add_message(
                Message.assistant(content="(synthetic note)")
            )

        await agent.run("ask", after_input_appended=cb)

        first_call = llm.call_history[0]
        contents = [m.content for m in first_call]
        assert "(synthetic note)" in contents

    @pytest.mark.asyncio
    async def test_callback_exception_routes_through_on_error(self) -> None:
        llm = MockLLM()
        llm.add_text_response("never reached")
        agent = ReActAgent(name="test", llm=llm, max_steps=2)

        async def cb(a: BaseAgent, msg: Message, text: str) -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await agent.run("hi", after_input_appended=cb)

        assert agent.context.state.current == AgentState.ERROR

    @pytest.mark.asyncio
    async def test_callback_cancelled_propagates(self) -> None:
        llm = MockLLM()
        llm.add_text_response("never")
        agent = ReActAgent(name="test", llm=llm, max_steps=2)

        async def cb(a: BaseAgent, msg: Message, text: str) -> None:
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await agent.run("hi", after_input_appended=cb)


class TestPlanAndExecuteCallback:
    @pytest.mark.asyncio
    async def test_callback_fires_at_top_level_only(self) -> None:
        from agent_harness.agent.planner.plan_and_execute import PlanAndExecuteAgent

        llm = MockLLM()
        llm.add_text_response('{"steps":[]}')
        llm.add_text_response("final")
        agent = PlanAndExecuteAgent(
            name="top", llm=llm, max_steps=1, executor_max_steps=1,
        )

        fired: list[str] = []

        async def cb(a: BaseAgent, msg: Message, text: str) -> None:
            fired.append(a.name)

        try:
            await agent.run("hi", after_input_appended=cb)
        except Exception:
            pass

        assert fired == ["top"], (
            f"callback should fire once on top-level only; got {fired}"
        )
