"""Runtime-layer wrapper around BackgroundTaskManager private access."""
from __future__ import annotations

import asyncio

from agent_harness.agent.base import BaseAgent
from agent_harness.background import BackgroundTask


def has_running(agent: BaseAgent) -> bool:
    return agent._bg_manager.has_running()


async def wait_next(agent: BaseAgent) -> None:
    await agent._bg_manager.wait_next()


async def collect_results(agent: BaseAgent) -> bool:
    return bool(await agent._collect_background_results())


def cancel_all(agent: BaseAgent) -> int:
    return agent._bg_manager.cancel_all()


async def shutdown(agent: BaseAgent) -> None:
    await agent._bg_manager.shutdown()


def get_all(agent: BaseAgent) -> list[BackgroundTask]:
    return agent._bg_manager.get_all()


def clear_tasks(agent: BaseAgent) -> None:
    agent._bg_manager._tasks.clear()


def is_current_task_background(agent: BaseAgent) -> bool:
    current = asyncio.current_task()
    if current is None:
        return False
    return any(bg.asyncio_task is current for bg in agent._bg_manager.get_all())
