"""Base agent class for agent_harness."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_harness.background import BackgroundTask
    from agent_harness.prompt.system_builder import SystemPromptBuilder
    from agent_harness.sandbox.backend import SandboxBackend
    from agent_harness.session.base import BaseSession

from pydantic import BaseModel, Field

from agent_harness.approval import (
    ApprovalAction,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResult,
    resolve_approval,
    resolve_approval_handler,
)
from agent_harness.approval.rules import extract_resource
from agent_harness.context.context import AgentContext
from agent_harness.context.state import AgentState
from agent_harness.core.config import HarnessConfig
from agent_harness.core.errors import MaxStepsExceededError
from agent_harness.core.event import EventEmitter
from agent_harness.core.message import Message, Role, ToolCall, ToolResult
from agent_harness.hooks import DefaultHooks, resolve_hooks
from agent_harness.llm import create_llm
from agent_harness.llm.base import BaseLLM
from agent_harness.llm.types import LLMResponse, StreamDelta, Usage
from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.tool.executor import ToolExecutor
from agent_harness.tool.registry import ToolRegistry

logger = logging.getLogger(__name__)


class StepResult(BaseModel):
    """Result of a single agent step."""

    thought: str | None = None
    action: list[ToolCall] | None = None
    observation: list[ToolResult] | None = None
    response: str | None = None  # final response if step produced one


class AgentResult(BaseModel):
    """Final result of an agent run."""

    output: str
    messages: list[Message] = Field(default_factory=list)
    steps: list[StepResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

    @property
    def step_count(self) -> int:
        return len(self.steps)


class BaseAgent(ABC, EventEmitter):
    """Abstract base class for all agents.

    Provides the run loop, tool execution, and lifecycle management.
    Subclasses implement step() to define their reasoning strategy.

    Args:
        name: Unique agent name.
        llm: LLM provider for generation.
        tools: List of available tools.
        context: Agent runtime context.
        hooks: Lifecycle hooks (inherits DefaultHooks). When tracing.enabled=True
            and hooks is not provided, TracingHooks is auto-created from config.
        max_steps: Maximum steps before forced termination.
        system_prompt: System prompt for the agent.
        use_long_term_memory: If True, call_llm() queries long-term memory by default.
        config: Optional config used to create context when context is not provided.
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM | None = None,
        tools: list[BaseTool] | None = None,
        context: AgentContext | None = None,
        hooks: DefaultHooks | None = None,
        max_steps: int = 100,
        system_prompt: str = "",
        use_long_term_memory: bool = False,
        stream: bool = True,
        *,
        config: HarnessConfig | None = None,
        approval: ApprovalPolicy | None = None,
        approval_handler: ApprovalHandler | None = None,
        prompt_builder: SystemPromptBuilder | None = None,
        sandbox: SandboxBackend | None = None,
    ) -> None:
        from agent_harness.prompt.runtime_context import RuntimeContextProvider
        from agent_harness.prompt.sections import create_default_builder, make_intro_section

        self.name = name
        if context is not None:
            self.context = context
        else:
            self.context = AgentContext.create(config=config)
        self.llm = llm or create_llm(self.context.config)
        self.hooks = resolve_hooks(hooks, self.context.config)
        self.max_steps = max_steps
        self.use_long_term_memory = use_long_term_memory
        self._stream = stream
        self._total_usage = Usage()
        self._session_created_at: datetime | None = None

        # Approval setup
        self._approval = resolve_approval(approval, self.context.config)
        self._approval_handler: ApprovalHandler | None = (
            resolve_approval_handler(approval_handler) if self._approval is not None else None
        )

        # Set up tool registry and executor
        from agent_harness.tool.base import AgentAware

        self.tool_registry = ToolRegistry()
        for t in tools or []:
            self.tool_registry.register(t)
            if isinstance(t, AgentAware):
                t.bind_agent(self)
        self.tool_executor = ToolExecutor(
            self.tool_registry,
            config=self.context.config,
        )

        # System prompt builder
        if prompt_builder is not None:
            self._prompt_builder = prompt_builder
            if system_prompt:
                self._prompt_builder.register(make_intro_section(system_prompt))
        else:
            self._prompt_builder = create_default_builder(system_prompt)
        self.system_prompt = self._prompt_builder.build(self._make_builder_context())

        # Runtime context provider (ephemeral layer)
        self._runtime_ctx = RuntimeContextProvider()

        # Wire event bus
        self.set_event_bus(self.context.event_bus)
        self.tool_executor.set_event_bus(self.context.event_bus)
        self.llm.set_event_bus(self.context.event_bus)

        # Loop detection
        from agent_harness.utils.loop_detector import LoopDetector

        self._loop_detector = LoopDetector()
        self._pending_loop_warning: Message | None = None

        # Background task manager
        from agent_harness.background import BackgroundTaskManager

        self._bg_manager = BackgroundTaskManager()

        # Sandbox, LocalBackend when disabled, DockerBackend when enabled)
        from agent_harness.sandbox import SandboxManager, resolve_sandbox

        self._sandbox: SandboxManager = resolve_sandbox(sandbox, self.context.config)

        # Context compression setup
        if (
            self.context.config.memory.strategy == "summarize"
            and self.context.short_term_memory.compressor is None
        ):
            self._init_compressor()

    def _init_compressor(self) -> None:
        from agent_harness.memory.compressor import create_compressor

        comp_cfg = self.context.config.memory.compression
        summary_llm = self.llm
        if comp_cfg.summary_model:
            summary_llm = create_llm(
                self.context.config,
                model_override=comp_cfg.summary_model,
            )
        compressor = create_compressor(
            llm=summary_llm,
            memory_config=self.context.config.memory,
            model=self.context.config.llm.model,
        )
        self.context.short_term_memory.compressor = compressor
        self.context._compressor = compressor

    def _make_builder_context(self) -> dict[str, Any]:
        """Prepare context dict for SystemPromptBuilder.build()."""
        from pathlib import Path

        skill_loader = None
        for tool in self.tools:
            if tool.name == "skill_tool" and hasattr(tool, "loader"):
                skill_loader = tool.loader
                break
        return {
            "tools": self.tools,
            "config": self.context.config,
            "cwd": str(Path.cwd()),
            "skill_loader": skill_loader,
        }
    
    async def _collect_background_results(self) -> list[Any]:
        """Harvest completed background tasks and inject results into memory."""
        completed = self._bg_manager.collect_completed()
        for task in completed:
            if task.status == "completed" and task.result:
                content = (
                    f"[Background Task Completed] {task.task_id} ({task.tool_name}): "
                    f"{task.description}\n{task.result.summary}"
                )
                if task.result.output_path:
                    content += f"\nFull output: {task.result.output_path}"
            elif task.status == "failed":
                content = (
                    f"[Background Task Failed] {task.task_id} ({task.tool_name}): "
                    f"{task.description}\nError: {task.error}"
                )
            else:
                continue
            await self.context.short_term_memory.add_message(
                Message.system(content, metadata={"is_background_result": True})
            )
        return completed

    async def _check_loop(self, tool_calls: list[ToolCall]) -> None:
        """Record tool calls and check for repetitive loop pattern."""
        self._loop_detector.record(tool_calls)
        signal = self._loop_detector._check()
        if signal.level == "break":
            from agent_harness.core.errors import LoopDetectedError

            names = [tc.name for tc in tool_calls]
            raise LoopDetectedError(
                f"Agent '{self.name}' stuck in loop: "
                f"{signal.streak} consecutive identical calls to {names}",
                streak=signal.streak,
            )
        warning = self._loop_detector.build_warning_message(signal)
        if warning:
            self._pending_loop_warning = warning
            logger.warning(
                "Loop %s warning for agent '%s': streak=%d",
                signal.level,
                self.name,
                signal.streak,
            )

    async def _should_inject_system_prompt(self) -> bool:
        if not self.system_prompt:
            return False

        context_messages = await self.context.short_term_memory.get_context_messages()
        if not context_messages:
            return True

        first_message = context_messages[0]
        return not (
            first_message.role == Role.SYSTEM
            and (first_message.content or "") == self.system_prompt
        )

    @property
    def tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = self.tool_registry.list_tools()
        return tools

    @property
    def tool_schemas(self) -> list[ToolSchema]:
        schemas: list[ToolSchema] = self.tool_registry.get_schemas()
        return schemas


    async def run(
        self,
        input: str | Message,
        *,
        session: str | BaseSession | None = None,
    ) -> AgentResult:
        """Main execution loop.

        Repeatedly calls step() until:
        1. step() returns a final response, or
        2. max_steps is reached.

        Safe to call multiple times — state is reset automatically when
        the agent is in a terminal state (FINISHED or ERROR).

        Pass session (str or BaseSession) to enable persistence across restarts.
        """
        from agent_harness.session.base import resolve_session

        resolved_session: BaseSession | None = resolve_session(session)

        # Propagate session_id to compressor and background manager
        if resolved_session:
            if self.context.short_term_memory.compressor:
                self.context.short_term_memory.compressor.bind_session(resolved_session.session_id)
            self._bg_manager.bind_session(resolved_session.session_id)

        # Reset state for agent reuse (e.g., team orchestration, pipelines)
        if self.context.state.is_terminal:
            self.context.state.reset()

        # Session, Compressor, Tool restore: only when context is empty (first call or cross-process)
        if resolved_session and not await self.context.short_term_memory.get_context_messages():
            state = await resolved_session.load_state()
            if state:
                await self.context.restore_from_state(state, self.system_prompt)
                self._session_created_at = state.created_at
                
                compressor = self.context.short_term_memory.compressor
                if compressor:
                    compressor.restore_runtime_state(state.messages)

                await self.tool_registry.restore_states(
                    state.metadata.get("_tool_states", {}),
                    self.hooks,
                    self.name,
                )

        # Normalize input
        if isinstance(input, str):
            input_msg = Message.user(input)
            input_text = input
        else:
            input_msg = input
            input_text = input.content or ""

        # Initialize context
        if await self._should_inject_system_prompt():
            await self.context.short_term_memory.add_message(Message.system(self.system_prompt))
        await self.context.short_term_memory.add_message(input_msg)
        self.context.state.transition(AgentState.THINKING)

        await self.hooks.on_run_start(self.name, input_text)
        await self.emit("agent.run.start", agent=self.name, input=input_text)

        steps: list[StepResult] = []
        self._total_usage = Usage()
        self._loop_detector.reset()
        self._pending_loop_warning = None
        final_output = ""

        try:
            for step_num in range(1, self.max_steps + 1):
                # Harvest completed background tasks
                await self._collect_background_results()

                await self.hooks.on_step_start(self.name, step_num)
                await self.emit("agent.step.start", agent=self.name, step=step_num)

                step_result = await self.step()
                steps.append(step_result)

                await self.hooks.on_step_end(self.name, step_num)
                await self.emit("agent.step.end", agent=self.name, step=step_num)

                # Loop detection (after step hooks to ensure span closure)
                if step_result.action:
                    await self._check_loop(step_result.action)

                if step_result.response is not None:
                    final_output = step_result.response
                    self.context.state.transition(AgentState.FINISHED)
                    break
            else:
                # max_steps exceeded
                self.context.state.transition(AgentState.ERROR)
                raise MaxStepsExceededError(f"Agent '{self.name}' exceeded {self.max_steps} steps")

        except Exception as e:
            await self.hooks.on_error(self.name, e)
            await self.emit("agent.run.error", agent=self.name, error=str(e))
            if not isinstance(e, MaxStepsExceededError):
                if not self.context.state.is_terminal:
                    self.context.state.transition(AgentState.ERROR)
            raise

        finally:
            if resolved_session:
                now = datetime.now()
                ss = self.context.to_session_state(
                    resolved_session.session_id,
                    agent_name=self.name,
                )
                ss.created_at = self._session_created_at or now
                ss.updated_at = now
                tool_states = self.tool_registry.save_states()
                if tool_states:
                    ss.metadata["_tool_states"] = tool_states
                await resolved_session.save_state(ss)

        messages = await self.context.short_term_memory.get_context_messages()
        result = AgentResult(
            output=final_output,
            messages=messages,
            steps=steps,
            usage=self._total_usage,
        )

        await self.hooks.on_run_end(self.name, final_output)
        await self.emit("agent.run.end", agent=self.name, output=final_output, steps=len(steps))
        return result

    async def chat(
        self,
        *,
        session: str | BaseSession | None = None,
        prompt: str = "> ",
        exit_commands: tuple[str, ...] = ("exit", "quit", "bye"),
    ) -> None:
        """Interactive REPL with auto-trigger on background task completion."""
        import asyncio as _asyncio
        import readline  # noqa: F401 — enables arrow keys, history, proper backspace

        from agent_harness.utils.input_mux import mux_input

        if self._approval is not None:
            self._approval.reset_session()

        input_task: _asyncio.Task[str] | None = None

        try:
            while True:
                # Collect completed background tasks before waiting
                collected = await self._collect_background_results()
                if collected:
                    if input_task and not input_task.done():
                        input_task.cancel()
                        input_task = None
                    print("\n[Background task completed]")
                    try:
                        await self.run(
                            Message.system(
                                "[Background Task Notification] "
                                "Process the completed background task results.",
                                metadata={"is_background_result": True},
                            ),
                            session=session,
                        )
                    except Exception:
                        pass
                    continue

                # Only create new input task if previous one is done
                if input_task is None or input_task.done():
                    input_task = _asyncio.create_task(mux_input(prompt, priority=0))

                # Race: user input vs background completion
                wait_set: set[_asyncio.Task[Any]] = {input_task}
                bg_wait_task: _asyncio.Task[Any] | None = None
                if self._bg_manager.has_running():
                    bg_wait_task = _asyncio.create_task(self._bg_manager.wait_next())
                    wait_set.add(bg_wait_task)

                done, _ = await _asyncio.wait(
                    wait_set, return_when=_asyncio.FIRST_COMPLETED
                )

                # Cancel bg observer if it didn't fire
                if bg_wait_task and bg_wait_task not in done:
                    bg_wait_task.cancel()

                if input_task in done:
                    user_input = input_task.result().strip()
                    input_task = None
                    if not user_input:
                        continue
                    if user_input.lower() in exit_commands:
                        break
                    try:
                        result = await self.run(user_input, session=session)
                        if not self._stream:
                            print(result.output)
                    except Exception as e:
                        print(f"Error: {e}")

                elif bg_wait_task and bg_wait_task in done:
                    if input_task and not input_task.done():
                        input_task.cancel()
                        input_task = None
                    print("\n[Background task completed]")
                    await self._collect_background_results()
                    try:
                        await self.run(
                            Message.system(
                                "[Background Task Notification] "
                                "Process the completed background task results.",
                                metadata={"is_background_result": True},
                            ),
                            session=session,
                        )
                    except Exception:
                        pass

        except (KeyboardInterrupt, EOFError, _asyncio.CancelledError):
            pass

        # Cleanup: cancel pending input and background tasks
        if input_task and not input_task.done():
            input_task.cancel()
        if self._bg_manager.has_running():
            count = self._bg_manager.cancel_all()
            if count:
                print(f"\nCancelled {count} running background task(s).")

    @abstractmethod
    async def step(self) -> StepResult:
        """Execute a single reasoning step.

        Subclasses implement their strategy here:
        - ReActAgent: think -> act -> observe
        - PlanAgent: plan -> execute step
        - ConversationalAgent: generate response

        Returns:
            StepResult. If response is not None, the run loop ends.
        """
        ...

    async def call_llm(
        self,
        messages: list[Message] | None = None,
        tools: list[ToolSchema] | None = None,
        use_long_term: bool | None = None,
        long_term_query: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call the LLM with current context messages or provided messages.

        Args:
            messages: Override messages. If None, uses short-term memory.
            tools: Override tool schemas. If None, uses registered tools.
            use_long_term: Query long-term memory and inject results.
                If None, falls back to self.use_long_term_memory.
            long_term_query: Custom query for long-term retrieval.
            **kwargs: Passed through to the LLM call.
        """
        if use_long_term is None:
            use_long_term = self.use_long_term_memory

        # Notify before compression (if it will happen)
        compressor = self.context.short_term_memory.compressor
        if compressor and compressor.should_compress(
            self.context.short_term_memory._messages,
            self.context.short_term_memory.max_tokens,
        ):
            await self.hooks.on_compression_start(self.name)

        # Collect ephemeral context from stateful tools (sorted by context_order)
        extra_sys: list[Message] = []
        sorted_tools = sorted(self.tools, key=lambda t: t.context_order)
        for tool in sorted_tools:
            ctx_msg = tool.build_context_message()
            if ctx_msg:
                extra_sys.append(ctx_msg)

        # Runtime context (injected after tool context)
        runtime_msg = self._runtime_ctx.build_context_message()
        if runtime_msg:
            extra_sys.append(runtime_msg)

        messages = await self.context.build_llm_messages(
            base_messages=messages,
            include_working=True,
            include_long_term=use_long_term,
            long_term_query=long_term_query,
            extra_system_messages=extra_sys or None,
        )

        # Notify after compression (if it happened)
        if compressor:
            comp_result = compressor.take_last_result()
            if comp_result:
                await self.hooks.on_compression_end(
                    self.name,
                    comp_result.original_count,
                    comp_result.compressed_count,
                    comp_result.summary_tokens,
                )

        # Ephemeral loop warning (one-shot, appended at end for context)
        if self._pending_loop_warning:
            messages.append(self._pending_loop_warning)
            self._pending_loop_warning = None

        if tools is None and self.tool_schemas:
            tools = self.tool_schemas

        await self.hooks.on_llm_call(self.name, messages)
        if self.context.state.current != AgentState.THINKING:
            self.context.state.transition(AgentState.THINKING)

        if self._stream:

            async def _on_delta(delta: StreamDelta) -> None:
                await self.hooks.on_llm_stream_delta(self.name, delta)

            response = await self.llm.stream_with_events(
                messages,
                tools=tools,
                on_delta=_on_delta,
                **kwargs,
            )
        else:
            response = await self.llm.generate_with_events(messages, tools=tools, **kwargs)

        self._total_usage = self._total_usage + response.usage

        await self.context.short_term_memory.add_message(response.message)
        return response

    async def execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls with optional human approval."""
        self.context.state.transition(AgentState.ACTING)

        approved: list[ToolCall] = []
        denied_results: list[ToolResult] = []

        for tc in tool_calls:
            if self._approval is None:
                await self.hooks.on_tool_call(self.name, tc)
                approved.append(tc)
                continue

            # Resource extraction
            tool_obj = (
                self.tool_executor.registry.get(tc.name)
                if self.tool_executor.registry.has(tc.name)
                else None
            )
            resource_key = tool_obj.approval_resource_key if tool_obj else None
            default_val: str | None = None
            if tool_obj and resource_key:
                props = tool_obj.get_schema().parameters.get("properties", {})
                prop = props.get(resource_key, {})
                if "default" in prop:
                    default_val = str(prop["default"])
            resource, kind = extract_resource(
                tc.name,
                tc.arguments,
                resource_key,
                default=default_val,
            )

            action = self._approval.check(tc, resource=resource, kind=kind)

            if action == ApprovalAction.EXECUTE:
                await self.hooks.on_tool_call(self.name, tc)
                approved.append(tc)

            elif action == ApprovalAction.DENY:
                await self.hooks.on_approval_result(
                    self.name,
                    ApprovalResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        decision=ApprovalDecision.DENY,
                        reason="Not allowed by policy.",
                    ),
                )
                denied_results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        content=f"Tool '{tc.name}' is not allowed by policy.",
                        is_error=True,
                    )
                )

            else:  # ApprovalAction.ASK
                assert self._approval_handler is not None
                request = ApprovalRequest(
                    tool_call=tc,
                    agent_name=self.name,
                    resource=resource,
                    resource_kind=kind,
                )
                await self.hooks.on_approval_request(self.name, request)

                try:
                    result = await self._approval_handler.request_approval(request)
                except Exception as e:
                    logger.warning("Approval handler failed for '%s': %s", tc.name, e)
                    result = ApprovalResult(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        decision=ApprovalDecision.DENY,
                        reason=f"Approval handler error: {e}",
                    )

                await self.hooks.on_approval_result(self.name, result)

                if result.decision == ApprovalDecision.ALLOW_ONCE:
                    await self.hooks.on_tool_call(self.name, tc)
                    approved.append(tc)
                elif result.decision == ApprovalDecision.ALLOW_SESSION:
                    self._approval.grant_session(
                        tc.name,
                        resource=resource,
                        kind=kind,
                    )
                    await self.hooks.on_tool_call(self.name, tc)
                    approved.append(tc)
                else:
                    reason = result.reason or "Denied by user."
                    denied_results.append(
                        ToolResult(
                            tool_call_id=tc.id,
                            content=f"Tool '{tc.name}' was denied: {reason}",
                            is_error=True,
                        )
                    )

        results: list[ToolResult] = []
        if approved:
            results = await self.tool_executor.execute_batch(approved)

        all_results = denied_results + results
        result_map = {r.tool_call_id: r for r in all_results}
        ordered = [result_map[tc.id] for tc in tool_calls]

        self.context.state.transition(AgentState.OBSERVING)

        for r in ordered:
            await self.hooks.on_tool_result(self.name, r)
            await self.context.short_term_memory.add_message(
                Message.tool(
                    tool_call_id=r.tool_call_id,
                    content=r.content,
                    is_error=r.is_error,
                )
            )

        # Notify stateful tools after successful execution
        for tc in approved:
            tc_result = result_map.get(tc.id)
            if tc_result and not tc_result.is_error:
                tool = (
                    self.tool_registry.get(tc.name)
                    if self.tool_registry.has(tc.name)
                    else None
                )
                if tool:
                    await tool.notify_state(self.hooks, self.name)

        return ordered

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} tools={len(self.tools)}>"
