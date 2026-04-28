"""/model — switch LLM model for the current process."""
from __future__ import annotations

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.ui import err, info, ok
from agent_cli.runtime import session as sess
from agent_harness.llm import create_llm


async def handle(ctx: CommandContext, args: str) -> CommandResult:
    new_model = args.strip()
    agent = ctx.agent
    if not new_model:
        return CommandResult(output=info(
            "Current model: ",
            (agent.llm.model_name, "bold"),
        ))

    config = agent.context.config
    # Clone the full current LLMConfig and only swap `model` — preserves
    # temperature / max_tokens / reasoning_effort etc. The framework's
    # `create_llm(config, model_override=...)` drops those fields, which would
    # silently reset user-configured inference parameters.
    try:
        new_cfg = config.llm.model_copy(update={"model": new_model})
        new_llm = create_llm(new_cfg)
    except Exception as e:
        return CommandResult(output=err(f"Failed to switch model: {e}"))

    new_llm.set_event_bus(agent.context.event_bus)
    agent.llm = new_llm
    agent.context.short_term_memory.model = new_model
    agent.context.short_term_memory.clear_call_snapshot()

    sess.update_compressor_model(agent, new_model, new_llm)

    config.llm.model = new_model
    return CommandResult(output=ok(
        "Model switched to ",
        (new_model, "bold"),
    ))


CMD = Command(
    name="/model",
    description="Switch LLM model for this session",
    handler=handle,
)
