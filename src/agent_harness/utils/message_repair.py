"""Message history repair utilities."""
from __future__ import annotations

import logging

from agent_harness.core.message import Message, Role

logger = logging.getLogger(__name__)

_DANGLING_CONTENT = "[Tool result missing due to internal error]"


def patch_dangling_tool_calls(messages: list[Message]) -> list[Message]:
    """Ensure every tool_call has a matching tool_result, and vice versa.

    Repairs two types of orphans:
    1. Dangling tool_calls: assistant message has tool_calls but no
       corresponding tool result message exists -- inject synthetic error result.
    2. Orphaned tool_results: tool message references a tool_call_id that
       doesn't exist in any assistant message -- remove it.

    Returns a new list; does not mutate the input.
    """
    # Pass 1: collect all tool_call_ids from assistant messages
    all_tool_call_ids: set[str] = set()
    for msg in messages:
        if msg.role == Role.ASSISTANT and msg.tool_calls:
            for tc in msg.tool_calls:
                all_tool_call_ids.add(tc.id)

    # Pass 2: collect all tool_call_ids that have results
    resolved_ids: set[str] = set()
    for msg in messages:
        if msg.role == Role.TOOL and msg.tool_result:
            resolved_ids.add(msg.tool_result.tool_call_id)

    # Identify problems
    dangling_ids = all_tool_call_ids - resolved_ids
    orphaned_ids = resolved_ids - all_tool_call_ids

    if not dangling_ids and not orphaned_ids:
        return messages  # no-op fast path, return original reference

    if dangling_ids:
        logger.debug(
            "Patching %d dangling tool call(s): %s",
            len(dangling_ids),
            dangling_ids,
        )
    if orphaned_ids:
        logger.debug(
            "Removing %d orphaned tool result(s): %s",
            len(orphaned_ids),
            orphaned_ids,
        )

    # Pass 3: rebuild message list
    # Strategy for assistant messages with tool_calls:
    #   1. Append the assistant message
    #   2. Forward-consume consecutive TOOL messages, keep only those
    #      belonging to this assistant (filter by expected_ids)
    #   3. Append synthetic results for dangling IDs at the end
    result: list[Message] = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        # Remove orphaned tool results (not adjacent to any assistant)
        if msg.role == Role.TOOL and msg.tool_result:
            if msg.tool_result.tool_call_id in orphaned_ids:
                i += 1
                continue

        result.append(msg)

        # After assistant with tool_calls: consume results block, then patch
        if msg.role == Role.ASSISTANT and msg.tool_calls:
            expected_ids = {tc.id for tc in msg.tool_calls}
            pending = [tc.id for tc in msg.tool_calls if tc.id in dangling_ids]
            # Forward past consecutive TOOL messages, keep only matching ones
            j = i + 1
            while j < len(messages) and messages[j].role == Role.TOOL:
                tool_msg = messages[j]
                if (
                    tool_msg.tool_result
                    and tool_msg.tool_result.tool_call_id in expected_ids
                ):
                    result.append(tool_msg)
                j += 1
            # Append synthetics after all existing results
            for tc_id in pending:
                result.append(
                    Message.tool(
                        tool_call_id=tc_id,
                        content=_DANGLING_CONTENT,
                        is_error=True,
                    )
                )
            i = j
            continue

        i += 1

    return result
