"""Resource-aware approval policy engine."""
from __future__ import annotations

import os
import re
from urllib.parse import urlparse

from agent_harness.approval.rules import (
    any_rule_matches,
    has_tool_level_rule,
    parse_rules,
)
from agent_harness.approval.types import ApprovalAction
from agent_harness.core.message import ToolCall

_CHAIN_RE = re.compile(r"\s*(?:&&|\|\||[;|])\s*")
_UNSAFE_SHELL_RE = re.compile(
    r"`"
    r"|\$\("
    r"|\$'"
    r"|\$\{"
    r"|[<>]"
    r"|(?<!&)&(?!&)"
    r"|\n"
)


def derive_session_prefix(resource: str, kind: str) -> str:
    """Derive a session grant prefix from a concrete resource.

    path:    "src/utils/helper.py" -> "src/utils" (parent directory)
    url:     "https://github.com/repo" -> "github.com" (hostname)
    command: "git status" -> "git" (first word)
    """
    if kind == "path":
        parent = os.path.dirname(os.path.normpath(resource))
        return parent if parent else resource
    if kind == "url":
        try:
            return urlparse(resource).hostname or resource
        except Exception:
            return resource
    if kind == "command":
        return resource.split()[0] if resource.strip() else resource
    return resource


class ApprovalPolicy:
    """Resource-aware approval policy engine.

    Combines deny/allow rule evaluation with session-level grants.
    Supports segment-aware checking for chained shell commands.
    """

    def __init__(
        self,
        *,
        mode: str = "auto",
        always_allow: set[str] | None = None,
        always_deny: set[str] | None = None,
    ) -> None:
        self._mode = mode
        self._deny_rules = parse_rules(always_deny or set())
        self._allow_rules = parse_rules(always_allow or set())
        self._session_grants: dict[str, None | set[tuple[str, str]]] = {}

    def check(
        self,
        tool_call: ToolCall,
        resource: str | None = None,
        kind: str | None = None,
    ) -> ApprovalAction:
        """Resource-aware approval check."""
        if self._mode == "never":
            return ApprovalAction.EXECUTE

        name = tool_call.name

        if kind == "command" and resource is not None:
            return self._check_command(name, resource)

        return self._check_generic(name, resource)

    def grant_session(
        self,
        tool_name: str,
        resource: str | None = None,
        kind: str | None = None,
    ) -> None:
        """Grant session-level approval."""
        if resource is None or kind is None:
            self._session_grants[tool_name] = None
            return
        if tool_name in self._session_grants and self._session_grants[tool_name] is None:
            return

        bucket = self._session_grants.setdefault(tool_name, set())
        assert isinstance(bucket, set)

        if kind == "command":
            segments = [s.strip() for s in _CHAIN_RE.split(resource) if s.strip()]
            for seg in segments:
                prefix = derive_session_prefix(seg, kind)
                bucket.add((prefix, kind))
        else:
            prefix = derive_session_prefix(resource, kind)
            bucket.add((prefix, kind))

    def reset_session(self) -> None:
        """Clear all session-level grants."""
        self._session_grants.clear()

    # ── internal ──

    def _check_generic(self, name: str, resource: str | None) -> ApprovalAction:
        """deny > allow > session > ASK."""
        if any_rule_matches(self._deny_rules, name, resource):
            return ApprovalAction.DENY
        if any_rule_matches(self._allow_rules, name, resource):
            return ApprovalAction.EXECUTE
        if self._session_matches(name, resource):
            return ApprovalAction.EXECUTE
        return ApprovalAction.ASK

    def _check_command(self, name: str, command: str) -> ApprovalAction:
        """Segment-aware check for chained commands.

        Unsafe shell patterns fall back to tool-level, but deny rules are
        still checked against the full command string first.
        """
        if _UNSAFE_SHELL_RE.search(command):
            if any_rule_matches(self._deny_rules, name, command):
                return ApprovalAction.DENY
            return self._check_generic(name, None)

        segments = [s.strip() for s in _CHAIN_RE.split(command) if s.strip()]
        if not segments:
            return self._check_generic(name, None)

        for seg in segments:
            if any_rule_matches(self._deny_rules, name, seg):
                return ApprovalAction.DENY

        if has_tool_level_rule(self._allow_rules, name):
            return ApprovalAction.EXECUTE

        if all(
            any_rule_matches(self._allow_rules, name, seg)
            or self._segment_in_session(name, seg)
            for seg in segments
        ):
            return ApprovalAction.EXECUTE

        return ApprovalAction.ASK

    def _session_matches(self, name: str, resource: str | None) -> bool:
        """Generic session grant matching (path / url / tool-level)."""
        if name not in self._session_grants:
            return False
        grants = self._session_grants[name]
        if grants is None:
            return True
        if resource is None:
            return False
        for prefix, kind in grants:
            if kind == "path":
                normed = os.path.normpath(resource)
                if normed == prefix or normed.startswith(prefix + os.sep):
                    return True
            elif kind == "url":
                try:
                    if (urlparse(resource).hostname or "") == prefix:
                        return True
                except Exception:
                    pass
        return False

    def _segment_in_session(self, name: str, segment: str) -> bool:
        """Check if a single command segment is covered by session grants."""
        if name not in self._session_grants:
            return False
        grants = self._session_grants[name]
        if grants is None:
            return True
        cmd_prefixes = {p for p, k in grants if k == "command"}
        first_word = segment.split()[0] if segment.strip() else ""
        return first_word in cmd_prefixes
