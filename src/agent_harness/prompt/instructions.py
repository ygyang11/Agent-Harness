"""Instruction file discovery — loads AGENTS.md / CLAUDE.md from workspace."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_FILE_CHARS = 4_000
MAX_TOTAL_CHARS = 12_000

_CANDIDATES = ("AGENTS.md", "CLAUDE.md")

_SCOPE_LABELS: dict[str, str] = {
    "global": "global instructions, applies to all projects",
    "project": "project instructions, checked into the codebase",
    "parent": "parent directory instructions",
}


@dataclass(frozen=True)
class InstructionFile:
    """A discovered instruction file with scope metadata."""

    path: Path
    content: str
    scope: str  # "global" | "project" | "parent"


def discover_instruction_files(cwd: Path | None = None) -> list[InstructionFile]:
    """Discover instruction files from global dir and cwd upward to root.

    Discovery order:
      1. ~/.agent_harness/ (global scope)
      2. cwd upward to filesystem root (project/parent scope)

    Deduplicates by content hash. Orders global -> root -> cwd.
    """
    if cwd is None:
        cwd = Path.cwd()
    cwd = cwd.resolve()

    raw: list[tuple[Path, str, str]] = []

    # 1. Global: ~/.agent_harness/
    global_dir = Path.home() / ".agent_harness"
    if global_dir.is_dir():
        for candidate in _CANDIDATES:
            path = global_dir / candidate
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                if content.strip():
                    raw.append((path, content, "global"))

    # 2. Traverse cwd upward to root
    upward: list[tuple[Path, str, str]] = []
    current = cwd
    while True:
        for candidate in _CANDIDATES:
            path = current / candidate
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                if content.strip():
                    scope = "project" if current == cwd else "parent"
                    upward.append((path, content, scope))
        parent = current.parent
        if parent == current:
            break
        current = parent

    upward.reverse()  # root -> cwd
    raw.extend(upward)

    # Deduplicate by content hash
    seen: set[str] = set()
    files: list[InstructionFile] = []
    for path, content, scope in raw:
        h = hashlib.md5(content.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        files.append(InstructionFile(path=path, content=content, scope=scope))

    return files


def render_instructions(files: list[InstructionFile]) -> str:
    """Render instruction files with scope labels and budget enforcement.

    Single file: truncate at MAX_FILE_CHARS.
    Total: stop at MAX_TOTAL_CHARS (truncated files still fit if within budget).
    """
    if not files:
        return ""

    parts: list[str] = []
    total = 0

    for f in files:
        content = f.content
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + "\n\n[truncated]"

        if total + len(content) > MAX_TOTAL_CHARS:
            parts.append(
                "_Additional instruction content omitted after reaching the prompt budget._"
            )
            break

        scope_label = _SCOPE_LABELS.get(f.scope, f.scope)
        parts.append(f"Contents of {f.path} ({scope_label}):\n\n{content}")
        total += len(content)

    return "\n\n---\n\n".join(parts)
