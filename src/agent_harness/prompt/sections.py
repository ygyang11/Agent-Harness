"""Standard prompt section factories and default builder."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_harness.prompt.system_builder import SystemPromptBuilder

from agent_harness.prompt.instructions import (
    discover_instruction_files,
    render_instructions,
)
from agent_harness.prompt.section import (
    ORDER_CUSTOM,
    ORDER_GUIDELINES,
    ORDER_INSTRUCTIONS,
    ORDER_INTRO,
    ORDER_SKILLS,
    ORDER_TOOLS,
    PromptSection,
)

# ---------------------------------------------------------------------------
# Default intro
# ---------------------------------------------------------------------------

DEFAULT_INTRO = """\
You are an interactive AI agent with access to a set of tools you can use \
to accomplish tasks.

You are highly capable and can help users with a wide range of tasks — \
coding, debugging, research, analysis, data processing, writing, and more. \
When given a task, use your available tools to gather information, take \
action, and deliver results.

Important:
- Use tools when you need external information or capabilities
- Think step by step about what you need to do, then take action
- You can call multiple tools at once if they are independent
- After receiving tool results, analyze them before deciding next steps
- When the task is complete, provide a clear summary of what was done \
and the outcome"""

# ---------------------------------------------------------------------------
# Tool supplements — filesystem and terminal content preserved verbatim
# ---------------------------------------------------------------------------

_TOOL_SUPPLEMENTS: dict[str, str] = {
    "filesystem": """\
## File Operations

You have access to dedicated filesystem tools for reading, writing, editing, \
and searching files.

### Following Conventions
- Read files before editing — understand existing content before making changes
- Mimic existing style, naming conventions, and patterns in the codebase
- Use list_dir, glob_files, or grep_files to locate files before reading them
- Use pagination (offset/limit) when reading large files
- It is better to speculatively read multiple files as a batch when exploring

### Prefer Dedicated Tools
When a dedicated filesystem tool can do the job, use it instead of terminal_tool:
- read_file instead of cat/head/tail
- list_dir instead of ls
- glob_files instead of find
- grep_files instead of grep/rg
- write_file/edit_file instead of echo/sed/awk
These tools provide structured output optimized for your context window.
Reserve terminal_tool for commands that have no dedicated tool equivalent.""",
    "terminal": """\
## Terminal Commands

You have access to a terminal_tool for running shell commands in a bash subprocess.

### When to Use
Use terminal_tool for operations without a dedicated tool:
- Version control: git add, git commit, git push, git diff, git log
- Testing: pytest, npm test, cargo test
- Package management: pip install, npm install, cargo build
- Build tools: make, cmake, gradle
- Run scripts: python script.py, node app.js, bash setup.sh, ./run.sh
- System commands: docker, kubectl, curl, wget

When dedicated tools exist for an operation (e.g. file reading, searching),
prefer those over terminal_tool — they provide structured output optimized
for your context window.

### Command Execution
- Each call spawns a fresh bash subprocess — shell state (variables, cwd, aliases) \
does not persist between calls
- Commands always start from the workspace root directory
- To run in a subdirectory, chain with cd: 'cd src && pytest'
- Always quote file paths containing spaces with double quotes
- If a command creates files or directories, verify the parent directory exists first

### Multiple Commands
- Use && to chain dependent commands (second runs only if first succeeds)
- Use ; to chain independent commands (runs regardless of previous exit code)
- For independent operations, prefer making separate terminal_tool calls over long chains

### Git Safety
- Never skip hooks (--no-verify) unless explicitly asked
- Never force push or reset --hard without explicit user permission
- Prefer creating new commits over amending existing ones
- Always check git status before committing
- Before running destructive operations, consider whether there is a safer alternative""",
}

_FS_TOOL_NAMES = frozenset(
    {
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "glob_files",
        "grep_files",
    }
)


def _has_tool(ctx: dict[str, Any], name: str) -> bool:
    return any(getattr(t, "name", "") == name for t in ctx.get("tools", []))


def _has_any_tool(ctx: dict[str, Any], names: frozenset[str]) -> bool:
    return any(getattr(t, "name", "") in names for t in ctx.get("tools", []))


# ---------------------------------------------------------------------------
# Section factories
# ---------------------------------------------------------------------------


def make_intro_section(content: str) -> PromptSection:
    """Create the intro section with custom content."""
    return PromptSection(name="intro", order=ORDER_INTRO, content=content)


def make_guidelines_section() -> PromptSection:
    """Guidelines — tool-usage principles. Only active when agent has tools.

    Pure-reasoning agents (PlannerAgent, ConversationalAgent judge) do not
    see these guidelines, as they add noise to non-tool-using contexts.
    """
    return PromptSection(
        name="guidelines",
        order=ORDER_GUIDELINES,
        content="""\
# Guidelines

## Doing Tasks

- In general, do not propose changes to code you haven't read. \
Read and understand existing code before suggesting modifications.
- Do not create files unless they're absolutely necessary for achieving \
your goal. Prefer editing existing files over creating new ones.
- Don't add features, refactor code, or make "improvements" beyond what \
was asked. A bug fix doesn't need surrounding code cleaned up.
- Don't add error handling, fallbacks, or validation for scenarios that \
can't happen. Only validate at system boundaries (user input, external APIs).
- Don't create helpers, utilities, or abstractions for one-time operations. \
Three similar lines of code is better than a premature abstraction.
- Be careful not to introduce security vulnerabilities such as command \
injection, XSS, SQL injection, and other OWASP top 10 vulnerabilities.

## Executing Actions with Care

Carefully consider the reversibility and blast radius of actions. \
You can freely take local, reversible actions like editing files or \
running tests. But for actions that are hard to reverse, affect shared \
systems, or could be destructive, confirm with the user before proceeding.

Examples of risky actions that warrant confirmation:
- Destructive operations: deleting files/branches, dropping database \
tables, overwriting uncommitted changes
- Hard-to-reverse operations: force-pushing, git reset --hard, \
amending published commits
- Actions visible to others: pushing code, creating/closing PRs or issues""",
        condition=lambda ctx: bool(ctx.get("tools")),
    )


def make_tools_section() -> PromptSection:
    """Dynamic tool supplement based on registered tools."""

    def _resolve(ctx: dict[str, Any]) -> str:
        parts: list[str] = []
        if _has_any_tool(ctx, _FS_TOOL_NAMES):
            parts.append(_TOOL_SUPPLEMENTS["filesystem"])
        if _has_tool(ctx, "terminal_tool"):
            parts.append(_TOOL_SUPPLEMENTS["terminal"])
        if not parts:
            return ""
        return "# Using Your Tools\n\n" + "\n\n".join(parts)

    return PromptSection(name="tools", order=ORDER_TOOLS, content=_resolve)


def make_skills_section() -> PromptSection:
    """Skills section — professional skill usage instructions."""

    def _resolve(ctx: dict[str, Any]) -> str:
        skill_loader = ctx.get("skill_loader")
        if skill_loader is None:
            return ""
        catalog = skill_loader.get_catalog()
        if catalog == "(no skills available)":
            return ""
        return (
            "# Skills\n\n"
            "You have access to a `skill_tool` that loads domain-specific "
            "instructions for specialized tasks. Skills provide structured "
            "workflow guidance beyond your training data.\n\n"
            "<available_skills>\n"
            f"{catalog}\n"
            "</available_skills>\n\n"
            "When to use:\n"
            "- A user request clearly matches a skill description listed above\n"
            "- You need domain-specific workflow guidance not in your "
            "training data\n\n"
            "Rules:\n"
            "- Before responding to the user, check if the request matches "
            "an available skill. If a match exists, load the skill first — "
            "its instructions provide more thorough guidance than your "
            "default behavior\n"
            "- Do not skip a matching skill because you believe you can "
            "handle the task without it\n"
            "- Use skill names exactly as listed above\n"
            "- After loading a skill, follow its instructions to complete "
            "the task\n"
            "- Do not call skill_tool for skills that are already loaded "
            "in this conversation\n"
            "- If no skill matches, proceed normally with your available tools"
        )

    return PromptSection(
        name="skills",
        order=ORDER_SKILLS,
        content=_resolve,
        condition=lambda ctx: _has_tool(ctx, "skill_tool"),
    )


def make_instructions_section() -> PromptSection:
    """Instructions section — discovers AGENTS.md / CLAUDE.md files."""

    def _resolve(ctx: dict[str, Any]) -> str:
        cwd = Path(ctx.get("cwd") or Path.cwd())
        files = discover_instruction_files(cwd)
        if not files:
            return ""
        return f"# Project Instructions\n\n{render_instructions(files)}"

    return PromptSection(name="instructions", order=ORDER_INSTRUCTIONS, content=_resolve)


def make_custom_section(content: str) -> PromptSection:
    """Create a custom section for user-provided additional content."""
    return PromptSection(name="custom", order=ORDER_CUSTOM, content=content)


# ---------------------------------------------------------------------------
# Builder factory
# ---------------------------------------------------------------------------


def create_default_builder(system_prompt: str = "") -> SystemPromptBuilder:
    """Create a SystemPromptBuilder with all standard sections.

    All sections are registered by default. When system_prompt is provided,
    it replaces the DEFAULT_INTRO as the intro section content.
    """
    from agent_harness.prompt.system_builder import SystemPromptBuilder

    builder = SystemPromptBuilder()

    intro = system_prompt if system_prompt else DEFAULT_INTRO
    builder.register(make_intro_section(intro))
    builder.register(make_guidelines_section())
    builder.register(make_tools_section())
    builder.register(make_skills_section())
    builder.register(make_instructions_section())

    return builder
