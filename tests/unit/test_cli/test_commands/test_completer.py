from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from agent_cli.commands.base import Command, CommandContext, CommandResult
from agent_cli.commands.registry import CommandRegistry
from agent_cli.repl.completer import build_command_completer


async def _noop(ctx: CommandContext, args: str) -> CommandResult:
    return CommandResult()


def _registry_with(*names: str) -> CommandRegistry:
    r = CommandRegistry()
    for n in names:
        r.register_command(Command(name=n, description=f"desc {n}", handler=_noop))
    return r


def _completions(completer, text: str) -> list[str]:
    doc = Document(text=text, cursor_position=len(text))
    return [c.text for c in completer.get_completions(doc, CompleteEvent())]


def test_no_completion_on_plain_text() -> None:
    c = build_command_completer(_registry_with("/clear", "/compact"))
    assert _completions(c, "explain this") == []
    assert _completions(c, "cl") == []
    assert _completions(c, "") == []


def test_slash_prefix_shows_all_commands() -> None:
    c = build_command_completer(_registry_with("/clear", "/compact", "/exit"))
    out = _completions(c, "/")
    assert set(out) == {"/clear", "/compact", "/exit"}


def test_slash_prefix_narrows_with_typed_letters() -> None:
    c = build_command_completer(_registry_with("/clear", "/compact", "/exit"))
    out = _completions(c, "/cl")
    assert out == ["/clear"]


def test_fuzzy_match_under_slash() -> None:
    c = build_command_completer(_registry_with("/compact", "/clear"))
    assert "/compact" in _completions(c, "/cmpct")


def test_no_completion_after_space() -> None:
    c = build_command_completer(_registry_with("/compact"))
    assert _completions(c, "/compact ") == []
    assert _completions(c, "/compact focus") == []
