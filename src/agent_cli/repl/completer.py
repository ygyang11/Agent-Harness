"""Slash-command completer: only activates while typing a `/`-prefixed command name."""
from __future__ import annotations

from collections.abc import Iterable

from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    FuzzyCompleter,
    WordCompleter,
)
from prompt_toolkit.document import Document

from agent_cli.commands.registry import CommandRegistry


class _SlashGatedCompleter(Completer):
    """Forward to the inner completer only when the pending input is a
    ``/``-prefixed token with no space yet"""

    def __init__(self, inner: Completer) -> None:
        self._inner = inner

    def get_completions(
        self, document: Document, complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/") or " " in text:
            return
        yield from self._inner.get_completions(document, complete_event)


def build_command_completer(registry: CommandRegistry) -> Completer:
    pairs = registry.get_completions()
    words = [name for name, _ in pairs]
    meta = {name: desc for name, desc in pairs}
    base = WordCompleter(words=words, meta_dict=meta, WORD=True)
    return _SlashGatedCompleter(FuzzyCompleter(base, WORD=True))
