from types import SimpleNamespace
from unittest.mock import MagicMock

from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.completion import Completion
from prompt_toolkit.document import Document

from agent_cli.repl.keybindings import build_keybindings


def _find_handler(kb: object, key: str):
    for binding in kb.bindings:
        values = tuple(getattr(k, "value", str(k)) for k in binding.keys)
        if values == (key,):
            return binding.handler
    raise AssertionError(f"binding for {key!r} not found")


def test_bindings_registered() -> None:
    kb = build_keybindings()
    # Tab + Ctrl+C + Ctrl+D + Alt+Enter at minimum
    assert len(kb.bindings) >= 4


def test_keys_cover_expected_set() -> None:
    kb = build_keybindings()
    values = {tuple(getattr(k, "value", str(k)) for k in b.keys) for b in kb.bindings}
    assert ("c-i",) in values
    assert ("c-c",) in values
    assert ("c-d",) in values
    # Alt+Enter registers as (Escape, ControlM) after prompt_toolkit normalization.
    assert any("escape" in t and "c-m" in t for t in values)


def test_tab_applies_current_completion() -> None:
    kb = build_keybindings()
    handler = _find_handler(kb, "c-i")
    buf = Buffer(document=Document("@sr", cursor_position=3))
    completion = Completion("src/", start_position=-2)
    buf.complete_state = CompletionState(
        original_document=buf.document,
        completions=[completion],
        complete_index=0,
    )

    handler(SimpleNamespace(current_buffer=buf))

    assert buf.text == "@src/"


def test_tab_selects_first_completion_when_menu_is_open_without_selection() -> None:
    kb = build_keybindings()
    handler = _find_handler(kb, "c-i")
    buf = Buffer(document=Document("/cl", cursor_position=3))
    completion = Completion("/clear", start_position=-3)
    buf.complete_state = CompletionState(
        original_document=buf.document,
        completions=[completion],
    )

    handler(SimpleNamespace(current_buffer=buf))

    assert buf.text == "/clear"


def test_tab_starts_completion_when_menu_is_closed() -> None:
    kb = build_keybindings()
    handler = _find_handler(kb, "c-i")
    buf = MagicMock()
    buf.complete_state = None

    handler(SimpleNamespace(current_buffer=buf))

    buf.start_completion.assert_called_once_with(select_first=True)
