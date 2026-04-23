from agent_cli.repl.keybindings import build_keybindings


def test_bindings_registered() -> None:
    kb = build_keybindings()
    # Ctrl+C + Ctrl+D + Alt+Enter at minimum
    assert len(kb.bindings) >= 3


def test_keys_cover_expected_set() -> None:
    kb = build_keybindings()
    values = {tuple(getattr(k, "value", str(k)) for k in b.keys) for b in kb.bindings}
    assert ("c-c",) in values
    assert ("c-d",) in values
    # Alt+Enter registers as (Escape, ControlM) after prompt_toolkit normalization.
    assert any("escape" in t and "c-m" in t for t in values)
