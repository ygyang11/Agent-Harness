"""Main REPL key bindings: Ctrl+C (single/double), Ctrl+D, Alt+Enter."""
from __future__ import annotations

import time

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent

_CTRL_C_DOUBLE_WINDOW_S = 2.0

# Module-level so it can be reset between prompts; otherwise a stale first-click
# timestamp can trigger a false double-click exit across prompt boundaries.
_ctrl_c_state: list[float] = [0.0]

_HINT = "\x1b[2m  ⎋ Ctrl+C again or /exit\x1b[0m\n"


def reset_ctrl_c_state() -> None:
    _ctrl_c_state[0] = 0.0


def build_keybindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("tab")
    def _(event: KeyPressEvent) -> None:
        buf = event.current_buffer
        state = buf.complete_state
        if state is not None:
            completion = state.current_completion
            if completion is None:
                buf.complete_next()
                state = buf.complete_state
                completion = state.current_completion if state is not None else None
            if completion is not None:
                buf.apply_completion(completion)
                return
        buf.start_completion(select_first=True)

    @kb.add("c-c")
    def _(event: KeyPressEvent) -> None:
        buf = event.current_buffer
        if buf.text:
            buf.reset()
            _ctrl_c_state[0] = 0.0
            return
        now = time.monotonic()
        if now - _ctrl_c_state[0] < _CTRL_C_DOUBLE_WINDOW_S:
            # Use EOFError (not KeyboardInterrupt) so asyncio.Task doesn't
            # re-raise a BaseException and abort the event loop.
            event.app.exit(exception=EOFError)
            return
        _ctrl_c_state[0] = now
        event.app.output.write_raw(_HINT)
        event.app.invalidate()

    @kb.add("c-d")
    def _(event: KeyPressEvent) -> None:
        if not event.current_buffer.text:
            event.app.exit(exception=EOFError)

    @kb.add("escape", "enter")
    def _(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    return kb
