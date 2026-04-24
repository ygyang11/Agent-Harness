import io

from rich.console import Console

from agent_cli.render.markdown_stream import MarkdownStream
from agent_cli.theme import FLEXOKI_DARK


def _stream() -> tuple[MarkdownStream, io.StringIO]:
    buf = io.StringIO()
    con = Console(file=buf, width=80, force_terminal=True, color_system=None)
    ms = MarkdownStream(con, FLEXOKI_DARK)
    ms._min_delay = 0.0
    return ms, buf


def test_update_buffers_delta() -> None:
    ms, _ = _stream()
    ms.update("hel")
    ms.update("lo")
    assert ms._buffer == "hello"


def test_update_with_empty_delta_is_noop() -> None:
    ms, buf = _stream()
    ms.update("")
    assert ms._buffer == ""
    assert ms._live is None
    assert buf.getvalue() == ""


def test_finalize_on_empty_stream_is_noop() -> None:
    ms, buf = _stream()
    ms.finalize()
    assert ms._live is None
    assert ms._buffer == ""
    assert buf.getvalue() == ""


def test_finalize_clears_buffer_and_stops_live() -> None:
    ms, _ = _stream()
    ms.update("# Hello\n\nBody text.\n")
    assert ms._live is not None
    ms.finalize()
    assert ms._live is None
    assert ms._buffer == ""
    assert ms._printed == []


def test_finalize_promotes_all_lines_to_scrollback() -> None:
    ms, buf = _stream()
    body = "# Heading\n\nLine one.\nLine two.\nLine three.\n"
    ms.update(body)
    ms.finalize()
    out = buf.getvalue()
    assert "Heading" in out
    assert "Line one" in out
    assert "Line two" in out
    assert "Line three" in out


def test_pause_commits_like_finalize() -> None:
    ms, buf = _stream()
    ms.update("# Hello\n\nStreaming...\n")
    ms.pause()
    assert ms._live is None
    assert ms._buffer == ""
    assert ms._printed == []
    assert "Hello" in buf.getvalue()


def test_finalize_is_idempotent() -> None:
    ms, _ = _stream()
    ms.update("# Content")
    ms.finalize()
    ms.finalize()
    assert ms._live is None
    assert ms._buffer == ""


def test_reset_allows_second_stream_after_finalize() -> None:
    ms, buf = _stream()
    ms.update("# First\n")
    ms.finalize()
    ms.update("# Second\n")
    assert ms._live is not None
    ms.finalize()
    out = buf.getvalue()
    assert "First" in out
    assert "Second" in out


def test_stable_prefix_promoted_when_exceeding_live_window() -> None:
    ms, buf = _stream()
    many_items = "\n".join(f"- Item {i}" for i in range(20)) + "\n"
    ms.update(many_items)
    # Non-final render promotes lines past LIVE_WINDOW (6) to scrollback.
    assert len(ms._printed) > 0
    # The earliest items must already be in the real console's output.
    out = buf.getvalue()
    assert "Item 0" in out
    ms.finalize()


def test_first_line_has_assistant_marker_prefix() -> None:
    ms, _ = _stream()
    lines = ms._render_to_ansi_lines("# Hello")
    assert len(lines) >= 1
    # First line carries the "●" marker
    assert "●" in lines[0]
    # Content follows at 4-column indent; the Markdown rendering of "# Hello"
    # contains the word "Hello" somewhere on the first line.
    assert "Hello" in lines[0]


def test_continuation_lines_have_two_space_prefix() -> None:
    ms, _ = _stream()
    lines = ms._render_to_ansi_lines("- a\n- b\n- c\n")
    assert len(lines) >= 2
    for line in lines[1:]:
        stripped = line.lstrip("\x1b[0m")
        assert stripped.startswith("  "), f"missing 2-space indent: {line!r}"


def test_offline_render_width_reserves_indent_cols() -> None:
    ms, _ = _stream()
    # Terminal width=80; offline renders at 78 (reserves 2 for prefix). A
    # single 80-char line should wrap into >1 line.
    lines = ms._render_to_ansi_lines("x" * 80)
    assert len(lines) >= 2


def test_short_update_does_not_drop_head_lines() -> None:
    # Regression: when total rendered lines < LIVE_WINDOW, `num_lines` used
    # to go negative and `lines[num_lines:]` sliced from the tail, dropping
    # head lines until more content arrived. Hits hardest at 4-5 lines.
    from unittest.mock import MagicMock
    ms, _ = _stream()

    fake_lines = [f"line{i}\n" for i in range(4)]
    ms._render_to_ansi_lines = MagicMock(return_value=fake_lines)  # type: ignore[method-assign]
    ms._buffer = "anything"
    ms._render(final=False)
    assert ms._live is not None

    captured: list[str] = []
    ms._live.update = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda r, **k: captured.append(r.plain),
    )
    # Bypass debounce for the second render call.
    ms._last_update_ts = 0.0
    ms._render(final=False)

    assert captured, "live.update must be called on subsequent render"
    tail_plain = captured[-1]
    for i in range(4):
        assert f"line{i}" in tail_plain, f"line{i} dropped: {tail_plain!r}"
    ms.finalize()


def test_abort_drops_live_tail_without_promoting_to_scrollback() -> None:
    ms, _ = _stream()
    ms.update("Partial answer that should be discarded.")
    # Short text fits entirely in the Live window — nothing was promoted yet.
    assert ms._printed == []

    ms.abort()

    assert ms._buffer == ""
    assert ms._printed == []
    assert ms._live is None


def test_abort_preserves_promoted_lines_drops_only_live_tail() -> None:
    ms, buf = _stream()
    long_body = "\n\n".join(f"Promoted paragraph {i}." for i in range(10))
    ms.update(long_body + "\n\nUncommitted tail paragraph.")

    promoted_snapshot = list(ms._printed)
    assert promoted_snapshot, "expected promote to have occurred for long text"
    before_len = len(buf.getvalue())

    ms.abort()

    # abort writes nothing new to the console (no promote), only stops Live.
    # The Live.stop output (cursor-show etc.) is acceptable; assert no new
    # promoted lines were committed by abort.
    assert ms._buffer == ""
    assert ms._printed == []
    assert ms._live is None
    # Scrollback (buf) still contains the paragraphs promoted before abort.
    after_text = buf.getvalue()
    assert "Promoted paragraph 0." in after_text
    # Any write that happened during abort is bounded to Live-stop cleanup,
    # not a full promote of the remaining tail.
    new_bytes = after_text[before_len:]
    assert "Uncommitted tail paragraph." not in new_bytes


def test_abort_idempotent_when_live_never_started() -> None:
    ms, buf = _stream()

    ms.abort()

    assert ms._live is None
    assert buf.getvalue() == ""
