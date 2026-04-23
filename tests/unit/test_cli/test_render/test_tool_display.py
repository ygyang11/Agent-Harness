import io
from unittest.mock import MagicMock

from rich.console import Console

from agent_cli.render.tool_display import (
    ToolDisplay,
    _format_call_line,
    _format_result_line,
    _ToolRow,
)
from agent_cli.theme import SPINNER_STATIC


def _mock_live_display() -> ToolDisplay:
    d = ToolDisplay(MagicMock())
    d._open_live = MagicMock(side_effect=lambda: setattr(d, "_live", MagicMock()))
    d._close_live = MagicMock(side_effect=lambda: setattr(d, "_live", None))
    return d


def _read_result(content: str) -> MagicMock:
    return MagicMock(tool_call_id="t1", content=content, is_error=False)


def test_add_call_opens_live_and_appends_row() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={"file_path": "a.py"})
    tc.name = "read_file"
    d.add_call(tc)
    assert len(d._rows) == 1
    assert d._rows[0].status == "running"
    d._open_live.assert_called_once()


def test_mark_result_done_status_and_summary_formats() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={"file_path": "a.py"})
    tc.name = "read_file"
    d.add_call(tc)
    d.mark_result(_read_result("[a.py] lines 1-3 of 3\n1\tfoo\n2\tbar\n3\tbaz"))
    row = d._rows[0]
    assert row.status == "done"
    line = _format_result_line(row)
    assert line is not None and "Read 3 lines" in line.plain


def test_mark_result_error_summary_is_error_prefixed() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={})
    tc.name = "terminal_tool"
    d.add_call(tc)
    d.mark_result(MagicMock(
        tool_call_id="t1", content="command failed: permission", is_error=True,
    ))
    row = d._rows[0]
    assert row.status == "error"
    line = _format_result_line(row)
    assert line is not None and line.plain.lstrip().startswith("⎿  Error:")


def test_error_content_string_without_flag_is_detected() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={"file_path": "x"})
    tc.name = "read_file"
    d.add_call(tc)
    d.mark_result(MagicMock(
        tool_call_id="t1",
        content="Error: Permission denied: /etc/shadow",
        is_error=False,
    ))
    row = d._rows[0]
    assert row.status == "error"
    line = _format_result_line(row)
    assert line is not None
    assert "Error:" in line.plain and "Permission denied" in line.plain


def test_args_preview_strips_newlines() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={"pattern": "foo\nbar"})
    tc.name = "grep_files"
    d.add_call(tc)
    assert "\n" not in d._rows[0].args_preview


def test_args_preview_filters_to_primary_keys() -> None:
    d = _mock_live_display()
    tc = MagicMock(
        id="t1",
        arguments={"file_path": "a.py", "limit": 260, "offset": 0},
    )
    tc.name = "read_file"
    d.add_call(tc)
    preview = d._rows[0].args_preview
    assert preview == "a.py"
    assert "file_path" not in preview
    assert "limit" not in preview
    assert "offset" not in preview


def test_args_preview_unknown_tool_shows_bare_values() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={"foo": "1", "bar": "2"})
    tc.name = "unregistered_tool"
    d.add_call(tc)
    preview = d._rows[0].args_preview
    assert preview == "1, 2"
    assert "foo" not in preview and "bar" not in preview
    assert '"' not in preview


def test_add_denied_appends_denied_row_with_reason() -> None:
    d = _mock_live_display()
    ar = MagicMock(tool_call_id="t1", tool_name="edit_file", reason="policy")
    d.add_denied(ar)
    row = d._rows[0]
    assert row.status == "denied"
    assert row.reason == "policy"
    line = _format_result_line(row)
    assert line is not None and "Denied" in line.plain


def test_suppressed_tools_have_no_result_line() -> None:
    row = _ToolRow(id="t1", name="sub_agent", args_preview="", status="done")
    assert _format_result_line(row) is None
    row = _ToolRow(id="t2", name="todo_write", args_preview="", status="done")
    assert _format_result_line(row) is None


def test_end_closes_live_prints_call_and_summary_and_diff_expander() -> None:
    buf = io.StringIO()
    con = Console(file=buf, color_system=None, width=120)
    d = ToolDisplay(con)
    d._live = MagicMock()
    tc = MagicMock(id="t1", arguments={"file_path": "a.py"})
    tc.name = "edit_file"
    d._rows = [_ToolRow(
        id="t1", name="edit_file", args_preview="a.py",
        status="done", tool_call=tc,
        result=MagicMock(
            content="Edited a.py (1 replacement)\n--- a/a.py\n+++ b/a.py\n@@\n-x\n+y\n",
            is_error=False,
        ),
    )]
    d.end()
    out = buf.getvalue()
    assert "Edit" in out
    assert "edit_file" not in out
    assert "Edited file (1 replacement)" in out
    assert "-x" in out and "+y" in out
    assert d._rows == []
    assert d._live is None


def test_format_call_line_running_uses_static_glyph() -> None:
    row = _ToolRow(id="t1", name="x", args_preview="", status="running")
    result = _format_call_line(row)
    assert SPINNER_STATIC in result.plain


def test_pause_closes_live_but_keeps_rows() -> None:
    d = _mock_live_display()
    tc = MagicMock(id="t1", arguments={})
    tc.name = "x"
    d.add_call(tc)
    d.pause()
    assert d._live is None
    assert len(d._rows) == 1


def test_mark_result_without_matching_row_skips_refresh() -> None:
    # Regression: suppressed tools (sub_agent/todo_write) skip add_call but
    # mark_result used to refresh anyway, opening an empty Live that collided
    # with the next markdown phase.
    d = _mock_live_display()
    d.mark_result(_read_result("anything"))
    assert d._live is None
    d._open_live.assert_not_called()


def test_show_todos_escapes_markup_in_content() -> None:
    # Regression: content like "Fix [linux] path" used to be parsed as Rich
    # markup and lose text.
    from agent_cli.theme import FLEXOKI_DARK
    buf = io.StringIO()
    con = Console(file=buf, color_system=None, width=120, theme=FLEXOKI_DARK.rich)
    d = ToolDisplay(con)
    d.show_todos(
        [
            {"status": "pending", "content": "Fix [linux] path handling"},
            {"status": "completed", "content": "[done] wrap it up"},
            {"status": "in_progress", "content": "refactor [core]"},
        ],
        {"total": 3, "completed": 1},
    )
    out = buf.getvalue()
    assert "Fix [linux] path handling" in out
    assert "[done] wrap it up" in out
    assert "refactor [core]" in out
