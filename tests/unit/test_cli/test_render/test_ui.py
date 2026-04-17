import io

from rich.console import Console

from agent_cli.render.ui import _BANNER_LINES, _BANNER_WIDTH, render_welcome


def test_banner_lines_all_width_58() -> None:
    for line in _BANNER_LINES:
        assert len(line) == _BANNER_WIDTH, f"len={len(line)}: {line!r}"


def test_render_welcome_emits_tagline_and_meta() -> None:
    buf = io.StringIO()
    console = Console(file=buf, color_system=None, width=100)
    render_welcome(
        console,
        version="0.4.10",
        model="gpt-5",
        cwd="/home/user/proj",
    )
    out = buf.getvalue()
    assert "Agents, harnessed." in out
    assert "v0.4.10" in out
    assert "model" in out and "gpt-5" in out
    assert "cwd" in out and "/home/user/proj" in out
    assert "session" in out
    assert "fresh" in out
    assert "/resume to restore" in out
    assert "commands" in out and "files" in out and "help" in out
