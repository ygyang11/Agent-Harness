import io
from unittest.mock import MagicMock

from prompt_toolkit.formatted_text import HTML
from rich.console import Console

from agent_cli.render.ui import (
    _BANNER_LINES,
    _BANNER_WIDTH,
    _fmt,
    make_status_bar_text,
    render_welcome,
)


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
        config_source="/home/user/.agent-harness/config.yaml",
    )
    out = buf.getvalue()
    assert "Agents, harnessed." in out
    assert "v0.4.10" in out
    assert "model" in out and "gpt-5" in out
    assert "cwd" in out and "/home/user/proj" in out
    assert "config" in out and ".agent-harness/config.yaml" in out
    assert "session" in out
    assert "fresh" in out
    assert "/resume to restore" in out
    assert "commands" in out and "files" in out and "help" in out


def test_fmt_thresholds() -> None:
    assert _fmt(500) == "500"
    assert _fmt(1500) == "1k"
    assert _fmt(42_000) == "42k"
    assert _fmt(1_500_000) == "1.5M"


def test_status_bar_text_contains_model_and_tokens() -> None:
    """Render via a stubbed prompt_toolkit app to bypass the real get_app()."""
    from unittest.mock import patch

    agent = MagicMock()
    agent.llm.model_name = "gpt-5"
    agent.context.short_term_memory.token_count = 12_345
    agent.context.short_term_memory.max_tokens = 100_000

    renderer = make_status_bar_text(agent)
    fake_app = MagicMock()
    fake_app.output.get_size.return_value = MagicMock(columns=80)
    with patch("agent_cli.render.ui.get_app", return_value=fake_app):
        out = renderer()
    assert isinstance(out, HTML)
    html = out.value
    assert "gpt-5" in html
    assert "12k/100k" in html
    assert "session" not in html.lower()


def test_status_bar_text_returns_callable() -> None:
    """Callable contract: outer call snapshots the O(N) token_count once,
    inner closure cheaply right-aligns on each redraw."""
    agent = MagicMock()
    agent.llm.model_name = "gpt-5"
    agent.context.short_term_memory.token_count = 500
    agent.context.short_term_memory.max_tokens = 100_000

    renderer = make_status_bar_text(agent)
    assert callable(renderer)


def test_status_bar_text_right_aligns_to_terminal_width() -> None:
    from unittest.mock import patch

    agent = MagicMock()
    agent.llm.model_name = "m"
    agent.context.short_term_memory.token_count = 1
    agent.context.short_term_memory.max_tokens = 1000

    renderer = make_status_bar_text(agent)
    fake_app = MagicMock()
    fake_app.output.get_size.return_value = MagicMock(columns=40)
    with patch("agent_cli.render.ui.get_app", return_value=fake_app):
        out = renderer()
    assert out.value.startswith(" ")
    assert out.value.rstrip().endswith("1/1k")
    # Leading pad + content should fill up to 40 cols
    assert len(out.value) == 40
