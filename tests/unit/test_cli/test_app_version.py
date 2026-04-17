import pytest

from agent_cli import __version__
from agent_cli.app import main


def test_version_flag_prints_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--version"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "harness" in out
    assert __version__ in out


def test_no_args_reports_not_wired(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    err = capsys.readouterr().err
    assert rc == 1
    assert "REPL not yet wired" in err
