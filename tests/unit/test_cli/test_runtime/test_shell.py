"""Tests for runtime/shell.py — ShellState + exec_shell + helpers."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_cli.runtime.shell import (
    ShellState,
    _build_script,
    _detect_interactive,
    _detect_shell,
    _merge_env,
    _parse_env0,
    capture_startup_snapshot,
    exec_shell,
)


def _agent(*, bg_running: bool = False) -> MagicMock:
    a = MagicMock()
    a._prompt_builder.build = MagicMock(return_value="SP")
    a._make_builder_context = MagicMock(return_value={})
    a._bg_manager.has_running = MagicMock(return_value=bg_running)
    return a


def _adapter() -> MagicMock:
    ad = MagicMock()
    ad.on_shell_run = AsyncMock()
    ad.print_inline = AsyncMock()
    return ad


def _completer() -> MagicMock:
    c = MagicMock()
    c.invalidate_file_root = MagicMock()
    return c


def _state(tmp_path: Path) -> ShellState:
    s = ShellState(cwd=str(tmp_path), env=dict(os.environ))
    return s


# ---------- helpers ----------


class TestDetectInteractive:
    def test_bare_vim(self) -> None:
        assert _detect_interactive("vim foo") == "vim"

    def test_full_path_vim(self) -> None:
        assert _detect_interactive("/usr/bin/nano file") == "nano"

    def test_env_prefix_stripped(self) -> None:
        assert _detect_interactive("FOO=bar BAZ=qux vim file") == "vim"

    def test_env_prefix_then_safe_command(self) -> None:
        assert _detect_interactive("EDITOR=vim git commit") is None

    def test_sudo_target(self) -> None:
        assert _detect_interactive("sudo htop") == "htop"

    def test_sudo_with_full_path(self) -> None:
        assert _detect_interactive("sudo /usr/bin/top") == "top"

    def test_safe_command(self) -> None:
        assert _detect_interactive("ls -la") is None
        assert _detect_interactive("python script.py") is None
        assert _detect_interactive("ssh host 'ls'") is None
        assert _detect_interactive("bash -c 'vim foo'") is None

    def test_pipeline_second_segment_not_detected(self) -> None:
        assert _detect_interactive("cat foo | less") is None

    def test_empty(self) -> None:
        assert _detect_interactive("") is None
        assert _detect_interactive("   ") is None


class TestMergeEnv:
    def test_add_new_var(self) -> None:
        out = _merge_env({"PATH": "/a"}, {"PATH": "/a", "FOO": "bar"})
        assert out["FOO"] == "bar"
        assert out["PATH"] == "/a"

    def test_change_existing(self) -> None:
        out = _merge_env({"PATH": "/old"}, {"PATH": "/new"})
        assert out["PATH"] == "/new"

    def test_unset_detected(self) -> None:
        out = _merge_env({"PATH": "/a", "FOO": "bar"}, {"PATH": "/a"})
        assert "FOO" not in out

    def test_shlvl_filtered_out(self) -> None:
        out = _merge_env(
            {"PATH": "/a", "SHLVL": "1"},
            {"PATH": "/a", "SHLVL": "2"},
        )
        assert "SHLVL" not in out

    def test_harness_keys_filtered(self) -> None:
        out = _merge_env(
            {"PATH": "/a", "HARNESS_CWD_F": "/tmp/x"},
            {"PATH": "/a", "HARNESS_CWD_F": "/tmp/x"},
        )
        assert "HARNESS_CWD_F" not in out

    def test_empty_parsed_drops_all_non_transient(self) -> None:
        out = _merge_env({"PATH": "/a", "FOO": "bar"}, {})
        assert "FOO" not in out
        assert "PATH" not in out


class TestParseEnv0:
    def test_basic_pairs(self) -> None:
        blob = b"FOO=bar\0BAZ=qux\0"
        assert _parse_env0(blob) == {"FOO": "bar", "BAZ": "qux"}

    def test_empty(self) -> None:
        assert _parse_env0(b"") == {}

    def test_value_with_equals(self) -> None:
        assert _parse_env0(b"KEY=val=with=equals\0") == {"KEY": "val=with=equals"}

    def test_skip_lines_without_equals(self) -> None:
        assert _parse_env0(b"justaword\0KEY=val\0") == {"KEY": "val"}


class TestBuildScript:
    def test_includes_eval_and_trap(self, tmp_path: Path) -> None:
        s = _state(tmp_path)
        out = _build_script(s, "/tmp/cwd", "/tmp/env")
        assert 'eval "$HARNESS_CMD"' in out
        assert "trap" in out
        assert "expand_aliases" in out

    def test_source_when_snapshot_exists(self, tmp_path: Path) -> None:
        snap = tmp_path / "snap.sh"
        snap.write_text("alias hi='echo hi'\n")
        s = _state(tmp_path)
        s._startup_path = str(snap)
        out = _build_script(s, "/tmp/cwd", "/tmp/env")
        assert "source" in out
        assert str(snap) in out

    def test_self_heal_when_snapshot_missing(self, tmp_path: Path) -> None:
        s = _state(tmp_path)
        s._startup_path = str(tmp_path / "nonexistent.sh")
        s._snapshot_tried = True
        out = _build_script(s, "/tmp/cwd", "/tmp/env")
        assert "source" not in out
        assert s._startup_path is None
        assert s._snapshot_tried is False


class TestDetectShell:
    def test_uses_shell_env(self) -> None:
        with patch.dict(os.environ, {"SHELL": "/bin/bash"}, clear=False):
            assert _detect_shell() == "/bin/bash"

    def test_fallback_when_no_shell(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _detect_shell() == "/bin/sh"

    def test_fallback_for_non_sh_compatible_shell(self) -> None:
        fake_fish = Path("/tmp/fake_fish_for_test")
        fake_fish.touch()
        try:
            with patch.dict(os.environ, {"SHELL": str(fake_fish)}, clear=False):
                # name is "fake_fish_for_test" — not in _SH_COMPATIBLE
                assert _detect_shell() == "/bin/sh"
        finally:
            fake_fish.unlink()


# ---------- exec_shell integration tests (real shell) ----------

_REAL_SHELL = os.environ.get("SHELL", "/bin/sh")
_HAS_REAL_SHELL = Path(_REAL_SHELL).is_file() and sys.platform != "win32"

skip_no_shell = pytest.mark.skipif(
    not _HAS_REAL_SHELL,
    reason="needs a real POSIX shell",
)


@skip_no_shell
async def test_exec_shell_persists_cwd(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    state = _state(tmp_path)
    saved_cwd = os.getcwd()
    try:
        await exec_shell(state, "cd sub", _agent(), _completer(), _adapter())
        assert state.cwd == str(sub.resolve())
    finally:
        os.chdir(saved_cwd)


@skip_no_shell
async def test_exec_shell_persists_export_and_unset(tmp_path: Path) -> None:
    state = _state(tmp_path)
    saved_cwd = os.getcwd()
    try:
        await exec_shell(state, "export HARNESS_T_X=hello", _agent(), _completer(), _adapter())
        assert state.env.get("HARNESS_T_X") == "hello"
        await exec_shell(state, "unset HARNESS_T_X", _agent(), _completer(), _adapter())
        assert "HARNESS_T_X" not in state.env
    finally:
        os.chdir(saved_cwd)


@skip_no_shell
async def test_exec_shell_persists_cwd_on_failed_command(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    state = _state(tmp_path)
    saved_cwd = os.getcwd()
    try:
        await exec_shell(
            state,
            "cd sub; false",
            _agent(),
            _completer(),
            _adapter(),
        )
        assert state.cwd == str(sub.resolve())
    finally:
        os.chdir(saved_cwd)


@skip_no_shell
async def test_exec_shell_renders_run_with_output(tmp_path: Path) -> None:
    state = _state(tmp_path)
    adapter = _adapter()
    saved_cwd = os.getcwd()
    try:
        await exec_shell(state, "echo hi", _agent(), _completer(), adapter)
    finally:
        os.chdir(saved_cwd)
    adapter.on_shell_run.assert_awaited_once()
    args = adapter.on_shell_run.await_args.args
    assert args[0] == "echo hi"
    assert args[1] == 0
    assert "hi" in args[2]


@skip_no_shell
async def test_exec_shell_non_zero_exit_propagated(tmp_path: Path) -> None:
    state = _state(tmp_path)
    adapter = _adapter()
    saved_cwd = os.getcwd()
    try:
        await exec_shell(state, "false", _agent(), _completer(), adapter)
    finally:
        os.chdir(saved_cwd)
    args = adapter.on_shell_run.await_args.args
    assert args[1] == 1


# ---------- platform / fallback tests ----------


async def test_windows_graceful_disable() -> None:
    state = ShellState()
    adapter = _adapter()
    with patch.object(sys, "platform", "win32"):
        await exec_shell(state, "ls", _agent(), _completer(), adapter)
    args = adapter.on_shell_run.await_args.args
    assert "Windows" in args[2]


@skip_no_shell
async def test_interactive_command_rejected(tmp_path: Path) -> None:
    state = _state(tmp_path)
    adapter = _adapter()
    saved_cwd = os.getcwd()
    try:
        await exec_shell(state, "vim foo", _agent(), _completer(), adapter)
    finally:
        os.chdir(saved_cwd)
    adapter.on_shell_run.assert_not_called()
    adapter.print_inline.assert_awaited_once()


@skip_no_shell
async def test_safe_command_not_blocked(tmp_path: Path) -> None:
    state = _state(tmp_path)
    adapter = _adapter()
    saved_cwd = os.getcwd()
    try:
        await exec_shell(state, "echo hi", _agent(), _completer(), adapter)
    finally:
        os.chdir(saved_cwd)
    adapter.on_shell_run.assert_awaited_once()


# ---------- shell binary missing ----------


async def test_shell_binary_not_found(tmp_path: Path) -> None:
    state = ShellState(
        cwd=str(tmp_path),
        env=dict(os.environ),
        shell_bin="/nonexistent/shell",
    )
    state._snapshot_tried = True  # skip snapshot path
    adapter = _adapter()
    saved_cwd = os.getcwd()
    try:
        await exec_shell(state, "echo hi", _agent(), _completer(), adapter)
    finally:
        os.chdir(saved_cwd)
    adapter.on_shell_run.assert_awaited_once()
    args = adapter.on_shell_run.await_args.args
    assert args[1] == 127
    assert "Shell binary not found" in args[2]


# ---------- cwd vanished self-heal ----------


@skip_no_shell
async def test_cwd_vanished_self_heal(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    sub = parent / "sub"
    sub.mkdir(parents=True)
    state = ShellState(cwd=str(sub), env=dict(os.environ))
    saved_cwd = os.getcwd()
    try:
        shutil.rmtree(sub)
        adapter = _adapter()
        await exec_shell(state, "echo hi", _agent(), _completer(), adapter)
        assert state.cwd != str(sub)
        adapter.print_inline.assert_awaited()
    finally:
        os.chdir(saved_cwd)


@skip_no_shell
async def test_cd_blocked_when_background_running(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    state = _state(tmp_path)
    adapter = _adapter()
    saved_cwd = os.getcwd()
    try:
        await exec_shell(
            state, "cd sub", _agent(bg_running=True), _completer(), adapter,
        )
        assert state.cwd == str(tmp_path)
        adapter.print_inline.assert_awaited()
    finally:
        os.chdir(saved_cwd)


# ---------- snapshot ----------


@skip_no_shell
async def test_snapshot_round_trip(tmp_path: Path) -> None:
    base = Path(_REAL_SHELL).name
    if base not in ("bash", "zsh"):
        pytest.skip("only bash/zsh have snapshot scripts")

    state = ShellState(env=dict(os.environ))
    await capture_startup_snapshot(state)
    if state._startup_path is None:
        pytest.skip("snapshot capture unavailable in this env")

    proc = subprocess.run(
        [_REAL_SHELL, "-c", f"source {state._startup_path}; echo OK"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    state.cleanup()
    assert proc.returncode == 0
    assert "OK" in proc.stdout
    assert "command not found" not in proc.stderr.lower()


async def test_snapshot_cancellation_resets_tried(tmp_path: Path) -> None:
    state = ShellState(shell_bin="/bin/bash" if Path("/bin/bash").is_file() else "/bin/sh")
    base = Path(state.shell_bin).name
    if base not in ("bash", "zsh"):
        pytest.skip("snapshot only runs for bash/zsh")

    cancel_event = asyncio.Event()

    async def slow_capture(*args: Any, **kwargs: Any) -> Any:
        await cancel_event.wait()
        return MagicMock(returncode=0, wait=AsyncMock())

    with patch(
        "agent_cli.runtime.shell.asyncio.create_subprocess_exec",
        new=AsyncMock(side_effect=asyncio.CancelledError()),
    ):
        with pytest.raises(asyncio.CancelledError):
            await capture_startup_snapshot(state)
    assert state._snapshot_tried is False


async def test_snapshot_skipped_for_unknown_shell(tmp_path: Path) -> None:
    state = ShellState(shell_bin="/bin/fish")
    await capture_startup_snapshot(state)
    assert state._startup_path is None
    assert state._snapshot_pending_notice is None


# ---------- cleanup ----------


def test_state_cleanup_removes_snapshot(tmp_path: Path) -> None:
    snap = tmp_path / "snap.sh"
    snap.write_text("alias hi='echo'\n")
    state = ShellState()
    state._startup_path = str(snap)

    state.cleanup()

    assert not snap.exists()
    assert state._startup_path is None


def test_state_cleanup_idempotent(tmp_path: Path) -> None:
    state = ShellState()
    state.cleanup()
    state.cleanup()
