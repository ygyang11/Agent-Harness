"""Tests for InstructionDiscovery."""

from __future__ import annotations

from pathlib import Path

from agent_harness.prompt.instructions import (
    MAX_FILE_CHARS,
    MAX_TOTAL_CHARS,
    discover_instruction_files,
    render_instructions,
)


class TestDiscover:
    def test_discovers_agents_md(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("# Rules\nBe helpful")
        files = discover_instruction_files(tmp_path)
        assert len(files) == 1
        assert files[0].path.name == "AGENTS.md"
        assert files[0].scope == "project"

    def test_discovers_claude_md(self, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_text("# Project\nUse pytest")
        files = discover_instruction_files(tmp_path)
        assert len(files) == 1
        assert files[0].path.name == "CLAUDE.md"

    def test_discovers_both(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("Agents content")
        (tmp_path / "CLAUDE.md").write_text("Claude content")
        files = discover_instruction_files(tmp_path)
        assert len(files) == 2

    def test_traverses_upward(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("Root rules")
        child = tmp_path / "src" / "app"
        child.mkdir(parents=True)
        (child / "AGENTS.md").write_text("App rules")
        files = discover_instruction_files(child)
        assert len(files) == 2
        assert "Root" in files[0].content
        assert "App" in files[1].content

    def test_parent_scope(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("Parent content")
        child = tmp_path / "sub"
        child.mkdir()
        files = discover_instruction_files(child)
        found = [f for f in files if f.path.parent == tmp_path]
        assert len(found) == 1
        assert found[0].scope == "parent"

    def test_dedup_by_content(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("Same content")
        (tmp_path / "CLAUDE.md").write_text("Same content")
        files = discover_instruction_files(tmp_path)
        assert len(files) == 1

    def test_empty_file_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("")
        files = discover_instruction_files(tmp_path)
        assert files == []

    def test_whitespace_only_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("   \n  ")
        files = discover_instruction_files(tmp_path)
        assert files == []

    def test_empty_dir(self, tmp_path: Path) -> None:
        files = discover_instruction_files(tmp_path)
        assert files == []

    def test_global_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F821
        global_dir = tmp_path / "home" / ".agent_harness"
        global_dir.mkdir(parents=True)
        (global_dir / "AGENTS.md").write_text("Global rules")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

        project = tmp_path / "project"
        project.mkdir()
        files = discover_instruction_files(project)
        global_files = [f for f in files if f.scope == "global"]
        assert len(global_files) == 1
        assert "Global rules" in global_files[0].content

    def test_global_before_project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F821
        global_dir = tmp_path / "home" / ".agent_harness"
        global_dir.mkdir(parents=True)
        (global_dir / "AGENTS.md").write_text("Global")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

        project = tmp_path / "project"
        project.mkdir()
        (project / "AGENTS.md").write_text("Project")

        files = discover_instruction_files(project)
        assert files[0].scope == "global"
        assert files[1].scope == "project"


class TestRender:
    def test_empty(self) -> None:
        assert render_instructions([]) == ""

    def test_single_file(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("# Rules")
        files = discover_instruction_files(tmp_path)
        rendered = render_instructions(files)
        assert "# Rules" in rendered
        assert "project instructions" in rendered

    def test_scope_labels(self, tmp_path: Path) -> None:
        from agent_harness.prompt.instructions import InstructionFile

        files = [
            InstructionFile(path=Path("/g/AGENTS.md"), content="G", scope="global"),
            InstructionFile(path=Path("/p/AGENTS.md"), content="P", scope="project"),
            InstructionFile(path=Path("/x/AGENTS.md"), content="X", scope="parent"),
        ]
        rendered = render_instructions(files)
        assert "global instructions" in rendered
        assert "project instructions" in rendered
        assert "parent directory instructions" in rendered

    def test_truncation_single_file(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("x" * (MAX_FILE_CHARS + 1000))
        files = discover_instruction_files(tmp_path)
        rendered = render_instructions(files)
        assert "[truncated]" in rendered

    def test_truncation_total_budget(self, tmp_path: Path) -> None:
        current = tmp_path
        for i in range(5):
            child = current / f"level{i}"
            child.mkdir()
            (child / "AGENTS.md").write_text(f"Content {i}\n" + "y" * 3000)
            current = child
        files = discover_instruction_files(current)
        rendered = render_instructions(files)
        assert "omitted" in rendered

    def test_truncated_file_fits_within_budget(self) -> None:
        from agent_harness.prompt.instructions import InstructionFile

        large = "x" * (MAX_FILE_CHARS + 500)
        files = [
            InstructionFile(path=Path("/a.md"), content=large, scope="project"),
        ]
        rendered = render_instructions(files)
        assert "[truncated]" in rendered
        assert len(rendered) < MAX_TOTAL_CHARS + 200
