from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from llm_council import context as context_module
from llm_council.config import find_config, resolve_config_data
from llm_council.context import (
    MAX_ACCEPTANCE_CONTRACT_CHARS,
    MAX_CONTEXT_FILES,
    MAX_CONTEXT_FILE_CHARS,
    build_prompt,
    read_context_file,
    _read_git_diff_sections,
    resolve_acceptance_contract,
)
from llm_council.env import (
    env_get,
    load_project_env,
    project_env_context,
    resolve_project_env,
)


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"symlinks unavailable: {exc}")


def test_find_config_stop_at_is_inclusive_and_preserves_unbounded_default(
    tmp_path: Path,
) -> None:
    boundary = tmp_path / "project"
    leaf = boundary / "src" / "nested"
    leaf.mkdir(parents=True)
    parent_config = tmp_path / ".llm-council.yaml"
    parent_config.write_text("version: 1\n", encoding="utf-8")

    assert find_config(leaf) == parent_config
    assert find_config(leaf, stop_at=boundary) is None

    boundary_config = boundary / ".llm-council.yml"
    boundary_config.write_text("version: 1\n", encoding="utf-8")
    assert find_config(leaf, stop_at=boundary) == boundary_config


def test_find_config_stop_at_rejects_symlink_escapes(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_config = outside / ".llm-council.yaml"
    outside_config.write_text("version: 1\n", encoding="utf-8")

    linked_directory = trusted / "linked-directory"
    _symlink(linked_directory, outside, directory=True)
    with pytest.raises(ValueError, match="outside stop_at boundary"):
        find_config(linked_directory, stop_at=trusted)

    nested = trusted / "nested"
    nested.mkdir()
    _symlink(nested / ".llm-council.yaml", outside_config)
    assert find_config(nested, stop_at=trusted) is None


def test_project_env_stop_at_excludes_parent_and_is_context_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundary = tmp_path / "project"
    leaf = boundary / "src"
    leaf.mkdir(parents=True)
    (tmp_path / ".env").write_text(
        "BOUNDARY_PARENT_ONLY=parent\nBOUNDARY_SHARED=parent\n", encoding="utf-8"
    )
    boundary_env = boundary / ".env"
    boundary_env.write_text("BOUNDARY_SHARED=project\n", encoding="utf-8")

    bounded, loaded = resolve_project_env(leaf, base_env={}, stop_at=boundary)
    assert bounded["BOUNDARY_SHARED"] == "project"
    assert "BOUNDARY_PARENT_ONLY" not in bounded
    assert loaded == [boundary_env]

    unbounded, _ = resolve_project_env(leaf, base_env={})
    assert unbounded["BOUNDARY_PARENT_ONLY"] == "parent"
    assert unbounded["BOUNDARY_SHARED"] == "project"

    monkeypatch.delenv("BOUNDARY_PARENT_ONLY", raising=False)
    with project_env_context(leaf, stop_at=boundary):
        assert env_get("BOUNDARY_SHARED") == "project"
        assert env_get("BOUNDARY_PARENT_ONLY") is None


def test_project_env_stop_at_skips_dotenv_symlink_target_outside_boundary(
    tmp_path: Path,
) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    outside_env = tmp_path / "outside.env"
    outside_env.write_text("BOUNDARY_SYMLINK_SECRET=outside\n", encoding="utf-8")
    linked_env = trusted / ".env"
    _symlink(linked_env, outside_env)

    bounded, loaded = resolve_project_env(trusted, base_env={}, stop_at=trusted)
    assert "BOUNDARY_SYMLINK_SECRET" not in bounded
    assert linked_env not in loaded

    unbounded, loaded_unbounded = resolve_project_env(trusted, base_env={})
    assert unbounded["BOUNDARY_SYMLINK_SECRET"] == "outside"
    assert linked_env in loaded_unbounded


def test_process_env_loader_honors_stop_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundary = tmp_path / "project"
    leaf = boundary / "nested"
    leaf.mkdir(parents=True)
    (tmp_path / ".llm-council.env").write_text(
        "PROCESS_BOUNDARY_PARENT=parent\n", encoding="utf-8"
    )
    boundary_env = boundary / ".llm-council.env"
    boundary_env.write_text("PROCESS_BOUNDARY_PROJECT=project\n", encoding="utf-8")
    monkeypatch.delenv("PROCESS_BOUNDARY_PARENT", raising=False)
    monkeypatch.delenv("PROCESS_BOUNDARY_PROJECT", raising=False)

    loaded = load_project_env(leaf, stop_at=boundary)
    assert loaded == [boundary_env]
    assert env_get("PROCESS_BOUNDARY_PROJECT") == "project"
    assert env_get("PROCESS_BOUNDARY_PARENT") is None


def test_context_file_streams_only_bounded_character_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "huge.txt"
    source.write_text("é" * (MAX_CONTEXT_FILE_CHARS + 25), encoding="utf-8")

    def fail_unbounded_read(*_args, **_kwargs):
        raise AssertionError("Path.read_text must not be used for bounded context")

    monkeypatch.setattr(Path, "read_text", fail_unbounded_read)
    rendered = read_context_file(source, cwd=tmp_path)
    assert rendered.count("é") == MAX_CONTEXT_FILE_CHARS
    assert "[truncated]" in rendered


def test_acceptance_contract_file_and_literal_share_character_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "contract.md"
    source.write_text(
        "c" * (MAX_ACCEPTANCE_CONTRACT_CHARS + 25), encoding="utf-8"
    )

    def fail_unbounded_read(*_args, **_kwargs):
        raise AssertionError("Path.read_text must not be used for bounded contracts")

    monkeypatch.setattr(Path, "read_text", fail_unbounded_read)
    from_file = resolve_acceptance_contract("contract.md", cwd=tmp_path)
    assert from_file is not None
    file_prefix, _marker = from_file.split("\n\n[truncated]", 1)
    assert file_prefix == "c" * MAX_ACCEPTANCE_CONTRACT_CHARS
    assert from_file.endswith("[truncated]")

    literal = resolve_acceptance_contract(
        "l" * (MAX_ACCEPTANCE_CONTRACT_CHARS + 25), cwd=tmp_path
    )
    assert literal is not None
    literal_prefix, _marker = literal.split("\n\n[truncated]", 1)
    assert literal_prefix == "l" * MAX_ACCEPTANCE_CONTRACT_CHARS
    assert literal.endswith("[truncated]")


def test_context_symlink_target_outside_cwd_is_rejected(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    linked = trusted / "linked.txt"
    _symlink(linked, outside)

    with pytest.raises(ValueError, match="outside working directory"):
        read_context_file(linked, cwd=trusted)


def test_build_prompt_enforces_context_count_before_file_reads(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Too many context files"):
        build_prompt(
            "review",
            mode="quick",
            cwd=tmp_path,
            context_paths=["missing.txt"] * (MAX_CONTEXT_FILES + 1),
            include_diff=False,
            stdin_text=None,
        )


def test_build_prompt_enforces_aggregate_context_limit_before_full_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("a" * 8, encoding="utf-8")
    second.write_text("b" * 8, encoding="utf-8")
    monkeypatch.setattr(context_module, "MAX_CONTEXT_TOTAL_CHARS", 10)

    with pytest.raises(ValueError, match="aggregate character limit"):
        build_prompt(
            "review",
            mode="quick",
            cwd=tmp_path,
            context_paths=[str(first), str(second)],
            include_diff=False,
            stdin_text=None,
        )


def test_git_capture_uses_bounded_temporary_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(context_module, "MAX_GIT_DIFF_CAPTURE_BYTES", 64)
    monkeypatch.setattr(context_module, "MAX_GIT_STDERR_CAPTURE_BYTES", 32)
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed.update(kwargs)
        kwargs["stdout"].write(b"x" * 100)
        kwargs["stderr"].write(b"e" * 100)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(context_module.subprocess, "run", fake_run)
    result = context_module._run_git(tmp_path, ["diff", "--"])

    assert "capture_output" not in observed
    assert observed["stdin"] is subprocess.DEVNULL
    assert observed["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert observed["env"]["GIT_PAGER"] == "cat"
    assert result.stdout.split("\n[git ", 1)[0] == "x" * 64
    assert "truncated after 64 bytes" in result.stdout
    assert result.stderr.split("\n[git ", 1)[0] == "e" * 32
    assert "truncated after 32 bytes" in result.stderr


def test_git_capture_preserves_timeout_failure_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, 15)

    monkeypatch.setattr(context_module.subprocess, "run", fake_run)
    result = context_module._run_git(tmp_path, ["diff", "--"])
    assert result.returncode == 1
    assert result.stdout == ""
    assert "timed out after 15 seconds" in result.stderr


def test_git_output_skips_process_for_non_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_run(*_args, **_kwargs):
        raise AssertionError("Git must not launch outside a repository")

    monkeypatch.setattr(context_module.subprocess, "run", fail_run)

    assert context_module._git_output(tmp_path, ["diff", "--name-only"]) is None
    assert context_module._git_ok(tmp_path, ["rev-parse", "--is-inside-work-tree"]) is False


def test_semantic_filter_preserves_truncation_notice_from_ignored_block() -> None:
    raw = (
        "diff --git a/image.png b/image.png\n"
        "Binary files a/image.png and b/image.png differ\n"
        "[git diff output truncated after 64 bytes; narrow the diff before review]\n"
    )
    filtered = context_module._filter_semantic_diff(raw)
    assert "Binary files" not in filtered
    assert "git diff output truncated after 64 bytes" in filtered


def test_read_git_diff_surfaces_capture_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True
    )
    source = tmp_path / "large.txt"
    source.write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "large.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    source.write_text("new\n" + ("z" * 10_000), encoding="utf-8")
    monkeypatch.setattr(context_module, "MAX_GIT_DIFF_CAPTURE_BYTES", 512)

    rendered = "\n".join(_read_git_diff_sections(tmp_path)[0])
    assert "Git Diff" in rendered
    assert "truncated after 512 bytes" in rendered


@pytest.mark.parametrize("value", [0, -1, float("nan"), "1200", True])
def test_mcp_request_timeout_must_be_a_positive_number(value: object) -> None:
    with pytest.raises(ValueError, match="mcp_request_timeout_seconds"):
        resolve_config_data({"defaults": {"mcp_request_timeout_seconds": value}})
