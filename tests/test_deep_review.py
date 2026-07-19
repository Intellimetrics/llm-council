import pytest
from unittest.mock import patch

from llm_council.orchestrator import execute_council
from llm_council.config import select_participants
from llm_council.adapters import ParticipantResult


@pytest.mark.asyncio
async def test_deep_review_single_peer_consensus_multiplex(tmp_path):
    """A lone peer in a stance/debate mode multiplexes into three virtual
    stanced peers (the machinery behind README's single-model isolation)."""
    config = {
        "version": 1,
        "participants": {
            "claude": {
                "type": "cli",
                "family": "claude",
                "command": "claude",
                "model": "anthropic/claude-sonnet-4-6",
            }
        },
        "modes": {
            "consensus": {
                "participants": ["claude"],
            }
        }
    }
    
    question = "Is this code safe?"
    mode = "consensus"
    
    selected = select_participants(config, mode, current=None)
    assert selected == ["claude_for", "claude_against", "claude_neutral"]
    
    mode_cfg = config["modes"][mode]
    stances = mode_cfg.get("stances")
    
    captured_calls = []
    
    async def fake_run_participant(name, cfg, prompt, cwd, **kwargs):
        captured_calls.append((name, cfg, prompt))
        return ParticipantResult(
            name=name,
            ok=True,
            output=f"RECOMMENDATION: yes\nI support this change for virtual peer {name}.",
            error="",
            elapsed_seconds=0.5
        )
        
    with patch("llm_council.adapters.run_participant", side_effect=fake_run_participant):
        results, metadata = await execute_council(
            selected,
            config["participants"],
            question,
            tmp_path,
            config,
            mode=mode,
            stances=stances,
        )
        
    assert len(captured_calls) == 3
    # Check that each got their specific stance directives
    for name, cfg, prompt in captured_calls:
        assert "=== INDIVIDUAL ASSIGNMENT ===" in prompt
        if name == "claude_for":
            assert "representing stance: FOR" in prompt
            assert "Stance: FOR. Argue the strongest case" in prompt
        elif name == "claude_against":
            assert "representing stance: AGAINST" in prompt
            assert "Stance: AGAINST. Argue the strongest case" in prompt
        elif name == "claude_neutral":
            assert "representing stance: NEUTRAL" in prompt
            assert "Stance: NEUTRAL. Weigh both" in prompt
            
    assert len(results) == 3
    assert all(r.ok for r in results)


@pytest.mark.asyncio
async def test_deep_review_user_defined_stance_mode(tmp_path):
    """A project-defined mode carrying `stances` assigns the generic
    for/against/neutral stance prompts to the named peers."""
    config = {
        "version": 1,
        "participants": {
            "claude": {
                "type": "cli",
                "family": "claude",
                "command": "claude",
                "model": "anthropic/claude-sonnet-4-6",
            },
            "codex": {
                "type": "cli",
                "family": "codex",
                "command": "codex",
                "model": "openai/gpt-4o",
            },
            "gemini": {
                "type": "cli",
                "family": "gemini",
                "command": "gemini",
                "model": "google/gemini-1.5-pro",
            }
        },
        "modes": {
            "attack-defend": {
                "participants": ["claude", "codex", "gemini"],
                "stances": {
                    "claude": "against",
                    "codex": "for",
                    "gemini": "neutral",
                }
            }
        }
    }
    
    question = "Should we merge this PR?"
    mode = "attack-defend"
    
    selected = select_participants(config, mode, current=None)
    mode_cfg = config["modes"][mode]
    stances = mode_cfg.get("stances")
    
    captured_calls = []
    
    async def fake_run_participant(name, cfg, prompt, cwd, **kwargs):
        captured_calls.append((name, cfg, prompt))
        return ParticipantResult(
            name=name,
            ok=True,
            output=f"RECOMMENDATION: yes\nFeedback from {name}.",
            error="",
            elapsed_seconds=0.5
        )
        
    with patch("llm_council.adapters.run_participant", side_effect=fake_run_participant):
        results, metadata = await execute_council(
            selected,
            config["participants"],
            question,
            tmp_path,
            config,
            mode=mode,
            stances=stances,
        )
        
    assert len(captured_calls) == 3
    # Check that each got their specific stance directives according to mode config
    for name, cfg, prompt in captured_calls:
        assert "=== INDIVIDUAL ASSIGNMENT ===" in prompt
        if name == "claude":
            assert "representing stance: AGAINST" in prompt
            assert "Stance: AGAINST. Argue the strongest case" in prompt
        elif name == "codex":
            assert "representing stance: FOR" in prompt
            assert "Stance: FOR. Argue the strongest case" in prompt
        elif name == "gemini":
            assert "representing stance: NEUTRAL" in prompt
            assert "Stance: NEUTRAL. Weigh both" in prompt


@pytest.mark.asyncio
async def test_deep_review_custom_three_round_mode(tmp_path):
    """A project-defined mode with deliberate+max_rounds=3 runs three rounds."""
    config = {
        "version": 1,
        "participants": {
            "claude": {
                "type": "cli",
                "family": "claude",
                "command": "claude",
                "model": "anthropic/claude-sonnet-4-6",
            },
            "codex": {
                "type": "cli",
                "family": "codex",
                "command": "codex",
                "model": "openai/gpt-4o",
            }
        },
        "modes": {
            "audit3": {
                "participants": ["claude", "codex"],
                "deliberate": True,
                "max_rounds": 3,
            }
        }
    }
    
    question = "Perform a deep audit of this PR."
    mode = "audit3"
    
    selected = select_participants(config, mode, current=None)
    assert selected == ["claude", "codex"]
    
    calls = 0
    
    async def fake_run_participant(name, cfg, prompt, cwd, **kwargs):
        nonlocal calls
        calls += 1
        
        # Round 1
        if calls <= 2:
            if name == "claude":
                output = "RECOMMENDATION: yes\nRound 1: Claude approves."
            else:
                output = "RECOMMENDATION: no\nRound 1: Codex objects."
        # Round 2
        elif calls <= 4:
            if name == "claude":
                output = "RECOMMENDATION: yes\nRound 2: Claude still approves."
            else:
                output = "RECOMMENDATION: no\nRound 2: Codex still objects."
        # Round 3
        else:
            output = "RECOMMENDATION: yes\nRound 3: Converged."
            
        return ParticipantResult(
            name=name,
            ok=True,
            output=output,
            error="",
            elapsed_seconds=0.1
        )
        
    with patch("llm_council.adapters.run_participant", side_effect=fake_run_participant):
        results, metadata = await execute_council(
            selected,
            config["participants"],
            question,
            tmp_path,
            config,
            mode=mode,
            deliberate=True,
            max_rounds=3,
        )
        
    assert calls == 6
    assert metadata["rounds"] == 3
    assert metadata["deliberation_status"] == "ran_no_labeled_disagreement"
    
    result_names = [r.name for r in results]
    assert "claude" in result_names
    assert "codex" in result_names
    assert "claude:round2" in result_names
    assert "codex:round2" in result_names
    assert "claude:round3" in result_names
    assert "codex:round3" in result_names


@pytest.mark.asyncio
async def test_deep_review_cli_e2e_flow(tmp_path):
    from llm_council.cli import cmd_run_async, build_parser
    
    config_content = """
version: 1
defaults:
  mode: test-consensus
participants:
  claude:
    type: cli
    family: claude
    command: claude
    model: anthropic/claude-sonnet-4-6
modes:
  test-consensus:
    participants: [claude]
quorum_policies:
  standard:
    threshold: unanimous
"""
    config_file = tmp_path / ".llm-council.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    
    args = build_parser().parse_args(
        ["run", "--cwd", str(tmp_path), "--mode", "test-consensus", "Is this correct?"]
    )
    
    async def fake_run_participant(name, cfg, prompt, cwd, **kwargs):
        return ParticipantResult(
            name=name,
            ok=True,
            output="RECOMMENDATION: no\nThere is a flaw.",
            error="",
            elapsed_seconds=0.1
        )
        
    with patch("llm_council.adapters.run_participant", side_effect=fake_run_participant), \
         patch("llm_council.cli.find_config", return_value=str(config_file)), \
         patch("llm_council.cli.write_transcript"):
        
        exit_code = await cmd_run_async(args)
        
    assert exit_code == 1


@pytest.mark.asyncio
async def test_deep_review_cli_custom_quorum_policy_by_context(tmp_path):
    from llm_council.cli import cmd_run_async, build_parser
    
    config_content = """
version: 1
defaults:
  mode: test-consensus
participants:
  claude:
    type: cli
    family: claude
    command: claude
    model: anthropic/claude-sonnet-4-6
  codex:
    type: cli
    family: codex
    command: codex
    model: openai/gpt-4o
  gemini:
    type: cli
    family: gemini
    command: gemini
    model: google/gemini-1.5-pro
modes:
  test-consensus:
    participants: [claude, codex, gemini]
quorum_policies:
  standard:
    threshold: majority
  critical:
    threshold: unanimous
"""
    config_file = tmp_path / ".llm-council.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    
    # Create the files we are referencing in context so they pass validation
    safe_file = tmp_path / "safe.py"
    safe_file.write_text("# Safe logic", encoding="utf-8")
    critical_file = tmp_path / "critical_logic.py"
    critical_file.write_text("# Critical logic", encoding="utf-8")
    
    async def fake_run_participant(name, cfg, prompt, cwd, **kwargs):
        if name == "claude":
            output = "RECOMMENDATION: no\nSecurity risk found."
        else:
            output = "RECOMMENDATION: yes\nLooks good to me."
        return ParticipantResult(
            name=name,
            ok=True,
            output=output,
            error="",
            elapsed_seconds=0.1
        )
        
    # Case A: Context has standard file (safe.py). Majority threshold should pass.
    args_safe = build_parser().parse_args(
        ["run", "--cwd", str(tmp_path), "--mode", "test-consensus", "--context", "safe.py", "Is this correct?"]
    )
    with patch("llm_council.adapters.run_participant", side_effect=fake_run_participant), \
         patch("llm_council.cli.find_config", return_value=str(config_file)), \
         patch("llm_council.cli.write_transcript"):
        exit_code_safe = await cmd_run_async(args_safe)
        
    assert exit_code_safe == 0  # 2/3 yes is majority, passes
    
    # Case B: Context has critical file (critical_logic.py). Unanimous threshold should fail (due to 1 'no' vote).
    args_critical = build_parser().parse_args(
        ["run", "--cwd", str(tmp_path), "--mode", "test-consensus", "--context", "critical_logic.py", "Is this correct?"]
    )
    with patch("llm_council.adapters.run_participant", side_effect=fake_run_participant), \
         patch("llm_council.cli.find_config", return_value=str(config_file)), \
         patch("llm_council.cli.write_transcript"):
        exit_code_critical = await cmd_run_async(args_critical)
        
    assert exit_code_critical == 1  # Unanimous failed because of 1 'no' vote


@pytest.mark.asyncio
async def test_deep_review_contextual_persona_recruitment(tmp_path):
    config = {
        "version": 1,
        "participants": {
            "claude": {
                "type": "cli",
                "family": "claude",
                "command": "claude",
                "model": "anthropic/claude-sonnet-4-6",
            }
        },
        "modes": {
            "consensus": {
                "participants": ["claude"],
                "stances": {"claude": "for"}
            }
        }
    }
    
    question = "Is this migration safe?"
    mode = "consensus"
    selected = select_participants(config, mode, current=None)
    
    captured_calls = []
    async def fake_run_participant(name, cfg, prompt, cwd, **kwargs):
        captured_calls.append((name, cfg, prompt))
        return ParticipantResult(name=name, ok=True, output="RECOMMENDATION: yes", error="", elapsed_seconds=0.1)

    # Mock git output to return a database migration SQL file change
    def fake_git_output(cwd, args):
        if "diff" in args and "--name-only" in args:
            return "db/migrations/0001_initial.sql\n"
        return ""

    with patch("llm_council.adapters.run_participant", side_effect=fake_run_participant), \
         patch("llm_council.context._git_output", side_effect=fake_git_output):
        results, metadata = await execute_council(
            selected,
            config["participants"],
            question,
            tmp_path,
            config,
            mode=mode,
            stances={"claude": "for"},
        )
        
    assert len(captured_calls) == 3
    prompt_sent = next(c[2] for c in captured_calls if c[0] == "claude_for")
    assert "=== CONTEXTUAL ROLE ASSIGNMENT ===" in prompt_sent
    assert "DATABASE ARCHITECT" in prompt_sent


def test_deep_review_smart_model_routing(tmp_path):
    from llm_council.config import apply_smart_routing
    config = {
        "version": 1,
        "smart_routing": {
            "enabled": True,
        },
        "participants": {
            "claude": {
                "type": "cli",
                "family": "claude",
                "command": "claude",
                "model": "anthropic/claude-sonnet-4-6",
            }
        }
    }
    
    # Mock git output to show a small doc edit (low-risk)
    def fake_git_output(cwd, args):
        if "diff" in args and "--name-only" in args:
            return "README.md\n"
        if "diff" in args and "--shortstat" in args:
            return " 1 file changed, 5 insertions(+)"
        return ""
        
    with patch("llm_council.context._git_output", side_effect=fake_git_output):
        apply_smart_routing(config, "quick", tmp_path)
        
    # The premium model should be downgraded to Claude Haiku
    assert config["participants"]["claude"]["model"] == "anthropic/claude-haiku-4-5"


def test_deep_review_html_transcript_generation(tmp_path):
    from llm_council.transcript import write_transcript
    
    md_path = tmp_path / "runs" / "run.md"
    json_path = tmp_path / "runs" / "run.json"
    html_path = tmp_path / "runs" / "run.html"
    
    results = [
        ParticipantResult(name="claude", ok=True, output="RECOMMENDATION: yes\nApproved.", error="", elapsed_seconds=0.5)
    ]
    
    write_transcript(
        markdown_path=md_path,
        json_path=json_path,
        question="Is it safe?",
        mode="quick",
        current=None,
        participants=["claude"],
        prompt="Is it safe?",
        results=results,
        metadata={"recommendation": "yes"}
    )
    
    # Verify HTML file exists and has correct elements
    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "<title>LLM Council Transcript Dashboard</title>" in html_text
    assert "Decision: YES" in html_text
    assert "Approved." in html_text


def test_deep_review_git_hook_installer(tmp_path):
    from llm_council.cli import cmd_install_hook
    import argparse
    import subprocess

    # Use a real repository so the installer exercises git's hook-path
    # resolution (the same path also works for linked worktrees).
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    git_dir = tmp_path / ".git"

    args = argparse.Namespace(
        root=str(tmp_path),
        hook_type="pre-commit",
        mode="consensus",
        force=False,
    )
    
    exit_code = cmd_install_hook(args)
    assert exit_code == 0
    
    hook_file = git_dir / "hooks" / "pre-commit"
    assert hook_file.exists()
    
    hook_content = hook_file.read_text(encoding="utf-8")
    assert "pre-commit validation" in hook_content.lower()
    assert "--mode consensus" in hook_content


def test_deep_review_cmd_last_and_show_browser_open(tmp_path, capsys):
    from llm_council.cli import cmd_last, build_parser, cmd_transcripts
    import json
    
    # 1. Create a dummy config and transcript runs dir
    config_content = """
version: 1
transcripts_dir: ".llm-council/runs"
"""
    config_file = tmp_path / ".llm-council.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    
    runs_dir = tmp_path / ".llm-council" / "runs"
    runs_dir.mkdir(parents=True)
    
    # Write sample .md, .json, and .html transcripts
    md_file = runs_dir / "20260614_120000_consensus.md"
    md_file.write_text("Markdown Content", encoding="utf-8")
    
    json_file = runs_dir / "20260614_120000_consensus.json"
    json_file.write_text(json.dumps({"question": "Is this correct?"}), encoding="utf-8")
    
    html_file = runs_dir / "20260614_120000_consensus.html"
    html_file.write_text("<html>HTML Content</html>", encoding="utf-8")
    
    parser = build_parser()
    
    # Test A: cmd_last with md, json, html
    # Default is MD
    args_last_md = parser.parse_args(["last", "--cwd", str(tmp_path)])
    with patch("llm_council.cli.find_config", return_value=str(config_file)):
        cmd_last(args_last_md)
    out_md, _ = capsys.readouterr()
    assert "Markdown Content" in out_md
    
    # JSON file
    args_last_json = parser.parse_args(["last", "--cwd", str(tmp_path), "--json-file"])
    with patch("llm_council.cli.find_config", return_value=str(config_file)):
        cmd_last(args_last_json)
    out_json, _ = capsys.readouterr()
    assert "question" in out_json
    
    # HTML file
    args_last_html = parser.parse_args(["last", "--cwd", str(tmp_path), "--html-file"])
    with patch("llm_council.cli.find_config", return_value=str(config_file)):
        cmd_last(args_last_html)
    out_html, _ = capsys.readouterr()
    assert "HTML Content" in out_html

    # Test B: cmd_last --open (implying html-file by default)
    args_last_open = parser.parse_args(["last", "--cwd", str(tmp_path), "--open"])
    with patch("llm_council.cli.find_config", return_value=str(config_file)), \
         patch("webbrowser.open", return_value=True) as mock_open:
        cmd_last(args_last_open)
        mock_open.assert_called_once_with(html_file.resolve().as_uri())
    out_open, _ = capsys.readouterr()
    assert "Opening transcript:" in out_open

    # Test B2: cmd_last --open fail raises SystemExit
    args_last_open_fail = parser.parse_args(["last", "--cwd", str(tmp_path), "--open"])
    with patch("llm_council.cli.find_config", return_value=str(config_file)), \
         patch("webbrowser.open", return_value=False):
        with pytest.raises(SystemExit) as excinfo:
            cmd_last(args_last_open_fail)
        assert "Failed to open browser" in str(excinfo.value)

    # Test B3: cmd_last mutually exclusive check
    args_last_conflict = parser.parse_args(["last", "--cwd", str(tmp_path), "--html-file", "--json-file"])
    with patch("llm_council.cli.find_config", return_value=str(config_file)):
        with pytest.raises(SystemExit) as excinfo:
            cmd_last(args_last_conflict)
        assert "mutually exclusive" in str(excinfo.value)

    # Test C: transcripts show default (MD)
    args_show_md = parser.parse_args(["transcripts", "show", "--cwd", str(tmp_path)])
    with patch("llm_council.cli.find_config", return_value=str(config_file)):
        cmd_transcripts(args_show_md)
    out_show_md, _ = capsys.readouterr()
    assert "Markdown Content" in out_show_md

    # Test D: transcripts show HTML --open
    args_show_open = parser.parse_args(["transcripts", "show", "--cwd", str(tmp_path), "--html-file", "--open"])
    with patch("llm_council.cli.find_config", return_value=str(config_file)), \
         patch("webbrowser.open", return_value=True) as mock_open:
        cmd_transcripts(args_show_open)
        mock_open.assert_called_once_with(html_file.resolve().as_uri())
    out_show_open, _ = capsys.readouterr()
    assert "Opening transcript:" in out_show_open

    # Test E: transcripts show with explicit path and --html-file override
    explicit_md_path = str(md_file)
    args_show_explicit_html = parser.parse_args(["transcripts", "show", explicit_md_path, "--cwd", str(tmp_path), "--html-file"])
    with patch("llm_council.cli.find_config", return_value=str(config_file)):
        cmd_transcripts(args_show_explicit_html)
    out_explicit_html, _ = capsys.readouterr()
    assert "HTML Content" in out_explicit_html

    # Test F: transcripts show with missing path
    args_show_missing = parser.parse_args(["transcripts", "show", "nonexistent.md", "--cwd", str(tmp_path)])
    with patch("llm_council.cli.find_config", return_value=str(config_file)):
        with pytest.raises(SystemExit) as excinfo:
            cmd_transcripts(args_show_missing)
        assert "does not exist" in str(excinfo.value)

    # Test G: transcripts show --open on missing path
    args_show_missing_open = parser.parse_args(["transcripts", "show", "nonexistent.md", "--cwd", str(tmp_path), "--open"])
    with patch("llm_council.cli.find_config", return_value=str(config_file)):
        with pytest.raises(SystemExit) as excinfo:
            cmd_transcripts(args_show_missing_open)
        assert "does not exist" in str(excinfo.value)

    # Test H: transcripts show runs/run.md --open resolves to .html if exists
    args_show_md_open = parser.parse_args(["transcripts", "show", str(md_file), "--cwd", str(tmp_path), "--open"])
    with patch("llm_council.cli.find_config", return_value=str(config_file)), \
         patch("webbrowser.open", return_value=True) as mock_open:
        cmd_transcripts(args_show_md_open)
        mock_open.assert_called_once_with(html_file.resolve().as_uri())


@pytest.mark.asyncio
async def test_deep_review_run_command_auto_open_browser(tmp_path):
    from llm_council.cli import cmd_run_async, build_parser
    from llm_council.adapters import ParticipantResult
    
    config_content = """
version: 1
defaults:
  mode: test-mode
  auto_open_browser: true
participants:
  claude:
    type: cli
    family: claude
    command: claude
    model: anthropic/claude-sonnet-4-6
modes:
  test-mode:
    participants: [claude]
"""
    config_file = tmp_path / ".llm-council.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    
    args = build_parser().parse_args(
        ["run", "--cwd", str(tmp_path), "Is this correct?"]
    )
    
    async def fake_run_participant(name, cfg, prompt, cwd, **kwargs):
        return ParticipantResult(
            name=name,
            ok=True,
            output="RECOMMENDATION: yes\nApproved.",
            error="",
            elapsed_seconds=0.1
        )
        
    with patch("llm_council.adapters.run_participant", side_effect=fake_run_participant), \
         patch("llm_council.cli.find_config", return_value=str(config_file)), \
         patch("webbrowser.open", return_value=True) as mock_open:
        
        await cmd_run_async(args)
        
        # Verify webbrowser.open was called with the resolved html path
        assert mock_open.call_count == 1
        call_url = mock_open.call_args[0][0]
        assert call_url.endswith(".html")
        assert "runs" in call_url


def test_deep_review_config_command(tmp_path, capsys):
    from llm_council.cli import cmd_config, build_parser
    import yaml
    
    # 1. Create a dummy config file
    config_content = """
version: 1
defaults:
  mode: quick
  auto_open_browser: false
"""
    config_file = tmp_path / ".llm-council.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    
    parser = build_parser()
    
    # Test A: Get setting
    args_get = parser.parse_args(["config", "get", "defaults.auto_open_browser", "--cwd", str(tmp_path)])
    with patch("llm_council.cli.find_config", return_value=str(config_file)):
        cmd_config(args_get)
    out_get, _ = capsys.readouterr()
    assert "False" in out_get.strip()
    
    # Test B: Set setting to True
    args_set = parser.parse_args(["config", "set", "defaults.auto_open_browser", "true", "--cwd", str(tmp_path)])
    with patch("llm_council.cli.find_config", return_value=str(config_file)):
        cmd_config(args_set)
    out_set, _ = capsys.readouterr()
    assert "Successfully set defaults.auto_open_browser to True" in out_set
    
    # Reload and verify
    config_new = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert config_new["defaults"]["auto_open_browser"] is True
    
    # Test C: Set a nested key that doesn't exist yet
    args_set_nested = parser.parse_args(["config", "set", "defaults.some.nested.key", "123", "--cwd", str(tmp_path)])
    with patch("llm_council.cli.find_config", return_value=str(config_file)):
        cmd_config(args_set_nested)
        
    config_nested = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert config_nested["defaults"]["some"]["nested"]["key"] == 123


@pytest.mark.asyncio
async def test_deep_review_mcp_run_auto_open_browser(tmp_path):
    from llm_council.mcp_server import run_council
    from llm_council.adapters import ParticipantResult
    
    config_content = """
version: 1
defaults:
  mode: test-mode
  auto_open_browser: true
participants:
  claude:
    type: cli
    family: claude
    command: claude
    model: anthropic/claude-sonnet-4-6
modes:
  test-mode:
    participants: [claude]
"""
    config_file = tmp_path / ".llm-council.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    
    import os
    async def fake_run_participant(name, cfg, prompt, cwd, **kwargs):
        return ParticipantResult(
            name=name,
            ok=True,
            output="RECOMMENDATION: yes\nApproved.",
            error="",
            elapsed_seconds=0.1
        )
        
    with patch.dict(os.environ, {"LLM_COUNCIL_MCP_ROOT": str(tmp_path)}), \
         patch("llm_council.adapters.run_participant", side_effect=fake_run_participant), \
         patch("llm_council.mcp_server.find_config", return_value=str(config_file)), \
         patch("webbrowser.open", return_value=True) as mock_open:
        
        await run_council({
            "question": "Is this correct?",
            "working_directory": str(tmp_path)
        })
        
        # Verify webbrowser.open was called with the resolved html path
        assert mock_open.call_count == 1
        call_url = mock_open.call_args[0][0]
        assert call_url.endswith(".html")
        assert "runs" in call_url


@pytest.mark.asyncio
async def test_deep_review_mcp_run_open_parameter(tmp_path):
    from llm_council.mcp_server import run_council
    from llm_council.adapters import ParticipantResult
    
    config_content = """
version: 1
defaults:
  mode: test-mode
  auto_open_browser: false
participants:
  claude:
    type: cli
    family: claude
    command: claude
    model: anthropic/claude-sonnet-4-6
modes:
  test-mode:
    participants: [claude]
"""
    config_file = tmp_path / ".llm-council.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    
    import os
    async def fake_run_participant(name, cfg, prompt, cwd, **kwargs):
        return ParticipantResult(
            name=name,
            ok=True,
            output="RECOMMENDATION: yes\nApproved.",
            error="",
            elapsed_seconds=0.1
        )
        
    with patch.dict(os.environ, {"LLM_COUNCIL_MCP_ROOT": str(tmp_path)}), \
         patch("llm_council.adapters.run_participant", side_effect=fake_run_participant), \
         patch("llm_council.mcp_server.find_config", return_value=str(config_file)), \
         patch("webbrowser.open", return_value=True) as mock_open:
        
        await run_council({
            "question": "Is this correct?",
            "working_directory": str(tmp_path),
            "open": True
        })

        
        # Verify webbrowser.open was called
        assert mock_open.call_count == 1




