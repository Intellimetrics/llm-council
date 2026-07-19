from llm_council.config import select_participants
from llm_council.context import _filter_semantic_diff, resolve_stance_prompt

def _base_config() -> dict:
    return {
        "defaults": {},
        "participants": {
            "claude": {
                "type": "cli",
                "family": "claude",
                "origin": "US / Anthropic",
                "command": "claude",
                "args": ["-p", "--permission-mode", "default"],
                "model": "anthropic/claude-sonnet-4-6",
                "timeout": 240,
            }
        },
        "modes": {
            "consensus": {
                "participants": ["claude"],
            },
        },
    }

def test_single_peer_debate_mode_generates_three_virtual_peers():
    config = _base_config()
    selected = select_participants(config, "consensus", current=None)
    
    # Assert we got the three virtual peers
    assert selected == ["claude_for", "claude_against", "claude_neutral"]
    
    # Assert they are in the participants configuration
    assert "claude_for" in config["participants"]
    assert "claude_against" in config["participants"]
    assert "claude_neutral" in config["participants"]
    
    # Assert their stances are set correctly
    assert config["participants"]["claude_for"]["stance"] == "for"
    assert config["participants"]["claude_against"]["stance"] == "against"
    assert config["participants"]["claude_neutral"]["stance"] == "neutral"
    
    # Assert they are configured in modes.consensus.stances
    assert config["modes"]["consensus"]["stances"] == {
        "claude_for": "for",
        "claude_against": "against",
        "claude_neutral": "neutral",
    }

def test_semantic_diff_filtering_ignores_lockfiles_and_assets():
    diff_text = (
        "diff --git a/src/main.py b/src/main.py\n"
        "index 123456..789012 100\n"
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -1,3 +1,4 @@\n"
        "+# New logical change\n"
        "diff --git a/uv.lock b/uv.lock\n"
        "index 987654..321098 100\n"
        "--- a/uv.lock\n"
        "+++ b/uv.lock\n"
        "@@ -1,100 +1,100 @@\n"
        "+# Some package change\n"
        "diff --git b/assets/image.png b/assets/image.png\n"
        "Binary files differ\n"
    )
    
    filtered = _filter_semantic_diff(diff_text)
    
    # Assert main.py is kept
    assert "src/main.py" in filtered
    # Assert lockfile and png are ignored
    assert "uv.lock" not in filtered
    assert "image.png" not in filtered

def test_stance_prompts_are_mode_independent():
    """Stance prose comes from DEFAULT_STANCE_PROMPTS regardless of mode."""
    against_prompt = resolve_stance_prompt("against", mode="any-user-mode")
    assert "Stance: AGAINST" in against_prompt

    for_prompt = resolve_stance_prompt("for", mode="any-user-mode")
    assert "Stance: FOR" in for_prompt


def test_apply_per_peer_directives_appends_stance():
    from llm_council.context import apply_per_peer_directives
    prompt = "Base question prompt"
    
    # Assert stance is appended
    with_stance = apply_per_peer_directives(prompt, mode="consensus", family="claude", stance="for")
    assert "INDIVIDUAL ASSIGNMENT" in with_stance
    assert "representing stance: FOR" in with_stance
    assert "Stance: FOR" in with_stance
    
    # Assert no modification when stance is None
    without_stance = apply_per_peer_directives(prompt, mode="consensus", family="claude", stance=None)
    assert without_stance == prompt

