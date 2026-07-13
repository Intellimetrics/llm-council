from unittest.mock import patch
import pytest
from llm_council.cli import cmd_run_async, build_parser
from llm_council.adapters import ParticipantResult

@pytest.mark.asyncio
@patch("llm_council.cli.execute_council")
@patch("llm_council.cli.transcript_paths")
@patch("llm_council.cli.write_transcript")
@patch("httpx.post")
async def test_cli_webhook_notification(mock_post, mock_write_tr, mock_tr_paths, mock_execute):
    mock_execute.return_value = (
        [
            ParticipantResult(name="claude", ok=True, output="RECOMMENDATION: yes\n", error="", elapsed_seconds=1.0)
        ],
        {"recommendation": "yes"}
    )
    mock_tr_paths.return_value = ("md_path", "json_path")
    
    args = build_parser().parse_args(
        ["run", "--cwd", ".", "--mode", "quick", "Is this code safe?"]
    )
    
    config = {
        "version": 1,
        "participants": {
            "claude": {"type": "cli", "command": "claude"}
        },
        "modes": {
            "quick": {"participants": ["claude"]}
        },
        "notifications": {
            "webhook_url": "https://hooks.slack.com/services/test"
        }
    }
    
    with patch("llm_council.cli.load_config", return_value=config), \
         patch("llm_council.cli.find_config", return_value="path"):
        exit_code = await cmd_run_async(args)
        
    assert exit_code == 0
    assert mock_post.called
    # Assert notification payload contains summary
    call_args, call_kwargs = mock_post.call_args
    assert "LLM Council Run Finished!" in call_kwargs["json"]["text"]

@pytest.mark.asyncio
@patch("llm_council.cli.execute_council")
@patch("llm_council.cli.transcript_paths")
@patch("llm_council.cli.write_transcript")
async def test_cli_quorum_policy_failure(mock_write_tr, mock_tr_paths, mock_execute):
    mock_execute.return_value = (
        [
            ParticipantResult(name="claude", ok=True, output="RECOMMENDATION: no\n", error="", elapsed_seconds=1.0)
        ],
        {"recommendation": "no"}
    )
    mock_tr_paths.return_value = ("md_path", "json_path")
    
    args = build_parser().parse_args(
        ["run", "--cwd", ".", "--mode", "quick", "Is this code safe?"]
    )
    
    config = {
        "version": 1,
        "participants": {
            "claude": {"type": "cli", "command": "claude"}
        },
        "modes": {
            "quick": {"participants": ["claude"]}
        },
        "quorum_policies": {
            "standard": {
                "threshold": "unanimous"
            }
        }
    }
    
    with patch("llm_council.cli.load_config", return_value=config), \
         patch("llm_council.cli.find_config", return_value="path"):
        exit_code = await cmd_run_async(args)
        
    assert exit_code == 1  # Policy failed (unanimous yes failed because we got a "no")
