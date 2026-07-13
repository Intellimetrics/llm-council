"""Privacy-boundary and remote-Ollama cost regressions."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pytest

from llm_council import cli as cli_module
from llm_council import adapters as adapters_module
from llm_council import orchestrator as orchestrator_module
from llm_council.budget import (
    enforce_preflight_caps,
    mcp_budget_report,
    summarize_preflight_caps,
)
from llm_council.config import is_local_participant, select_participants
from llm_council.doctor import Check, check_environment
from llm_council.estimate import estimate_council


@pytest.mark.parametrize(
    ("participant", "expected"),
    [
        ({"type": "ollama", "model": "qwen"}, True),
        (
            {
                "type": "ollama",
                "model": "qwen",
                "base_url": "http://localhost:11434",
            },
            True,
        ),
        (
            {
                "type": "openai_compatible",
                "model": "qwen",
                "base_url": "http://127.0.0.1:8000/v1",
            },
            True,
        ),
        (
            {
                "type": "openai_compatible",
                "model": "qwen",
                "base_url": "http://[::1]:8000/v1",
            },
            True,
        ),
        (
            {
                "type": "ollama",
                "model": "qwen",
                "base_url": "http://10.0.0.5:11434",
            },
            False,
        ),
        (
            {
                "type": "ollama",
                "model": "qwen",
                "base_url": "https://ollama.example.com",
            },
            False,
        ),
        (
            {
                "type": "openai_compatible",
                "model": "qwen",
                "base_url": "http://192.168.1.9:8000/v1",
            },
            False,
        ),
        (
            {
                "type": "ollama",
                "model": "qwen",
                "base_url": "http://0.0.0.0:11434",
            },
            False,
        ),
        (
            {
                "type": "ollama",
                "model": "qwen",
                "base_url": "file://localhost/tmp/ollama.sock",
            },
            False,
        ),
    ],
)
def test_is_local_participant_means_same_machine_loopback(
    participant: dict, expected: bool
) -> None:
    assert is_local_participant(participant) is expected


def test_private_local_selects_only_loopback_endpoints() -> None:
    config = {
        "participants": {
            "default_ollama": {"type": "ollama", "model": "qwen"},
            "loopback_vllm": {
                "type": "openai_compatible",
                "model": "qwen",
                "base_url": "http://127.0.0.1:8000/v1",
            },
            "lan_ollama": {
                "type": "ollama",
                "model": "qwen",
                "base_url": "http://10.0.0.5:11434",
            },
            "public_ollama": {
                "type": "ollama",
                "model": "qwen",
                "base_url": "https://ollama.example.com",
            },
        },
        "modes": {"private-local": {"strategy": "local_only_peers"}},
    }

    selected = select_participants(config, "private-local", current=None)

    assert selected == ["default_ollama"]


def test_private_local_explicit_roster_rejects_loopback_gateway() -> None:
    config = {
        "participants": {
            "gateway": {
                "type": "openai_compatible",
                "model": "qwen",
                "base_url": "http://127.0.0.1:8000/v1",
            }
        },
        "modes": {"private-local": {"participants": ["gateway"]}},
    }

    with pytest.raises(ValueError, match="only loopback.*ollama.*gateway"):
        select_participants(config, "private-local", current=None)


def _estimate_for(participant: dict, tmp_path: Path) -> dict:
    config = {
        "defaults": {"mode": "test"},
        "participants": {"peer": participant},
        "modes": {"test": {"participants": ["peer"]}},
    }
    return estimate_council(
        config=config,
        cwd=tmp_path,
        question="Review this change",
        mode="test",
        current=None,
        allow_network=False,
    )


def test_remote_ollama_is_unpriced_not_local_zero(tmp_path: Path) -> None:
    preflight = _estimate_for(
        {
            "type": "ollama",
            "model": "remote:qwen",
            "base_url": "http://10.0.0.5:11434",
        },
        tmp_path,
    )

    [row] = preflight["rows"]
    assert row["pricing_source"] is None
    assert row["estimated_total_cost_usd"] is None
    assert "Remote Ollama endpoint pricing is not configured" in row["note"]
    assert preflight["unknown_cost_rows"] == ["peer"]
    assert preflight["paid_peer_count"] == 1
    assert preflight["free_peer_count"] == 0
    _, _, unpriced = summarize_preflight_caps(preflight)
    assert unpriced == ["peer"]
    with pytest.raises(ValueError, match="hosted peer.*peer"):
        enforce_preflight_caps(
            preflight,
            max_cost_usd=1.0,
            max_tokens=None,
            breakdown_hint="Inspect the estimate.",
        )


def test_default_ollama_remains_local_zero(tmp_path: Path) -> None:
    preflight = _estimate_for(
        {"type": "ollama", "model": "qwen"},
        tmp_path,
    )

    [row] = preflight["rows"]
    assert row["pricing_source"] == "local"
    assert row["estimated_total_cost_usd"] == 0.0
    assert preflight["unknown_cost_rows"] == []
    assert preflight["paid_peer_count"] == 0
    assert preflight["free_peer_count"] == 1


def test_mcp_budget_fails_closed_for_remote_ollama() -> None:
    remote = {
        "type": "ollama",
        "model": "remote:qwen",
        "base_url": "https://ollama.example.com",
    }
    report = mcp_budget_report(
        config={"participants": {"remote": remote}, "defaults": {}},
        participants=["remote"],
        prompt_chars=1_000,
        deliberate=False,
        max_rounds=1,
    )

    assert report["paid_hosted_participants"] == ["remote"]
    assert report["within_budget"] is False
    assert report["violations"] == [
        {
            "limit": "known_paid_hosted_pricing",
            "actual": "remote",
            "maximum": "configured input_per_million or cached catalog price",
            "participants": ["remote"],
        }
    ]


def test_mcp_budget_keeps_default_ollama_free() -> None:
    local = {"type": "ollama", "model": "qwen"}
    report = mcp_budget_report(
        config={"participants": {"local": local}, "defaults": {}},
        participants=["local"],
        prompt_chars=1_000,
        deliberate=False,
        max_rounds=1,
    )

    assert report["paid_hosted_participants"] == []
    assert report["within_budget"] is True


def test_doctor_reports_explicit_openai_compatible_key_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCAL_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = {
        "participants": {
            "local": {
                "type": "openai_compatible",
                "model": "qwen",
                "base_url": "http://127.0.0.1:8000/v1",
                "api_key_env": "LOCAL_OPENAI_API_KEY",
            },
            "unkeyed": {
                "type": "openai_compatible",
                "model": "qwen",
                "base_url": "http://127.0.0.1:8001/v1",
            },
        }
    }

    by_name = {check.name: check for check in check_environment(config)}

    assert by_name["env:LOCAL_OPENAI_API_KEY"] == Check(
        "env:LOCAL_OPENAI_API_KEY", False, "not set"
    )
    assert by_name["env:OPENROUTER_API_KEY"] == Check(
        "env:OPENROUTER_API_KEY", False, "not set"
    )


def test_doctor_requires_explicit_endpoint_key_only_on_default_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "defaults": {"mode": "native"},
        "participants": {
            "keyed": {
                "type": "openai_compatible",
                "model": "qwen",
                "base_url": "http://127.0.0.1:8000/v1",
                "api_key_env": "LOCAL_OPENAI_API_KEY",
            },
            "native": {"type": "cli", "command": "example"},
        },
        "modes": {
            "keyed": {"participants": ["keyed"]},
            "native": {"participants": ["native"]},
        },
    }
    checks = [
        Check("env:LOCAL_OPENAI_API_KEY", False, "not set"),
        Check("cli:native", True, "/usr/bin/example"),
        Check("python:mcp", True, "installed"),
    ]
    monkeypatch.setattr(cli_module, "load_project_env", lambda *_a, **_k: [])
    monkeypatch.setattr(cli_module, "load_config", lambda *_a, **_k: config)
    monkeypatch.setattr(cli_module, "check_environment", lambda *_a, **_k: list(checks))
    args = argparse.Namespace(
        config=None,
        json=False,
        probe_openrouter=False,
        probe_ollama=False,
        probe_local_openai=None,
        check_update=False,
    )

    assert cli_module.cmd_doctor(args) == 0
    config["defaults"]["mode"] = "keyed"
    assert cli_module.cmd_doctor(args) == 1


def test_doctor_requires_default_key_for_unkeyed_openai_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "defaults": {"mode": "local"},
        "participants": {
            "local": {
                "type": "openai_compatible",
                "model": "qwen",
                "base_url": "http://127.0.0.1:8000/v1",
            }
        },
        "modes": {"local": {"participants": ["local"]}},
    }
    checks = [
        Check("env:OPENROUTER_API_KEY", False, "not set"),
        Check("python:mcp", True, "installed"),
    ]
    monkeypatch.setattr(cli_module, "load_project_env", lambda *_a, **_k: [])
    monkeypatch.setattr(cli_module, "load_config", lambda *_a, **_k: config)
    monkeypatch.setattr(cli_module, "check_environment", lambda *_a, **_k: checks)
    args = argparse.Namespace(
        config=None,
        json=False,
        probe_openrouter=False,
        probe_ollama=False,
        probe_local_openai=None,
        check_update=False,
    )

    assert cli_module.cmd_doctor(args) == 1


def test_local_openai_client_and_label_retry_ignore_proxy_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_options: list[dict] = []
    request_count = 0

    class FakeResponse:
        status_code = 200

        def __init__(self, content: str) -> None:
            self._content = content

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "model": "qwen",
                "choices": [
                    {
                        "message": {"content": self._content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            client_options.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def request(self, *_args, **_kwargs):
            nonlocal request_count
            request_count += 1
            if request_count == 1:
                return FakeResponse("Reasoning without the required label")
            return FakeResponse("RECOMMENDATION: yes - fixed")

    monkeypatch.setenv("LOCAL_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example.invalid")
    monkeypatch.setattr(adapters_module.httpx, "AsyncClient", FakeClient)

    result = asyncio.run(
        adapters_module._run_openai_compatible_inner(
            "local",
            {
                "type": "openai_compatible",
                "model": "qwen",
                "base_url": "http://127.0.0.1:8000/v1",
                "api_key_env": "LOCAL_OPENAI_API_KEY",
                "require_sections": False,
            },
            "Review this",
        )
    )

    assert result.ok is True
    assert len(client_options) == 2
    assert [options["trust_env"] for options in client_options] == [False, False]


def test_local_ollama_client_and_label_retry_ignore_proxy_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_options: list[dict] = []
    request_count = 0

    class FakeResponse:
        status_code = 200

        def __init__(self, content: str) -> None:
            self._content = content

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "message": {"content": self._content},
                "done_reason": "stop",
            }

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            client_options.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def request(self, *_args, **_kwargs):
            nonlocal request_count
            request_count += 1
            if request_count == 1:
                return FakeResponse("Reasoning without the required label")
            return FakeResponse("RECOMMENDATION: yes - fixed")

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.invalid")
    monkeypatch.setattr(adapters_module.httpx, "AsyncClient", FakeClient)

    result = asyncio.run(
        adapters_module._run_ollama_inner(
            "local",
            {
                "type": "ollama",
                "model": "qwen",
                "require_sections": False,
            },
            "Review this",
        )
    )

    assert result.ok is True
    assert len(client_options) == 2
    assert [options["trust_env"] for options in client_options] == [False, False]


def test_hosted_client_preserves_proxy_env_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_options: list[dict] = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "model": "vendor/model",
                "choices": [
                    {
                        "message": {"content": "RECOMMENDATION: yes - ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            client_options.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def request(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setenv("HOSTED_API_KEY", "secret")
    monkeypatch.setattr(adapters_module.httpx, "AsyncClient", FakeClient)

    result = asyncio.run(
        adapters_module._run_openai_compatible_inner(
            "hosted",
            {
                "type": "openai_compatible",
                "model": "vendor/model",
                "base_url": "https://api.example.com/v1",
                "api_key_env": "HOSTED_API_KEY",
                "retries": 0,
                "require_sections": False,
            },
            "Review this",
        )
    )

    assert result.ok is True
    assert client_options[0]["trust_env"] is True


def test_local_preflight_ignores_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    client_options: list[dict] = []

    class FakeResponse:
        status_code = 200

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            client_options.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.invalid")
    monkeypatch.setattr(orchestrator_module.httpx, "AsyncClient", FakeClient)

    error = asyncio.run(
        orchestrator_module._preflight_one(
            "lan",
            {
                "type": "openai_compatible",
                "base_url": "http://10.0.0.5:8000/v1",
            },
        )
    )

    assert error is None
    assert client_options[0]["trust_env"] is False
