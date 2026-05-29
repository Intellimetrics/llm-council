"""Real MCP stdio round-trip test.

This is the integration counterpart to the in-process drift-guard
(``test_mcp_structured_results_keys_match_declared_schema``): it spawns the
actual ``llm_council.mcp_server`` over stdio, performs the JSON-RPC initialize +
list_tools + call_tool handshake through the `mcp` client, and asserts the
``council_run`` response carries the advertised ``schema_version`` and the
schema-required fields.

Why it matters: every prior schema bump (v5 -> v6 this release) was validated by
MANUAL dogfooding after a server restart (see MEMORY: MCP-restart dogfood). This
test exercises the same transport automatically, so a schema/transport
regression is caught by CI instead of by hand.

Uses ``dry_run`` (no peer subprocesses / HTTP) and a local-peer config so it
does not depend on any CLI being installed in the test environment. The whole
exchange is bounded by a wall-clock timeout so a server that fails to start
fails the test rather than hanging the suite.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from llm_council.mcp_server import COUNCIL_RUN_OUTPUT_SCHEMA_VERSION

_LOCAL_CONFIG = """
defaults:
  mode: review-local
participants:
  local_peer:
    type: ollama
    model: llama3
    base_url: http://localhost:11434
modes:
  review-local:
    participants: [local_peer]
""".lstrip()


async def _call_council_run_over_stdio(root: Path) -> tuple[set[str], dict]:
    """Spawn the server over stdio, list tools, and dry-run council_run.

    Returns (tool_names, council_run_payload).
    """
    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = os.environ.copy()
    # Make the subprocess import the same tree the test runs against, and pin
    # the MCP root so working_directory passes the containment check.
    repo_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    env["LLM_COUNCIL_MCP_ROOT"] = str(root)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "llm_council.mcp_server"],
        env=env,
    )

    async def _exchange() -> tuple[set[str], dict]:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                tool_names = {t.name for t in listed.tools}
                result = await session.call_tool(
                    "council_run",
                    {
                        "question": "stdio integration probe",
                        "working_directory": str(root),
                        "dry_run": True,
                    },
                )
                payload = _extract_payload(result)
                return tool_names, payload

    with anyio.fail_after(45):
        return await _exchange()


def _extract_payload(result) -> dict:
    """Pull the council_run dict from a CallToolResult — prefer the structured
    content, fall back to parsing the first text block as JSON."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict) and structured:
        # The low-level server may wrap a bare return under a "result" key.
        if "schema_version" in structured:
            return structured
        for value in structured.values():
            if isinstance(value, dict) and "schema_version" in value:
                return value
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except (ValueError, TypeError):
                continue
    raise AssertionError(f"could not extract council_run payload from {result!r}")


@pytest.mark.asyncio
async def test_mcp_server_stdio_round_trip_council_run(tmp_path: Path):
    (tmp_path / ".llm-council.yaml").write_text(_LOCAL_CONFIG, encoding="utf-8")

    tool_names, payload = await _call_council_run_over_stdio(tmp_path)

    # The tool surface is registered and reachable over the real transport.
    assert "council_run" in tool_names
    assert "council_doctor" in tool_names

    # The dry-run envelope carries the advertised schema version + the fields a
    # strict MCP client requires. This is exactly what the manual dogfood checks.
    assert payload["schema_version"] == COUNCIL_RUN_OUTPUT_SCHEMA_VERSION
    for required in ("recommendation", "agreement_count", "degraded", "rounds", "results"):
        assert required in payload, f"missing {required!r} in stdio council_run payload"
    assert payload["metadata"]["dry_run"] is True
