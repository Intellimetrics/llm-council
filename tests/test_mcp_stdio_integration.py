"""Real MCP stdio round-trip test.

This is the integration counterpart to the in-process drift-guard
(``test_mcp_structured_results_keys_match_declared_schema``): it spawns the
actual ``llm_council.mcp_server`` over stdio, performs the JSON-RPC initialize +
list_tools + call_tool handshake on the wire, and asserts the
``council_run`` response carries the advertised ``schema_version`` and the
schema-required fields.

Why it matters: prior schema bumps relied on manual dogfooding after a server
restart. This test exercises the same transport automatically, so a
schema/transport regression is caught by CI instead of by hand.

Uses ``dry_run`` (no peer subprocesses / HTTP) and a local-peer config so it
does not depend on any CLI being installed in the test environment. The whole
exchange is bounded by a wall-clock timeout so a server that fails to start
fails the test rather than hanging the suite.
"""

from __future__ import annotations

import asyncio
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
    env = os.environ.copy()
    # Make the subprocess import the same tree the test runs against, and pin
    # the MCP root so working_directory passes the containment check.
    repo_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    env["LLM_COUNCIL_MCP_ROOT"] = str(root)

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "llm_council.mcp_server",
        cwd=str(root),
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # The tools/list response includes the full council_run schema and can
        # exceed asyncio's conservative 64 KiB default line limit.
        limit=1024 * 1024,
    )

    unexpected_messages: list[dict] = []

    async def _request(request_id: int, method: str, params: dict) -> dict:
        assert proc.stdin is not None
        assert proc.stdout is not None
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        proc.stdin.write((json.dumps(message) + "\n").encode())
        await proc.stdin.drain()
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                stderr = (
                    (await proc.stderr.read()).decode(errors="replace")
                    if proc.stderr is not None
                    else ""
                )
                raise AssertionError(
                    f"MCP server exited before response {request_id}: {stderr}"
                )
            response = json.loads(raw)
            # Ignore server notifications; return the matching response.
            if response.get("id") != request_id:
                unexpected_messages.append(response)
                continue
            if "error" in response:
                raise AssertionError(
                    f"MCP request {method} failed: {response['error']}"
                )
            return response["result"]

    shutdown_timed_out = False
    try:
        async with asyncio.timeout(45):
            initialized = await _request(
                1,
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "llm-council-tests", "version": "1"},
                },
            )
            assert initialized.get("serverInfo", {}).get("name") == "llm-council"

            assert proc.stdin is not None
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
            proc.stdin.write((json.dumps(notification) + "\n").encode())
            await proc.stdin.drain()

            listed = await _request(2, "tools/list", {})
            tool_names = {tool["name"] for tool in listed["tools"]}
            called = await _request(
                3,
                "tools/call",
                {
                    "name": "council_run",
                    "arguments": {
                        "question": "stdio integration probe",
                        "working_directory": str(root),
                        "dry_run": True,
                    },
                },
            )
            payload = _extract_payload(called)
    except TimeoutError as exc:
        buffered = len(getattr(proc.stdout, "_buffer", b""))
        unexpected_summary = json.dumps(
            unexpected_messages[-3:],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        raise AssertionError(
            "MCP stdio exchange timed out with "
            f"{buffered} buffered byte(s); "
            f"last unexpected message(s): {unexpected_summary[-4000:]}"
        ) from exc
    finally:
        # Closing the OS pipe is the MCP stdio shutdown signal. Do this
        # explicitly instead of relying on mcp-python's stdio_client context:
        # its Windows cleanup can hang in process.wait() after a valid response.
        if proc.stdin is not None:
            proc.stdin.close()
            try:
                await asyncio.wait_for(proc.stdin.wait_closed(), timeout=3)
            except (OSError, TimeoutError):
                pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            shutdown_timed_out = True
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()

    assert not shutdown_timed_out, "MCP server did not exit after stdio closed"
    assert proc.returncode == 0
    return tool_names, payload


def _extract_payload(result) -> dict:
    """Pull the council_run dict from a typed or raw MCP call result."""
    structured = (
        result.get("structuredContent")
        if isinstance(result, dict)
        else getattr(result, "structuredContent", None)
    )
    if isinstance(structured, dict) and structured:
        # The low-level server may wrap a bare return under a "result" key.
        if "schema_version" in structured:
            return structured
        for value in structured.values():
            if isinstance(value, dict) and "schema_version" in value:
                return value
    content = (
        result.get("content", [])
        if isinstance(result, dict)
        else getattr(result, "content", [])
    )
    for block in content or []:
        text = (
            block.get("text")
            if isinstance(block, dict)
            else getattr(block, "text", None)
        )
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


@pytest.mark.asyncio
async def test_mcp_server_stdio_council_config(tmp_path: Path):
    (tmp_path / ".llm-council.yaml").write_text(_LOCAL_CONFIG, encoding="utf-8")

    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    env["LLM_COUNCIL_MCP_ROOT"] = str(tmp_path)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "llm_council.mcp_server"],
        env=env,
    )

    async def _exchange() -> tuple[dict, dict]:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Test council_config get
                res_get = await session.call_tool(
                    "council_config",
                    {
                        "action": "get",
                        "key": "defaults.mode",
                        "working_directory": str(tmp_path),
                    },
                )
                payload_get = json.loads(res_get.content[0].text)
                
                # Test council_config set
                res_set = await session.call_tool(
                    "council_config",
                    {
                        "action": "set",
                        "key": "defaults.auto_open_browser",
                        "value": "true",
                        "working_directory": str(tmp_path),
                    },
                )
                payload_set = json.loads(res_set.content[0].text)
                
                return payload_get, payload_set

    with anyio.fail_after(45):
        payload_get, payload_set = await _exchange()

    assert payload_get["key"] == "defaults.mode"
    assert payload_get["value"] == "review-local"
    assert payload_get["success"] is True

    assert payload_set["key"] == "defaults.auto_open_browser"
    assert payload_set["value"] is True
    assert payload_set["success"] is True

    # Verify YAML was actually written
    import yaml
    config = yaml.safe_load((tmp_path / ".llm-council.yaml").read_text(encoding="utf-8"))
    assert config["defaults"]["auto_open_browser"] is True
