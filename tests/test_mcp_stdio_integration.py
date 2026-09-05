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

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from llm_council.mcp_server import COUNCIL_RUN_OUTPUT_SCHEMA_VERSION

def test_stdio_deadline_returns_partial_transcript(tmp_path: Path):
    import yaml
    cfg = {"participants": {
        "fast": {"type": "cli", "family": "custom", "command": sys.executable,
                 "args": ["-c", "print('RECOMMENDATION: yes')"], "timeout": 60},
        "slow": {"type": "cli", "family": "custom", "command": sys.executable,
                 "args": ["-c", "import time; time.sleep(30)"], "timeout": 60},
    }}
    (tmp_path / ".llm-council.yaml").write_text(yaml.safe_dump(cfg))
    started = time.monotonic()
    _, payload, _ = _call_council_run_over_stdio(tmp_path, arguments={
        "working_directory": str(tmp_path), "current": "codex", "mode": "quick",
        "participants": ["fast", "slow"], "min_quorum": 1, "question": "fixture",
        "request_timeout_seconds": 2, "deliberate": False,
    })
    assert time.monotonic() - started < 6
    assert payload["metadata"]["partial"]
    assert payload["results"][0]["ok"]
    assert payload["results"][1]["error_kind"] == "timeout"
    assert "Partial result" in payload["summary_markdown"]
    assert "Partial result" in Path(payload["transcript"]).read_text()


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


def _call_council_run_over_stdio(
    root: Path,
    *,
    arguments: dict | None = None,
    meta: dict | None = None,
) -> tuple[set[str], dict, list[dict]]:
    """Spawn the server over stdio, list tools, and call council_run.

    `arguments` defaults to a dry-run probe; pass explicit arguments (and
    optionally a request `_meta`, e.g. a progressToken) to exercise other
    paths. Returns (tool_names, council_run_payload, other_messages) where
    other_messages holds every frame that wasn't a direct response —
    notifications included — in arrival order.
    """
    env = os.environ.copy()
    # Make the subprocess import the same tree the test runs against, and pin
    # the MCP root so working_directory passes the containment check.
    repo_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    env["LLM_COUNCIL_MCP_ROOT"] = str(root)

    proc = subprocess.Popen(
        [sys.executable, "-m", "llm_council.mcp_server"],
        cwd=str(root),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout_lines: queue.Queue[bytes | None] = queue.Queue()
    stderr_chunks: list[bytes] = []
    unexpected_messages: list[dict] = []
    deadline = time.monotonic() + 45

    def _read_stdout() -> None:
        assert proc.stdout is not None
        try:
            while raw := proc.stdout.readline():
                stdout_lines.put(raw)
        finally:
            stdout_lines.put(None)

    def _read_stderr() -> None:
        assert proc.stderr is not None
        while chunk := proc.stderr.read(4096):
            stderr_chunks.append(chunk)

    stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    def _stderr_tail() -> str:
        return b"".join(stderr_chunks).decode(errors="replace")[-4000:]

    def _request(request_id: int, method: str, params: dict) -> dict:
        assert proc.stdin is not None
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        proc.stdin.write((json.dumps(message) + "\n").encode())
        proc.stdin.flush()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            try:
                raw = stdout_lines.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError from exc
            if raw is None:
                raise AssertionError(
                    f"MCP server exited before response {request_id}: {_stderr_tail()}"
                )
            response = json.loads(raw.decode("utf-8", errors="replace"))
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
        initialized = _request(
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
        proc.stdin.flush()

        listed = _request(2, "tools/list", {})
        tool_names = {tool["name"] for tool in listed["tools"]}
        call_params: dict = {
            "name": "council_run",
            "arguments": arguments
            or {
                "question": "stdio integration probe",
                "working_directory": str(root),
                "dry_run": True,
            },
        }
        if meta is not None:
            call_params["_meta"] = meta
        called = _request(3, "tools/call", call_params)
        payload = _extract_payload(called)
    except TimeoutError as exc:
        unexpected_summary = json.dumps(
            unexpected_messages[-3:],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        raise AssertionError(
            "MCP stdio exchange timed out; "
            f"server stderr: {_stderr_tail()!r}; "
            f"last unexpected message(s): {unexpected_summary[-4000:]}"
        ) from exc
    finally:
        # Closing the OS pipe is the MCP stdio shutdown signal. Do this
        # explicitly instead of relying on mcp-python's stdio_client context:
        # its Windows cleanup can hang in process.wait() after a valid response.
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            shutdown_timed_out = True
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=5)
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)

    assert not shutdown_timed_out, "MCP server did not exit after stdio closed"
    assert proc.returncode == 0
    return tool_names, payload, unexpected_messages


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


def test_mcp_server_stdio_round_trip_council_run(tmp_path: Path):
    (tmp_path / ".llm-council.yaml").write_text(_LOCAL_CONFIG, encoding="utf-8")

    tool_names, payload, _notifications = _call_council_run_over_stdio(tmp_path)

    # The tool surface is registered and reachable over the real transport.
    assert "council_run" in tool_names
    assert "council_doctor" in tool_names

    # The dry-run envelope carries the advertised schema version + the fields a
    # strict MCP client requires. This is exactly what the manual dogfood checks.
    assert payload["schema_version"] == COUNCIL_RUN_OUTPUT_SCHEMA_VERSION
    for required in ("recommendation", "agreement_count", "degraded", "rounds", "results"):
        assert required in payload, f"missing {required!r} in stdio council_run payload"
    assert payload["metadata"]["dry_run"] is True


def test_mcp_server_stdio_progress_notifications(tmp_path: Path):
    """A real (non-dry) council_run with a client progressToken must emit
    notifications/progress frames on the wire.

    This pins the mcp 2.x progress path end-to-end: token extraction from
    the handler context (``ctx.meta["progress_token"]``) through
    ``ServerSession.send_progress_notification``. Dry-run returns before
    ``execute_council`` and never emits progress, so the run is real — but
    its only peer points at a reserved-then-released local port, so
    preflight fails instantly and no backend is needed. ``council_start``
    always formats a progress message, guaranteeing at least one frame.
    """
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    config = _LOCAL_CONFIG.replace(
        "http://localhost:11434", f"http://127.0.0.1:{port}"
    )
    (tmp_path / ".llm-council.yaml").write_text(config, encoding="utf-8")

    token = "probe-progress-1"
    _tools, payload, notifications = _call_council_run_over_stdio(
        tmp_path,
        arguments={
            "question": "stdio progress probe",
            "working_directory": str(tmp_path),
        },
        meta={"progressToken": token},
    )

    progress_frames = [
        frame
        for frame in notifications
        if frame.get("method") == "notifications/progress"
        and (frame.get("params") or {}).get("progressToken") == token
    ]
    assert progress_frames, (
        "no notifications/progress carried the client's progressToken; "
        f"other frames: {notifications[:5]!r}"
    )
    for frame in progress_frames:
        assert isinstance(frame["params"].get("progress"), (int, float))

    # The run itself completed degraded (its only peer is unreachable) but
    # still shipped a schema-valid envelope.
    assert payload["schema_version"] == COUNCIL_RUN_OUTPUT_SCHEMA_VERSION
    assert payload["degraded"] is True


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
