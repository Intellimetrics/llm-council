"""Shared fake-subprocess stubs for adapter tests.

One canonical fake for ``asyncio.create_subprocess_exec`` as seen by
``llm_council.adapters``, shared by test_usage_from_json.py and
test_fable_mode.py (previously byte-identical private copies). A change to
the adapter's subprocess invocation — e.g. a streamed-read path replacing
``proc.communicate`` — is now fixed here once instead of once per test file.

``FakeProc`` mirrors the touchpoints ``_run_cli_once`` uses on a successful
call (``communicate`` / ``wait`` / ``returncode``); ``TimingOutProc`` adds
the ``terminate`` / ``kill`` pair exercised by
``_cleanup_timed_out_process`` on the timeout path.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

_PATCH_TARGET = "llm_council.adapters.asyncio.create_subprocess_exec"


class FakeProc:
    """Successful one-shot subprocess: fixed stdout, exit 0."""

    returncode = 0

    def __init__(self, stdout: str):
        self._stdout = stdout

    async def communicate(self, _data=None):
        return (self._stdout.encode(), b"")

    async def wait(self):
        return 0


class TimingOutProc:
    """Subprocess whose ``communicate`` never returns until it is killed.

    Blocks on an event that ``terminate`` / ``kill`` set, so the adapter's
    ``wait_for`` trips its wall-clock timeout and the cleanup path completes
    promptly instead of sleeping out a fixed hang.
    """

    def __init__(self) -> None:
        self.returncode: int | None = None
        self._dead = asyncio.Event()

    async def communicate(self, _data=None):
        await self._dead.wait()
        return (b"", b"")

    def terminate(self) -> None:
        self.returncode = -15
        self._dead.set()

    def kill(self) -> None:
        self.returncode = -9
        self._dead.set()

    async def wait(self) -> int:
        return self.returncode if self.returncode is not None else 0


def fake_proc_returning(stdout: str):
    """Patch context: every subprocess launch returns ``stdout``."""

    return patch(_PATCH_TARGET, new=AsyncMock(return_value=FakeProc(stdout)))


def fake_proc_sequence(*outputs):
    """Patch context: launch N serves ``outputs[N]`` — for retry flows.

    Each entry is either a stdout string (wrapped in ``FakeProc``) or a
    ready process object (e.g. ``TimingOutProc()`` for a timed-out call).
    """

    queue = iter(outputs)

    async def _factory(*_args, **_kwargs):
        item = next(queue)
        return FakeProc(item) if isinstance(item, str) else item

    return patch(_PATCH_TARGET, new=_factory)
