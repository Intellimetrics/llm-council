"""Run blocking preparation off the event loop with cooperative cancellation."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from contextlib import contextmanager
import os
import signal
import subprocess
import threading
import time
from typing import Any, Callable, TypeVar

_STOP: ContextVar[threading.Event | None] = ContextVar("council_worker_stop", default=None)
_T = TypeVar("_T")
PEER_DEADLINE: ContextVar[float | None] = ContextVar("council_peer_deadline", default=None)


@contextmanager
def peer_deadline(deadline: float):
    token = PEER_DEADLINE.set(deadline)
    try:
        yield
    finally:
        PEER_DEADLINE.reset(token)


def deadline_reached() -> bool:
    deadline = PEER_DEADLINE.get()
    return deadline is not None and time.monotonic() >= deadline


def kill_process_group(pid: int) -> None:
    """Kill a POSIX session owned by this invocation, including descendants."""
    if os.name == "posix":
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def check_cancelled() -> None:
    stop = _STOP.get()
    if stop is not None and stop.is_set():
        raise asyncio.CancelledError


async def run_blocking(function: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """Wait for worker cleanup before propagating cancellation to the caller.

    ContextVars (including the project environment) travel with to_thread.
    Long-running subprocesses use run_process below to observe cancellation.
    """
    stop = threading.Event()

    def invoke() -> _T:
        check_cancelled()
        return function(*args, **kwargs)

    token = _STOP.set(stop)
    try:
        worker = asyncio.create_task(asyncio.to_thread(invoke))
    finally:
        _STOP.reset(token)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        stop.set()
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                # A second cancellation must not detach cleanup either.
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()  # Retrieve failures while preserving cancellation.
        raise


def run_process(command: list[str], *, timeout: float, **kwargs: Any) -> subprocess.CompletedProcess:
    """Wait on a captured subprocess; kill and reap it on timeout/cancellation.

    Callers own the disk-backed output streams. POSIX children get their own
    process group so a generator's descendants cannot outlive its tempdir.
    """
    check_cancelled()
    deadline = time.monotonic() + timeout
    with subprocess.Popen(command, start_new_session=os.name == "posix", **kwargs) as proc:
        try:
            while True:
                check_cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    code = proc.wait(timeout=min(0.1, remaining))
                    return subprocess.CompletedProcess(command, code)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # Successful parent exit does not imply its children exited.
            # Kill the owned group on every path before releasing temp output.
            try:
                if os.name == "posix":
                    kill_process_group(proc.pid)
                elif proc.poll() is None:
                    proc.kill()
            except ProcessLookupError:
                pass
            proc.wait()
