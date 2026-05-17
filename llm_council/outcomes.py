"""User-marked outcomes for council runs.

Operators tag a finished run with `llm-council outcome mark <run-id>
--decision shipped|reverted|rejected|unknown [--bug-found yes|no]
[--winning-peer <peer>] [--note <text>]`. Records persist as sidecar
JSON under `.llm-council/outcomes/<run-id>.json` so transcripts stay
immutable (transcript shape was promised stable in v0.7.x).

Stats consume these via `aggregate` to compute per-peer reliability
counters. No reliability math beyond simple counts in v0.8 — IRT-style
scoring waits for >=200 marked outcomes per the plan's out-of-scope list.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from llm_council.transcript import normalize_run_id


SCHEMA_VERSION = 1

Decision = Literal["shipped", "reverted", "rejected", "unknown"]

_VALID_DECISIONS: frozenset[str] = frozenset(
    {"shipped", "reverted", "rejected", "unknown"}
)


@dataclass
class OutcomeRecord:
    """One operator-tagged outcome for a finished council run.

    `marked_at` defaults to "now in UTC" so callers do not have to
    construct a datetime. Persisted as ISO-8601 with offset to keep
    round-trips lossless across timezones.
    """

    run_id: str
    decision: Decision
    bug_found: bool | None = None
    winning_peer: str | None = None
    note: str | None = None
    marked_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "decision": self.decision,
            "bug_found": self.bug_found,
            "winning_peer": self.winning_peer,
            "note": self.note,
            "marked_at": self.marked_at.isoformat(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "OutcomeRecord | None":
        """Reconstruct a record from on-disk JSON. Returns None on shape errors.

        Schema-tolerant: missing optional fields default to None; unknown
        decisions reject as malformed (better to surface than silently
        coerce). Older `schema_version` values that round-trip cleanly
        through the field set still load.
        """
        if not isinstance(payload, dict):
            return None
        run_id = payload.get("run_id")
        decision = payload.get("decision")
        marked_at_raw = payload.get("marked_at")
        if not isinstance(run_id, str) or not run_id:
            return None
        if decision not in _VALID_DECISIONS:
            return None
        if not isinstance(marked_at_raw, str) or not marked_at_raw:
            return None
        try:
            marked_at = datetime.fromisoformat(marked_at_raw)
        except ValueError:
            return None
        bug_found = payload.get("bug_found")
        if bug_found is not None and not isinstance(bug_found, bool):
            return None
        winning_peer = payload.get("winning_peer")
        if winning_peer is not None and not isinstance(winning_peer, str):
            return None
        note = payload.get("note")
        if note is not None and not isinstance(note, str):
            return None
        return cls(
            run_id=run_id,
            decision=decision,  # type: ignore[arg-type]
            bug_found=bug_found,
            winning_peer=winning_peer,
            note=note,
            marked_at=marked_at,
        )


def outcomes_dir(cwd: Path) -> Path:
    """Return `.llm-council/outcomes/` under cwd; create if missing."""
    target = cwd / ".llm-council" / "outcomes"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _outcome_path(cwd: Path, run_id: str) -> Path:
    return outcomes_dir(cwd) / f"{run_id}.json"


def write_outcome(cwd: Path, record: OutcomeRecord) -> Path:
    """Persist one record. Idempotent — overwrites by run_id.

    Returns the path written. Caller is responsible for resolving
    `record.run_id` to the canonical full filename via `resolve_run_id`
    BEFORE constructing the record; this function does not normalize.
    """
    path = _outcome_path(cwd, record.run_id)
    payload = record.to_payload()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_outcome(cwd: Path, run_id: str) -> OutcomeRecord | None:
    """Load one record by run_id, or None if missing or malformed."""
    path = _outcome_path(cwd, run_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return OutcomeRecord.from_payload(payload)


def iter_outcomes(cwd: Path) -> Iterator[OutcomeRecord]:
    """Yield all outcomes in the outcomes_dir, sorted by marked_at desc.

    Malformed records are skipped silently — they survive on disk for
    operator inspection, but do not poison the iteration.
    """
    target = cwd / ".llm-council" / "outcomes"
    if not target.is_dir():
        return
    records: list[OutcomeRecord] = []
    for path in target.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        record = OutcomeRecord.from_payload(payload)
        if record is not None:
            records.append(record)
    records.sort(key=lambda r: r.marked_at, reverse=True)
    yield from records


def resolve_run_id(cwd: Path, run_id_or_prefix: str) -> str | None:
    """Resolve a partial run_id (timestamp prefix) against transcripts.

    Mirrors how `cli` resolves IDs via
    `llm_council.transcript.find_transcript_by_id` — but where that helper
    raises on ambiguous/missing matches, this function returns None so
    callers can produce nice error messages without try/except plumbing.

    Returns the canonical run_id (filename stem of the matched transcript
    JSON) or None if zero or multiple matches are found.

    Resolution is purely textual against the transcripts dir; we do not
    require an outcome to already exist (mark-before-list is the common
    workflow). Pass the FULL prefix when ambiguous — same rule as
    `--continue`.
    """
    if not run_id_or_prefix:
        return None
    # Load the config-driven transcripts dir; fall back to the default
    # path. We avoid importing config.load_config here to keep this
    # module dependency-light (the CLI does the heavy resolution for us).
    transcripts = cwd / ".llm-council" / "runs"
    try:
        normalized = normalize_run_id(run_id_or_prefix)
    except ValueError:
        return None
    if not transcripts.is_dir():
        return None
    candidates = sorted(transcripts.glob(f"{normalized}*.json"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0].stem
    # Multiple matches: accept only an exact stem hit.
    for path in candidates:
        if path.stem == normalized:
            return path.stem
    return None
