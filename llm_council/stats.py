"""Aggregate statistics over council transcripts.

Pure read: scans `.llm-council/runs/*.json` and computes per-participant and
aggregate metrics. Backs the `llm-council stats` CLI subcommand and the
`council_stats` MCP tool.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from llm_council.deliberation import recommendation_label
from llm_council.transcript import _existing_paths, result_round


_LABELS = ("yes", "no", "tradeoff", "unknown")
# Subset of the response envelope we track presence-of for the optional→required
# rollout decision. List fields count as "present" only when non-empty; scalars
# count when non-null. Keep this in sync with the envelope contract in adapters.
_ENVELOPE_FIELDS = (
    "effort",
    "confidence",
    "risk",
    "blockers",
    "evidence",
    "tests_to_run",
    "assumptions",
)
_ENVELOPE_LIST_FIELDS = frozenset({"blockers", "evidence", "tests_to_run", "assumptions"})

# Char-size buckets for timeout telemetry. Aligned with the realistic
# prompt-build outcomes given the default `max_prompt_chars: 120_000`:
# quick/peer-only typically lands under 4K; mid-sized council prompts
# with one context file fall 4K-20K (pass-7 at 14.3K is exactly this);
# plan/review with multiple --context files lands 20K-60K; full-codebase
# diff runs push xlarge.
TIMEOUT_PROMPT_SIZE_BUCKETS: tuple[tuple[str, int | None], ...] = (
    ("small", 4_000),
    ("medium", 20_000),
    ("large", 60_000),
    ("xlarge", None),  # None = no upper bound
)


def _timeout_prompt_size_bucket(prompt_chars: int | None) -> str:
    if prompt_chars is None or prompt_chars <= 0:
        return "small"
    for name, ceiling in TIMEOUT_PROMPT_SIZE_BUCKETS:
        if ceiling is None or prompt_chars <= ceiling:
            return name
    return "xlarge"


def load_transcript_files(base_dir: Path) -> list[dict[str, Any]]:
    """Return raw transcript JSON dicts plus their on-disk mtime.

    Records are sorted oldest-first. Unreadable / malformed files are skipped
    silently, mirroring `transcript.transcript_records`.
    """
    records: list[dict[str, Any]] = []
    for path, mtime in sorted(
        _existing_paths(base_dir.glob("*.json")), key=lambda item: item[1]
    ):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append({"path": str(path), "mtime": mtime, "data": data})
    return records


def _final_round_only(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []
    final = max(result_round(r.get("name", "")) for r in results)
    return [r for r in results if result_round(r.get("name", "")) == final]


def _empty_label_counts() -> dict[str, int]:
    return {label: 0 for label in _LABELS}


def _new_peer_bucket() -> dict[str, Any]:
    return {
        "runs": 0,
        "successes": 0,
        "elapsed_total": 0.0,
        "elapsed_runs": 0,
        "label_counts": _empty_label_counts(),
        "tokens_total": 0,
        "tokens_runs": 0,
        "cost_total": 0.0,
        "cost_runs": 0,
        "invalid_label_runs": 0,
        # error_kind -> count, populated from failed runs. Stable enum from
        # adapters.KNOWN_ERROR_KINDS; new kinds appear as new keys.
        "error_kind_counts": {},
        # Envelope field presence per peer. Prerequisite telemetry before
        # flipping optional envelope fields to required (Pick A rollout).
        "envelope_field_present": {field: 0 for field in _ENVELOPE_FIELDS},
        # Timeout-by-prompt-size telemetry. Lets the operator see whether
        # bigger prompts disproportionately trip the timeout wall, which
        # is the signal for raising `defaults.timeout` or a mode's
        # `timeout_multiplier` rather than chunking.
        "timeout_by_prompt_size": {
            name: 0 for name, _ in TIMEOUT_PROMPT_SIZE_BUCKETS
        },
        # Count of successful runs that recovered from a timeout via the
        # terse-retry path. Together with `timeout_by_prompt_size` this
        # tells the operator how often the recovery path actually saves
        # the run vs. how often the prompt is just too big.
        "timeout_recoveries": 0,
        # Count of runs where the terse-retry path fired at all (success
        # or failure). Pass-8 dogfood surfaced a silent-failure mode where
        # the retry attempt was invisible in transcripts: only successful
        # recoveries flipped a flag, so a transcript showing only the
        # original timeout looked identical to "retry never fired".
        # `terse_retry_attempts - timeout_recoveries` is the count of
        # retries that fired but also failed; if that number stays high
        # in a given prompt-size bucket, the 60s terse window itself
        # needs raising (or the mode multiplier needs to apply to retries).
        "terse_retry_attempts": 0,
        # Recoveries broken out by prompt-size bucket. Cross-tab with
        # `timeout_by_prompt_size`: a bucket where timeouts and recoveries
        # both spike means the wall is real but terse-retry handles it; a
        # bucket where timeouts climb but recoveries stay flat means the
        # prompt is genuinely too big and `defaults.timeout` /
        # `timeout_multiplier` need to go up. Populated from
        # `result.prompt_chars` on the recovered result (set on every
        # adapter return path, not just failures), with the same `small/
        # medium/large/xlarge` cutoffs as `timeout_by_prompt_size`.
        "timeout_recoveries_by_prompt_size": {
            name: 0 for name, _ in TIMEOUT_PROMPT_SIZE_BUCKETS
        },
        # Evidence-tag distribution per peer. Counts each EVIDENCE bullet
        # by its tag, plus an `untagged` bin for entries without one.
        # Drives the optional→required rollout decision for
        # `defaults.strict_evidence`: when untagged stays small across
        # representative runs, flip the default.
        "evidence_tag_distribution": {
            "published": 0,
            "observable": 0,
            "inferred": 0,
            "speculative": 0,
            "untagged": 0,
        },
        "last_used": None,
    }


def aggregate(
    records: list[dict[str, Any]],
    *,
    participant: str | None = None,
    since_seconds: float | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Compute participant-level and aggregate metrics from raw transcripts.

    `records` should be the output of `load_transcript_files` (each entry has
    `path`, `mtime`, `data`). `participant` filters the per-participant view to
    a single name (aggregate counts still cover all peers in the matched
    transcripts). `since_seconds` drops transcripts with `mtime` older than
    `now - since_seconds`.
    """
    now = time.time() if now is None else now
    cutoff = now - since_seconds if since_seconds is not None else None

    peers: dict[str, dict[str, Any]] = {}
    mode_counts: dict[str, int] = {}
    transcripts_considered = 0
    total_runs = 0
    total_successes = 0

    for entry in records:
        mtime = entry.get("mtime") or 0.0
        if cutoff is not None and mtime < cutoff:
            continue
        data = entry.get("data") or {}
        results = data.get("results") or []
        final_results = _final_round_only(results)
        transcripts_considered += 1
        mode = data.get("mode") or ""
        if mode:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        for result in results:
            raw_name = result.get("name") or ""
            name = raw_name.split(":round")[0] or "unknown"
            bucket = peers.setdefault(name, _new_peer_bucket())
            elapsed = result.get("elapsed_seconds")
            if elapsed is not None:
                try:
                    bucket["elapsed_total"] += float(elapsed)
                    bucket["elapsed_runs"] += 1
                except (TypeError, ValueError):
                    pass
            tokens = result.get("total_tokens")
            if tokens is not None:
                try:
                    bucket["tokens_total"] += int(tokens)
                    bucket["tokens_runs"] += 1
                except (TypeError, ValueError):
                    pass
            cost = result.get("cost_usd")
            if cost is not None:
                try:
                    bucket["cost_total"] += float(cost)
                    bucket["cost_runs"] += 1
                except (TypeError, ValueError):
                    pass
            if bucket["last_used"] is None or mtime > bucket["last_used"]:
                bucket["last_used"] = mtime

        seen_in_transcript: set[str] = set()
        for result in final_results:
            raw_name = result.get("name") or ""
            name = raw_name.split(":round")[0] or "unknown"
            if name in seen_in_transcript:
                continue
            seen_in_transcript.add(name)
            total_runs += 1
            ok = bool(result.get("ok"))
            if ok:
                total_successes += 1
            bucket = peers.setdefault(name, _new_peer_bucket())
            bucket["runs"] += 1
            # Count terse-retry attempts on every final-round result that
            # has the flag set, regardless of ok status. Recovered results
            # carry it (success path); annotated original-timeout results
            # carry it too (failure path). Mutually exclusive at the same
            # final-round entry: a single result is either a recovery or
            # an annotated failure, never both.
            if result.get("terse_retry_attempted"):
                bucket["terse_retry_attempts"] += 1
            if ok:
                bucket["successes"] += 1
                label = recommendation_label(result.get("output") or "")
                if label not in bucket["label_counts"]:
                    label = "unknown"
                bucket["label_counts"][label] += 1
                if label == "unknown":
                    bucket["invalid_label_runs"] += 1
                for field_name in _ENVELOPE_FIELDS:
                    value = result.get(field_name)
                    if field_name in _ENVELOPE_LIST_FIELDS:
                        if value:
                            bucket["envelope_field_present"][field_name] += 1
                    elif value is not None:
                        bucket["envelope_field_present"][field_name] += 1
                if result.get("recovered_after_timeout"):
                    bucket["timeout_recoveries"] += 1
                    recovery_prompt_chars = result.get("prompt_chars")
                    try:
                        recovery_bucket = _timeout_prompt_size_bucket(
                            int(recovery_prompt_chars)
                            if recovery_prompt_chars
                            else None
                        )
                    except (TypeError, ValueError):
                        recovery_bucket = "small"
                    bucket["timeout_recoveries_by_prompt_size"][
                        recovery_bucket
                    ] += 1
                # Evidence-tag distribution: count each EVIDENCE bullet
                # by its tag. Entries shape from v0.7 onward is
                # list[{text, tag}]; legacy list[str] entries (pre-v0.7
                # cached transcripts) count as untagged.
                for entry in result.get("evidence") or []:
                    if isinstance(entry, dict):
                        tag = entry.get("tag")
                        bucket_key = tag if tag in (
                            "published", "observable", "inferred", "speculative",
                        ) else "untagged"
                    else:
                        bucket_key = "untagged"
                    bucket["evidence_tag_distribution"][bucket_key] += 1
            else:
                error_kind = result.get("error_kind") or "unknown"
                bucket["error_kind_counts"][error_kind] = (
                    bucket["error_kind_counts"].get(error_kind, 0) + 1
                )
                if error_kind == "timeout":
                    prompt_chars = result.get("prompt_chars")
                    try:
                        size_bucket = _timeout_prompt_size_bucket(
                            int(prompt_chars) if prompt_chars else None
                        )
                    except (TypeError, ValueError):
                        size_bucket = "small"
                    bucket["timeout_by_prompt_size"][size_bucket] += 1

    participant_rows = []
    for name, bucket in sorted(peers.items()):
        if participant and name != participant:
            continue
        if bucket["runs"] == 0:
            continue
        runs = bucket["runs"]
        successes = bucket["successes"]
        label_counts = bucket["label_counts"]
        elapsed_runs = bucket["elapsed_runs"]
        tokens_runs = bucket["tokens_runs"]
        cost_runs = bucket["cost_runs"]
        participant_rows.append(
            {
                "name": name,
                "runs": runs,
                "successes": successes,
                "success_rate": (successes / runs) if runs else 0.0,
                "avg_elapsed_seconds": (
                    (bucket["elapsed_total"] / elapsed_runs) if elapsed_runs else 0.0
                ),
                "label_counts": dict(label_counts),
                "tokens_total": (
                    bucket["tokens_total"] if tokens_runs else None
                ),
                "tokens_runs": tokens_runs,
                "cost_total": (
                    bucket["cost_total"] if cost_runs else None
                ),
                "cost_runs": cost_runs,
                "invalid_label_runs": bucket["invalid_label_runs"],
                "invalid_label_rate": (
                    bucket["invalid_label_runs"] / successes if successes else 0.0
                ),
                "error_kind_counts": dict(bucket["error_kind_counts"]),
                "envelope_field_present": dict(bucket["envelope_field_present"]),
                "timeout_by_prompt_size": dict(bucket["timeout_by_prompt_size"]),
                "timeout_recoveries": bucket["timeout_recoveries"],
                "terse_retry_attempts": bucket["terse_retry_attempts"],
                "timeout_recoveries_by_prompt_size": dict(
                    bucket["timeout_recoveries_by_prompt_size"]
                ),
                "evidence_tag_distribution": dict(bucket["evidence_tag_distribution"]),
                "last_used": bucket["last_used"],
            }
        )

    return {
        "transcripts_considered": transcripts_considered,
        "total_runs": total_runs,
        "total_successes": total_successes,
        "mode_counts": dict(sorted(mode_counts.items())),
        "participants": participant_rows,
        "filters": {
            "participant": participant,
            "since_seconds": since_seconds,
        },
    }


def compute_stats(
    base_dir: Path,
    *,
    participant: str | None = None,
    since_days: int | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Convenience: load transcripts from `base_dir` and aggregate."""
    records = load_transcript_files(base_dir)
    since_seconds = since_days * 86400 if since_days else None
    return aggregate(
        records,
        participant=participant,
        since_seconds=since_seconds,
        now=now,
    )


def _fmt_seconds(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _fmt_tokens(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value}"


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "$0"
    if value < 0.001:
        return f"${value:.6f}"
    return f"${value:.4f}"


def _fmt_last_used(epoch: float | None) -> str:
    if not epoch:
        return "—"
    from datetime import datetime

    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")


def format_stats_text(stats: dict[str, Any]) -> str:
    lines: list[str] = []
    filters = stats.get("filters") or {}
    header = (
        f"transcripts: {stats['transcripts_considered']}  "
        f"runs: {stats['total_runs']}  "
        f"successes: {stats['total_successes']}"
    )
    if filters.get("since_seconds"):
        days = int(filters["since_seconds"] // 86400)
        header += f"  since: last {days}d"
    if filters.get("participant"):
        header += f"  participant: {filters['participant']}"
    lines.append(header)

    mode_counts = stats.get("mode_counts") or {}
    if mode_counts:
        lines.append(
            "modes: "
            + ", ".join(f"{name}={count}" for name, count in mode_counts.items())
        )

    rows = stats.get("participants") or []
    if not rows:
        lines.append("(no participants in selection)")
        return "\n".join(lines)

    lines.append("")
    lines.append(
        f"{'participant':14} {'runs':>5} {'ok%':>5} {'avg':>7} "
        f"{'y':>3} {'n':>3} {'t':>3} {'?':>3} "
        f"{'inv%':>5} {'tokens':>10} {'cost':>10} {'last_used':>16}"
    )
    for row in rows:
        counts = row["label_counts"]
        lines.append(
            f"{row['name'][:14]:14} "
            f"{row['runs']:>5} "
            f"{_fmt_pct(row['success_rate']):>5} "
            f"{_fmt_seconds(row['avg_elapsed_seconds']):>7} "
            f"{counts['yes']:>3} {counts['no']:>3} "
            f"{counts['tradeoff']:>3} {counts['unknown']:>3} "
            f"{_fmt_pct(row['invalid_label_rate']):>5} "
            f"{_fmt_tokens(row['tokens_total']):>10} "
            f"{_fmt_cost(row['cost_total']):>10} "
            f"{_fmt_last_used(row['last_used']):>16}"
        )
    return "\n".join(lines)
