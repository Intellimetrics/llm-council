"""Aggregate statistics over council transcripts.

Pure read: scans `.llm-council/runs/*.json` and computes per-participant and
aggregate metrics. Backs the `llm-council stats` CLI subcommand and the
`council_stats` MCP tool.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from llm_council import display
from llm_council.deliberation import recommendation_label
from llm_council.transcript import iter_run_json, result_round


_LABELS = ("yes", "no", "tradeoff", "unknown")


def _base_peer_name(name, *, fallback="unknown"):
    base = (name or "").split(":round")[0].split(":rank")[0]
    return (base or fallback) if fallback is not None else base
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
    return [
        {"path": str(p), "mtime": m, "data": d} for p, m, d in iter_run_json(base_dir)
    ]


def _final_round_only(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []
    # Historical transcripts from the removed `--cross-rank` feature (pre
    # v0.19.0) carry ranking-round results (`peer:rank`, is_ranking_round=
    # True) alongside the primary votes. Those are post-deliberation
    # telemetry, not primary votes — exclude them from the final-round view
    # so old transcripts on disk don't inflate total_runs and per-peer
    # counts. Tolerant `.get()` read: never crashes on transcripts that
    # predate (or postdate) the field.
    primary = [r for r in results if not r.get("is_ranking_round")]
    if not primary:
        return []
    final = max(result_round(r.get("name", "")) for r in primary)
    return [r for r in primary if result_round(r.get("name", "")) == final]


def _empty_label_counts() -> dict[str, int]:
    return {label: 0 for label in _LABELS}


def _new_peer_bucket() -> dict[str, Any]:
    return {
        "runs": 0,
        "cache_hits": 0,
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
        # v0.11.6 Phase 2: count of successful runs that recovered from a
        # quota_exhausted error via the `fallback_chain` retry path.
        # Distinct from `timeout_recoveries` (terse-retry). See
        # `quota_incidents` below for the incident-side counterpart.
        "quota_recoveries": 0,
        # v0.19.0: mechanical count of every quota wall this peer hit in
        # the final round, whether or not the fallback chain rescued it
        # (`quota_incidents` = final-round `recovered_after_quota` results
        # PLUS final-round `error_kind == "quota_exhausted"` failures).
        # `quota_recoveries` above is the subset that recovered. Relocated
        # here from the removed `stats.aggregate_reliability` (which
        # walked every round, not just the final one) — this is the only
        # observable usage signal for text-mode CLI peers, so it stays
        # even though outcome-derived reliability did not earn its keep.
        "quota_incidents": 0,
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
            "verified": 0,
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
    # Run-level OKF enrichment outcomes (v0.24.0 `okf_context` metadata).
    # Only present on transcripts where the feature was enabled; the
    # attach rate vs. binary_missing / no_matched_concepts split is the
    # signal for whether the blast-radius excerpt is earning its keep.
    okf_status_counts: dict[str, int] = {}
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
        okf_record = (data.get("metadata") or {}).get("okf_context")
        if isinstance(okf_record, dict) and okf_record.get("status"):
            okf_status = str(okf_record["status"])
            okf_status_counts[okf_status] = okf_status_counts.get(okf_status, 0) + 1

        for result in results:
            raw_name = result.get("name") or ""
            # Strip both `:round\d+` (deliberation) and `:rank` (historical
            # `--cross-rank` transcripts, pre v0.19.0) so a peer's cost/
            # tokens/latency fold into its own row instead of a phantom
            # `<peer>:rank` participant.
            name = _base_peer_name(raw_name)
            bucket = peers.setdefault(name, _new_peer_bucket())
            if bucket["last_used"] is None or mtime > bucket["last_used"]:
                bucket["last_used"] = mtime
            # Cached results retain the original request's usage receipts.
            # Reading them again did not incur another request or charge.
            if result.get("from_cache"):
                bucket["cache_hits"] += 1
                continue
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

        seen_in_transcript: set[str] = set()
        for result in final_results:
            raw_name = result.get("name") or ""
            name = _base_peer_name(raw_name)
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
            if result.get("terse_retry_attempted") and not result.get("from_cache"):
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
                if result.get("recovered_after_timeout") and not result.get("from_cache"):
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
                if result.get("recovered_after_quota") and not result.get("from_cache"):
                    # The call DID hit quota; the fallback rescued it — so
                    # it counts as both an incident AND a recovery.
                    bucket["quota_incidents"] += 1
                    bucket["quota_recoveries"] += 1
                # Evidence-tag distribution: count each EVIDENCE bullet
                # by its tag. Entries shape from v0.7 onward is
                # list[{text, tag}]; legacy list[str] entries (pre-v0.7
                # cached transcripts) count as untagged.
                for entry in result.get("evidence") or []:
                    if isinstance(entry, dict):
                        tag = entry.get("tag")
                        bucket_key = tag if tag in (
                            "verified", "published", "observable", "inferred", "speculative",
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
                elif error_kind == "quota_exhausted":
                    bucket["quota_incidents"] += 1

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
                "cache_hits": bucket["cache_hits"],
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
                "quota_recoveries": bucket["quota_recoveries"],
                "quota_incidents": bucket["quota_incidents"],
                "quota_recovery_rate": (
                    (bucket["quota_recoveries"] / bucket["quota_incidents"])
                    if bucket["quota_incidents"]
                    else None
                ),
                "evidence_tag_distribution": dict(bucket["evidence_tag_distribution"]),
                "last_used": bucket["last_used"],
            }
        )

    stats = {
        "transcripts_considered": transcripts_considered,
        "total_runs": total_runs,
        "total_successes": total_successes,
        "mode_counts": dict(sorted(mode_counts.items())),
        "okf_context_status_counts": dict(sorted(okf_status_counts.items())),
        "participants": participant_rows,
        "filters": {
            "participant": participant,
            "since_seconds": since_seconds,
        },
    }
    stats["recommendations"] = derive_recommendations(stats)
    return stats


# Minimum sample sizes before a recommendation fires. Deliberately
# conservative: a recommendation printed on two runs of noise erodes trust
# in the whole block. Tune upward, not downward.
_RECO_MIN_TIMEOUTS = 3
_RECO_MIN_QUOTA_INCIDENTS = 2
_RECO_MIN_LABEL_RUNS = 5
_RECO_INVALID_LABEL_RATE = 0.2


def derive_recommendations(stats: dict[str, Any]) -> list[str]:
    """Turn aggregated telemetry into actionable configuration advice.

    These rules previously lived only as source comments on the metric
    definitions (`timeout_by_prompt_size`, `terse_retry_attempts`,
    `quota_incidents`, ...); an operator reading `llm-council stats`
    output never saw them. Each rule states the observation AND the
    concrete knob to turn. Advisory only — nothing here changes behavior.
    """

    recommendations: list[str] = []
    for row in stats.get("participants") or []:
        name = row.get("name") or "?"
        timeouts_by_size = row.get("timeout_by_prompt_size") or {}
        recoveries_by_size = row.get("timeout_recoveries_by_prompt_size") or {}
        timeout_rule_fired = False
        for size_bucket, timeouts in timeouts_by_size.items():
            recoveries = recoveries_by_size.get(size_bucket, 0)
            if timeouts >= _RECO_MIN_TIMEOUTS and recoveries == 0:
                timeout_rule_fired = True
                recommendations.append(
                    f"{name}: {timeouts} timeouts on {size_bucket} prompts "
                    "with 0 terse-retry recoveries — raise "
                    f"`participants.{name}.timeout` (or the mode's "
                    "`timeout_multiplier`)."
                )
        terse_attempts = row.get("terse_retry_attempts") or 0
        timeout_recoveries = row.get("timeout_recoveries") or 0
        # Skipped when a bucket rule above already fired for this peer:
        # both rules read the same underlying failures and printing
        # near-duplicate advice for one peer reads as noise.
        if (
            not timeout_rule_fired
            and terse_attempts >= _RECO_MIN_TIMEOUTS
            and timeout_recoveries == 0
        ):
            recommendations.append(
                f"{name}: terse-retry fired {terse_attempts}x and never "
                "recovered — the retry window is structurally too small "
                "for this peer; raise the base `timeout` or set "
                f"`participants.{name}.terse_retry_on_timeout: false` to "
                "stop paying for doomed retries."
            )
        quota_incidents = row.get("quota_incidents") or 0
        quota_recoveries = row.get("quota_recoveries") or 0
        if quota_incidents >= _RECO_MIN_QUOTA_INCIDENTS and quota_recoveries == 0:
            recommendations.append(
                f"{name}: hit {quota_incidents} quota walls with no "
                "fallback recovery — configure "
                f"`participants.{name}.fallback_chain` (ordered step-down "
                "model ids) or move the peer to a lighter model/tier."
            )
        # Gate on SUCCESSES, not runs: invalid_label_rate's denominator is
        # successful responses, so gating on runs let a peer with 5 runs
        # and 1 unlabeled success print "100%" from a sample of one
        # (2026-09-01 council finding).
        successes = row.get("successes") or 0
        invalid_rate = row.get("invalid_label_rate") or 0.0
        if (
            successes >= _RECO_MIN_LABEL_RUNS
            and invalid_rate >= _RECO_INVALID_LABEL_RATE
        ):
            recommendations.append(
                f"{name}: {invalid_rate * 100:.0f}% of successful responses "
                "lacked a usable RECOMMENDATION label — check custom prompt "
                "phrasing; for genuinely non-vote uses set "
                f"`participants.{name}.require_recommendation: false` "
                "instead of eating the repair-retry cost (already set? "
                "then this is expected — ignore)."
            )
        # Deliberate n=1 exception to the minimum-sample discipline: each
        # refusal is individually actionable (the fix is rephrasing that
        # specific prompt), unlike the rate-based rules above.
        content_refusals = (row.get("error_kind_counts") or {}).get(
            "content_refused", 0
        )
        if content_refusals:
            recommendations.append(
                f"{name}: {content_refusals} content-policy refusal(s) — "
                "rephrase security prompts as verification (\"verify this "
                "is safe\") rather than attack (\"find a bypass\"); see the "
                "⚠️ notes in the affected transcripts."
            )
    okf_counts = stats.get("okf_context_status_counts") or {}
    okf_missing = okf_counts.get("binary_missing", 0)
    if okf_missing:
        recommendations.append(
            f"{okf_missing} run(s) requested OKF blast-radius context but "
            "the configured OKF binary (default `okf-rs`) was not on PATH "
            "— install okf-rs (GitHub release binary or `cargo install "
            "--git https://github.com/jyjeanne/okf-rs okf-cli`) to give "
            "peers call-graph context on diff reviews."
        )
    okf_attached = okf_counts.get("attached", 0) + okf_counts.get(
        "stale_attached", 0
    )
    okf_unmatched = okf_counts.get("no_matched_concepts", 0)
    if okf_unmatched >= _RECO_MIN_TIMEOUTS and okf_unmatched > okf_attached:
        recommendations.append(
            f"OKF enrichment matched no concepts in {okf_unmatched} "
            "attempt(s) (vs "
            f"{okf_attached} attached) — the diffs may touch non-code "
            "files, or the languages involved are outside okf-rs's "
            "extraction coverage."
        )
    return recommendations


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
    return display.format_usd(value)


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
        # Still fall through: the okf-context and recommendations blocks
        # below can carry run-level signal (e.g. binary_missing advice)
        # even when no participant rows matched the filter.
        lines.append("(no participants in selection)")
        return _append_advice_blocks(lines, stats)

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

    # Quota telemetry: the only observable usage signal for text-mode CLI
    # peers (no per-request token metering hook exists). Rendered as a
    # secondary block, not extra main-table columns, and only for peers
    # that actually hit a quota wall — most peers never do, and folding
    # two more fixed-width columns into the main table above would widen
    # every row for a signal that is usually all dashes.
    quota_rows = [row for row in rows if row.get("quota_incidents")]
    if quota_rows:
        lines.append("")
        lines.append("quota (recovered/incidents):")
        for row in quota_rows:
            incidents = row["quota_incidents"]
            recoveries = row["quota_recoveries"]
            rate = row.get("quota_recovery_rate")
            pct = "" if rate is None else f" ({rate * 100:.0f}%)"
            lines.append(
                f"  {row['name'][:14]:14} {recoveries}/{incidents}{pct}"
            )

    return _append_advice_blocks(lines, stats)


def _append_advice_blocks(lines: list[str], stats: dict[str, Any]) -> str:
    """Shared tail: okf-context counts + the advisory recommendations.

    The interpretation rules previously lived only as source comments on
    the metric definitions, invisible to the operator running `stats`.
    """

    okf_counts = stats.get("okf_context_status_counts") or {}
    if okf_counts:
        lines.append("")
        lines.append(
            "okf-context: "
            + ", ".join(f"{name}={count}" for name, count in okf_counts.items())
        )
    recommendations = stats.get("recommendations") or []
    if recommendations:
        lines.append("")
        lines.append("recommendations:")
        for recommendation in recommendations:
            lines.append(f"  - {recommendation}")
    return "\n".join(lines)
