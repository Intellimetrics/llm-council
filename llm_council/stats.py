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
        # v0.11.6 Phase 2: count of successful runs that recovered from a
        # quota_exhausted error via the `fallback_chain` retry path.
        # Distinct from `timeout_recoveries` (terse-retry) and from the
        # error_kind_counts.quota_exhausted bucket (final-state failures).
        # Together: error_kind_counts.quota_exhausted + quota_recoveries =
        # total quota incidents this peer absorbed in the window.
        "quota_recoveries": 0,
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
                if result.get("recovered_after_quota"):
                    bucket["quota_recoveries"] += 1
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
                "quota_recoveries": bucket["quota_recoveries"],
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


# ---------- Eval scorecard aggregation -----------------------------------
#
# Reads JSON files written by `llm_council.eval.runner.to_json`. The
# trend deltas compare the newest scorecard in the window to the oldest;
# a sharp drop in `blocker_recall` between deploys is the regression
# signal we care about. Bucketed by `mode` so per-mode regressions
# (e.g. review-with-tools SNR collapse) surface separately.

_EVAL_AGGREGATE_KEYS = (
    "blocker_recall",
    "false_blocker_rate",
    "signal_to_noise_ratio",
    "evidence_density",
    "citation_accuracy",
)


def _load_scorecards(scorecards_dir: Path) -> list[dict[str, Any]]:
    """Return JSON scorecards from `scorecards_dir`, newest first."""
    if not scorecards_dir.is_dir():
        return []
    entries: list[tuple[float, dict[str, Any]]] = []
    for path in scorecards_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        mtime = path.stat().st_mtime
        entries.append((mtime, data))
    # newest first
    entries.sort(key=lambda item: item[0], reverse=True)
    return [entry[1] for entry in entries]


def _peer_block(scorecard: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract per-peer aggregate metrics across all fixtures in a scorecard.

    Returns `{peer_name: {metric: value, ...}}` where each metric is the
    mean across the fixtures the peer appeared in. Citation accuracy is
    averaged only over fixtures where it was non-None.
    """
    peer_acc: dict[str, dict[str, list[float]]] = {}
    citation_acc: dict[str, list[float]] = {}
    for fixture in scorecard.get("fixtures") or []:
        for peer in fixture.get("peers") or []:
            name = peer.get("name")
            if not name:
                continue
            metrics = peer.get("metrics") or {}
            bucket = peer_acc.setdefault(
                name, {key: [] for key in _EVAL_AGGREGATE_KEYS if key != "citation_accuracy"}
            )
            for key in bucket:
                value = metrics.get(key)
                if value is not None:
                    bucket[key].append(float(value))
            ca = metrics.get("citation_accuracy")
            if ca is not None:
                citation_acc.setdefault(name, []).append(float(ca))
    out: dict[str, dict[str, Any]] = {}
    for name, bucket in peer_acc.items():
        row: dict[str, Any] = {}
        for key, values in bucket.items():
            row[key] = sum(values) / len(values) if values else None
        ca_vals = citation_acc.get(name) or []
        row["citation_accuracy"] = (
            sum(ca_vals) / len(ca_vals) if ca_vals else None
        )
        out[name] = row
    return out


def aggregate_eval_runs(
    scorecards_dir: Path, last_n: int = 10
) -> dict[str, Any]:
    """Aggregate the last N eval scorecards under `scorecards_dir`.

    Returns:
        {
          "scorecards_considered": N,
          "by_mode": {mode: {
              "count": int,
              "latest": {<aggregate_metrics>},
              "oldest": {<aggregate_metrics>},
              "delta": {<metric>: newest - oldest},
              "by_peer_latest": {peer: {<metrics>}}
          }},
        }

    The `delta` block is the regression-detection surface: a negative
    `blocker_recall` delta or a positive `false_blocker_rate` delta
    between deploys is the signal to investigate.
    """
    scorecards_dir = Path(scorecards_dir)
    scorecards = _load_scorecards(scorecards_dir)
    if last_n > 0:
        scorecards = scorecards[:last_n]

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for sc in scorecards:
        mode = str(sc.get("mode") or "unknown")
        by_mode.setdefault(mode, []).append(sc)

    out: dict[str, Any] = {
        "scorecards_considered": len(scorecards),
        "scorecards_dir": str(scorecards_dir),
        "by_mode": {},
    }
    for mode, mode_scorecards in by_mode.items():
        latest = mode_scorecards[0]
        oldest = mode_scorecards[-1]
        latest_agg = latest.get("aggregate_metrics") or {}
        oldest_agg = oldest.get("aggregate_metrics") or {}
        delta: dict[str, Any] = {}
        for key in _EVAL_AGGREGATE_KEYS:
            new_val = latest_agg.get(key)
            old_val = oldest_agg.get(key)
            if new_val is None or old_val is None:
                delta[key] = None
            else:
                delta[key] = float(new_val) - float(old_val)
        out["by_mode"][mode] = {
            "count": len(mode_scorecards),
            "latest": dict(latest_agg),
            "oldest": dict(oldest_agg),
            "delta": delta,
            "by_peer_latest": _peer_block(latest),
        }
    return out


# ---------- Per-peer reliability counters (Phase C — v0.8 plan) ---------
#
# Surfaces simple counts derived from operator-marked outcomes
# (`.llm-council/outcomes/<run-id>.json`) cross-referenced against the
# transcripts under `.llm-council/runs/`. No IRT-style scoring; the plan
# explicitly defers weighted reliability math until >=200 outcomes exist.
#
# Counters per peer:
#   - outcomes_marked: outcomes where this peer participated in the run
#   - useful_count:    outcomes where decision==shipped AND bug_found==False
#   - false_blocker_count: peer voted RECOMMENDATION:no but the change
#                          shipped and no bug was found (peer wanted to
#                          block, ship was fine)
#   - unique_blocker_catch_count: winning_peer==this peer AND
#                                 bug_found==True (operator credits this
#                                 peer with catching a real bug)
#   - verified_citation_rate: mechanical signal across ALL transcripts
#                             the peer participated in (not just
#                             outcome-marked ones); fraction of
#                             tag=="verified" entries where verified==True.
#                             Returns None when the peer has no VERIFIED
#                             evidence entries to evaluate.


def _empty_reliability_bucket() -> dict[str, Any]:
    return {
        "outcomes_marked": 0,
        "useful_count": 0,
        "false_blocker_count": 0,
        "unique_blocker_catch_count": 0,
        # Internal accumulators for the verified-citation rate; the
        # public payload divides them when rendering.
        "_verified_total": 0,
        "_verified_ok": 0,
        # v0.9.0 Feature 2: accumulators for the per-peer mean rank
        # position from `--cross-rank` runs. Each contribution comes
        # from a transcript's `cross_rank_scores[peer]` (already a
        # per-run mean position). The aggregate is the mean across
        # those per-run means. Stays at zero scoring when the peer
        # never participated in a `--cross-rank` run — surfaced as
        # `None` in the public row to mirror `verified_citation_rate`'s
        # "no data" semantic.
        "_rank_position_sum": 0.0,
        "_rank_position_count": 0,
        # v0.11.7 Phase 3: mechanical quota counters across all
        # transcripts (not outcome-dependent). `quota_incidents` =
        # every result that hit a quota wall (failed-final
        # quota_exhausted OR recovered-after-quota). `quota_recoveries`
        # = subset that recovered via `fallback_chain`. The derived
        # `quota_recovery_rate` (recoveries / incidents) goes in the
        # public payload; both raw counters stay so the operator can
        # eyeball "1/2 vs 50/100".
        "quota_incidents": 0,
        "quota_recoveries": 0,
    }


def _final_round_record_for_peer(
    results: list[dict[str, Any]], peer: str
) -> dict[str, Any] | None:
    """Return the peer's final-round record from a transcript, or None."""
    final = _final_round_only(results)
    for record in final:
        raw = record.get("name") or ""
        name = raw.split(":round")[0]
        if name == peer:
            return record
    return None


def aggregate_reliability(
    cwd: Path,
    *,
    transcripts_dir: Path | None = None,
    peer: str | None = None,
) -> dict[str, Any]:
    """Compute the per-peer reliability counters from outcomes + transcripts.

    `cwd` anchors `.llm-council/outcomes/`. `transcripts_dir` defaults to
    `<cwd>/.llm-council/runs/`; pass an explicit path when the project
    config overrode `transcripts_dir`. `peer` filters the output to one
    peer (the aggregate `total_outcomes` count still reflects all
    outcomes on disk).

    Empty outcomes / empty transcripts are both safe — the function
    returns a stable shape with zero counters rather than raising, so
    `llm-council stats --reliability` works on a fresh checkout.
    """
    from llm_council.outcomes import iter_outcomes  # local import to avoid cycle

    transcripts_dir = (
        transcripts_dir
        if transcripts_dir is not None
        else cwd / ".llm-council" / "runs"
    )

    transcripts = load_transcript_files(transcripts_dir)
    # Index transcripts by run_id (filename stem) for O(1) lookup from
    # the outcomes side.
    by_run_id: dict[str, dict[str, Any]] = {}
    for entry in transcripts:
        path = Path(entry["path"])
        by_run_id[path.stem] = entry["data"]

    outcomes = list(iter_outcomes(cwd))

    peers: dict[str, dict[str, Any]] = {}

    # First pass: mechanical verified-citation rate across ALL
    # transcripts a peer participated in (not just outcome-marked ones).
    # This is the early signal Phase A produces; it works the moment a
    # peer emits any [VERIFIED:...] entry, no operator labeling needed.
    for transcript in by_run_id.values():
        for result in transcript.get("results") or []:
            raw_name = result.get("name") or ""
            # Strip both `:round\d+` (deliberation) and `:rank`
            # (v0.9.0 cross-rank pass) suffixes so the bucket merges
            # by primary peer identity.
            name = raw_name.split(":round")[0].split(":rank")[0] or "unknown"
            if not name:
                continue
            bucket = peers.setdefault(name, _empty_reliability_bucket())
            for entry in result.get("evidence") or []:
                if not isinstance(entry, dict):
                    continue
                if entry.get("tag") != "verified":
                    continue
                bucket["_verified_total"] += 1
                if entry.get("verified") is True:
                    bucket["_verified_ok"] += 1
            # v0.11.7 Phase 3 mechanical quota counters. Walk every
            # round (not just the final round) so a peer that hit
            # quota in round 1 and again in round 2 is counted twice,
            # which is the operationally useful "how often does this
            # peer hit a quota wall" signal. A recovered call counts
            # as both an incident AND a recovery — the call DID hit
            # quota; the fallback rescued it.
            if result.get("recovered_after_quota"):
                bucket["quota_incidents"] += 1
                bucket["quota_recoveries"] += 1
            elif result.get("error_kind") == "quota_exhausted":
                bucket["quota_incidents"] += 1
        # v0.9.0 Feature 2: accumulate per-peer rank-position mean from
        # transcripts that ran `--cross-rank`. Each per-run mean is one
        # observation; the aggregate is the mean across observations.
        cross_rank_scores = transcript.get("cross_rank_scores") or {}
        if isinstance(cross_rank_scores, dict):
            for peer_name, score in cross_rank_scores.items():
                if not isinstance(peer_name, str) or not peer_name:
                    continue
                try:
                    score_f = float(score)
                except (TypeError, ValueError):
                    continue
                bucket = peers.setdefault(peer_name, _empty_reliability_bucket())
                bucket["_rank_position_sum"] += score_f
                bucket["_rank_position_count"] += 1

    # Second pass: operator-marked outcome counters. Cross-reference the
    # outcome's run_id to the transcript to find which peers
    # participated (and how they voted).
    for outcome in outcomes:
        transcript = by_run_id.get(outcome.run_id)
        if transcript is None:
            continue
        participants = transcript.get("participants") or []
        results = transcript.get("results") or []
        decision = outcome.decision
        bug_found = outcome.bug_found
        winning_peer = outcome.winning_peer

        for participant in participants:
            if not isinstance(participant, str) or not participant:
                continue
            bucket = peers.setdefault(participant, _empty_reliability_bucket())
            bucket["outcomes_marked"] += 1

            shipped_clean = decision == "shipped" and bug_found is False

            # `useful_count` and `false_blocker_count` are mutually
            # exclusive and depend on the peer's actual vote on this run:
            #   - vote `yes` / `tradeoff` on shipped+no-bug → useful
            #   - vote `no` on shipped+no-bug → false_blocker
            #   - no usable label (abdicated / ok=False) → neither
            if shipped_clean:
                record = _final_round_record_for_peer(results, participant)
                if record is not None and record.get("ok"):
                    label = recommendation_label(record.get("output") or "")
                    if label == "no":
                        bucket["false_blocker_count"] += 1
                    elif label in ("yes", "tradeoff"):
                        bucket["useful_count"] += 1

            if (
                winning_peer
                and participant == winning_peer
                and bug_found is True
            ):
                bucket["unique_blocker_catch_count"] += 1

    rows: list[dict[str, Any]] = []
    for name, bucket in sorted(peers.items()):
        if peer and name != peer:
            continue
        verified_total = bucket["_verified_total"]
        verified_ok = bucket["_verified_ok"]
        verified_rate: float | None
        if verified_total > 0:
            verified_rate = verified_ok / verified_total
        else:
            verified_rate = None
        rank_count = bucket["_rank_position_count"]
        rank_position_mean: float | None
        if rank_count > 0:
            rank_position_mean = bucket["_rank_position_sum"] / rank_count
        else:
            rank_position_mean = None
        quota_incidents = bucket["quota_incidents"]
        quota_recoveries = bucket["quota_recoveries"]
        quota_recovery_rate: float | None
        if quota_incidents > 0:
            quota_recovery_rate = quota_recoveries / quota_incidents
        else:
            quota_recovery_rate = None
        # A peer that has neither outcomes nor VERIFIED evidence nor
        # rank-position data nor quota incidents contributes no
        # signal — drop it from the rendered output to avoid swamping
        # the table with zero rows. Callers that want "this peer exists
        # but has nothing yet" can use `llm-council stats --participant
        # <peer>` (separate code path).
        if (
            bucket["outcomes_marked"] == 0
            and verified_total == 0
            and rank_count == 0
            and quota_incidents == 0
        ):
            continue
        rows.append(
            {
                "name": name,
                "outcomes_marked": bucket["outcomes_marked"],
                "useful_count": bucket["useful_count"],
                "false_blocker_count": bucket["false_blocker_count"],
                "unique_blocker_catch_count": bucket["unique_blocker_catch_count"],
                "verified_citation_rate": verified_rate,
                "verified_total": verified_total,
                "rank_position_mean": rank_position_mean,
                "rank_position_count": rank_count,
                "quota_incidents": quota_incidents,
                "quota_recoveries": quota_recoveries,
                "quota_recovery_rate": quota_recovery_rate,
            }
        )

    return {
        "total_outcomes": len(outcomes),
        "transcripts_considered": len(transcripts),
        "filters": {"peer": peer},
        "peers": rows,
    }


def format_reliability_text(reliability: dict[str, Any]) -> str:
    """Render `aggregate_reliability` output as a fixed-width text table.

    Mirrors the look-and-feel of `format_stats_text`. Designed to be
    readable on an 80-col terminal even with five counters per row.
    """
    lines: list[str] = []
    filters = reliability.get("filters") or {}
    header = (
        f"outcomes: {reliability['total_outcomes']}  "
        f"transcripts: {reliability['transcripts_considered']}"
    )
    if filters.get("peer"):
        header += f"  peer: {filters['peer']}"
    lines.append(header)
    rows = reliability.get("peers") or []
    if not rows:
        lines.append("(no peer reliability signal yet)")
        return "\n".join(lines)
    lines.append("")
    lines.append(
        f"{'participant':14} {'marked':>7} {'useful':>7} "
        f"{'falseB':>7} {'uniqCatch':>10} {'verifCite':>10} "
        f"{'rankPos':>8} {'quotaInc':>9} {'quotaRec':>9}"
    )
    for row in rows:
        rate = row.get("verified_citation_rate")
        rate_str = "—" if rate is None else f"{rate * 100:.0f}%"
        rank_mean = row.get("rank_position_mean")
        rank_str = "—" if rank_mean is None else f"{rank_mean:.2f}"
        # Quota columns: incidents is a raw count; recoveries is rendered
        # as "<n>/<incidents>" so the operator can eyeball the rate
        # without scanning a separate column. Dash both when no incidents
        # (mirrors the verified/rank "no signal" rendering above).
        quota_inc = row.get("quota_incidents", 0)
        quota_rec = row.get("quota_recoveries", 0)
        if quota_inc == 0:
            quota_inc_str = "—"
            quota_rec_str = "—"
        else:
            quota_inc_str = str(quota_inc)
            rate = row.get("quota_recovery_rate")
            pct = "" if rate is None else f" ({rate * 100:.0f}%)"
            quota_rec_str = f"{quota_rec}/{quota_inc}{pct}"
        lines.append(
            f"{row['name'][:14]:14} "
            f"{row['outcomes_marked']:>7} "
            f"{row['useful_count']:>7} "
            f"{row['false_blocker_count']:>7} "
            f"{row['unique_blocker_catch_count']:>10} "
            f"{rate_str:>10} "
            f"{rank_str:>8} "
            f"{quota_inc_str:>9} "
            f"{quota_rec_str:>9}"
        )
    return "\n".join(lines)


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
