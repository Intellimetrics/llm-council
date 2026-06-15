"""Transcript writing."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_council.adapters import (
    ParticipantResult,
    command_for_display,
    is_context_overflow_error,
    is_timeout_error,
)
from llm_council.convergence import tally_states
from llm_council.deliberation import (
    default_min_quorum,
    labeled_quorum_count,
    model_comparison,
    recommendation_counts,
    recommendation_label,
    recommendation_line,
)

ROUND_SUFFIX_RE = re.compile(r":round(\d+)$")


def safe_slug(text: str, max_len: int = 60) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return (cleaned or "council")[:max_len].strip("-")


def transcript_paths(base_dir: Path, question: str) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{stamp}_{safe_slug(question)}"
    return base_dir / f"{stem}.md", base_dir / f"{stem}.json"


def transcript_dir(cwd: Path, config: dict) -> Path:
    """Resolve the transcripts directory from config, anchored at ``cwd``.

    Single source of truth for the previously-inlined copies across
    ``mcp_server.py`` and ``cli.py`` (the
    ``Path(config.get("transcripts_dir", ".llm-council/runs"))`` +
    relative-to-cwd resolution pattern).
    """

    out_dir = Path(config.get("transcripts_dir", ".llm-council/runs"))
    return out_dir if out_dir.is_absolute() else cwd / out_dir


_RUN_ID_RE = re.compile(r"^\d{8}_\d{6}")


def normalize_run_id(value: str) -> str:
    """Strip directory and known suffixes; require the timestamp prefix."""

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("run id is empty")
    raw = Path(raw).name
    for suffix in (".json", ".md"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            break
    match = _RUN_ID_RE.match(raw)
    if not match:
        raise ValueError(
            f"run id '{value}' does not start with a YYYYMMDD_HHMMSS prefix"
        )
    return raw


def find_transcript_by_id(base_dir: Path, run_id: str) -> dict[str, Any]:
    """Locate a JSON transcript by run-id prefix or filename and load it.

    Accepts either the bare timestamp prefix (``20260502_062608``) or a
    full filename (``20260502_062608_question.json`` / ``.md``). Raises
    ``FileNotFoundError`` if no JSON transcript matches.
    """

    normalized = normalize_run_id(run_id)
    candidates = sorted(base_dir.glob(f"{normalized}*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No council transcript matching run id '{run_id}' was found in "
            f"{base_dir}. Run `llm-council transcripts list` to see available ids."
        )
    if len(candidates) > 1:
        for candidate in candidates:
            if candidate.stem == normalized:
                return _load_transcript_json(candidate)
        names = ", ".join(c.name for c in candidates)
        raise ValueError(
            f"Run id '{run_id}' matches multiple transcripts ({names}); "
            "supply the full filename or a longer prefix."
        )
    return _load_transcript_json(candidates[0])


DEFAULT_MAX_CONTINUATION_DEPTH = 5


def count_continuation_depth(base_dir: Path, run_id: str, *, max_depth: int = 32) -> int:
    """Walk parent_run_id chain backwards and return the depth.

    Depth 1 means "this run has one parent" (i.e., it would be the second
    link in the chain when resumed). The traversal is bounded by
    ``max_depth`` so a corrupt cycle can't hang the caller.

    Callers that want to enforce a configured cap should pass
    ``max_depth=cap + 1`` so the walker can return a value strictly
    greater than the cap when the chain exceeds it. A cycle in the
    transcripts is always treated as corruption and surfaced via
    ``ValueError`` rather than silently truncating, since under-counting
    in that case would mistakenly approve a chain that should be
    rejected.
    """

    visited: set[str] = set()
    current = run_id
    depth = 0
    while current and depth < max_depth:
        normalized = normalize_run_id(current)
        if normalized in visited:
            raise ValueError(
                f"Continuation chain contains a cycle: '{normalized}' "
                "appears more than once. Inspect the affected transcript "
                "JSON files' parent_run_id fields and remove the loop."
            )
        visited.add(normalized)
        try:
            transcript = find_transcript_by_id(base_dir, normalized)
        except (FileNotFoundError, ValueError):
            break
        parent = transcript.get("parent_run_id")
        if not parent:
            break
        depth += 1
        current = str(parent)
    return depth


def continuation_depth_limit_error(
    config: dict[str, Any], transcripts_dir: Path, run_id: str
) -> str | None:
    """Return an error message if continuing ``run_id`` would exceed the
    configured ``defaults.max_continuation_depth``, else None.

    Shared by the CLI (`cmd_run_async`) and MCP (`run_council`) run pipelines so
    the cap computation and the message can't drift between them (they had
    already drifted slightly before this was extracted). Each caller raises its
    own exception type (SystemExit / ValueError) with the returned message.
    Passes ``max_depth + 1`` to the walker so it can count strictly past the cap
    even when the user-configured cap exceeds the walker's internal ceiling.
    """
    max_depth = int(
        config.get("defaults", {}).get(
            "max_continuation_depth", DEFAULT_MAX_CONTINUATION_DEPTH
        )
    )
    depth = count_continuation_depth(transcripts_dir, run_id, max_depth=max_depth + 1)
    if depth >= max_depth:
        return (
            f"Continuation chain depth ({depth} parents) reaches the configured "
            f"limit of {max_depth}. Each link summarizes its predecessor, so deep "
            "chains eat into MAX_PROMPT_CHARS without adding new signal. Start a "
            "fresh run, or raise `defaults.max_continuation_depth` in "
            "`.llm-council.yaml`."
        )
    return None


def _load_transcript_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to read transcript JSON at {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"Transcript JSON at {path} is not an object")
    data.setdefault("_path", str(path))
    return data


_PRIOR_QUESTION_MAX_CHARS = 2000
_PRIOR_PEER_SUMMARY_MAX_CHARS = 240


def _final_round_label(name: str) -> str:
    return ROUND_SUFFIX_RE.sub("", name)


def _strip_recommendation_from_summary(summary: str) -> str:
    return _strip_recommendation_prefix(summary or "").strip()


def _select_final_round_records(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []
    rounds = [result_round(str(r.get("name") or "")) for r in results]
    final_round = max(rounds)
    return [
        result
        for result, round_no in zip(results, rounds)
        if round_no == final_round
    ]


def _summarize_record_label(record: dict[str, Any]) -> tuple[str, str, bool]:
    output = str(record.get("output") or "")
    if record.get("ok") and output:
        label = recommendation_label(output)
        line = recommendation_line(output)
        summary = _strip_recommendation_from_summary(line)
        return label, _cap_peer_summary(summary), False
    error = str(record.get("error") or "")
    summary = _first_nonempty_line(error)
    return "unknown", _cap_peer_summary(summary), True


def _cap_peer_summary(text: str) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= _PRIOR_PEER_SUMMARY_MAX_CHARS:
        return cleaned
    return cleaned[: _PRIOR_PEER_SUMMARY_MAX_CHARS - 3].rstrip() + "..."


def format_prior_council_context(
    transcript: dict[str, Any],
    *,
    run_id: str | None = None,
) -> str:
    """Render a compact 'Prior council context' block for prompt prepending.

    The block summarizes the prior question, the final-round labels and a
    one-line rationale per peer, plus notes pulled from
    ``remaining_disagreement`` and ``degraded_consensus`` payloads when the
    prior run recorded them.
    """

    if not isinstance(transcript, dict):
        raise ValueError("transcript must be a dict loaded from JSON")
    if run_id is None:
        path = transcript.get("_path")
        if path:
            run_id = Path(str(path)).stem
        else:
            run_id = "unknown"
    question = str(transcript.get("question") or "").strip()
    truncated_question = question
    if len(truncated_question) > _PRIOR_QUESTION_MAX_CHARS:
        truncated_question = (
            truncated_question[:_PRIOR_QUESTION_MAX_CHARS].rstrip()
            + "...[truncated]"
        )

    results = transcript.get("results") or []
    if not isinstance(results, list):
        results = []
    final_records = _select_final_round_records(results)

    counts = {"yes": 0, "no": 0, "tradeoff": 0, "unknown": 0}
    peer_lines: list[str] = []
    for record in final_records:
        if not isinstance(record, dict):
            continue
        name = str(record.get("name") or "?")
        display_name = _final_round_label(name)
        label, summary, is_error = _summarize_record_label(record)
        counts[label if label in counts else "unknown"] += 1
        if is_error:
            rendered_summary = (
                f"error: {summary}" if summary else "error: no detail recorded"
            )
        else:
            rendered_summary = summary or "no rationale recorded"
        peer_lines.append(f"- {display_name}: {label} — {rendered_summary}")

    remaining = transcript.get("remaining_disagreement")
    if isinstance(remaining, dict):
        rem_participants = remaining.get("participants") or []
        for entry in rem_participants:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "?")
            display_name = _final_round_label(name)
            already = any(
                line.startswith(f"- {display_name}: ") for line in peer_lines
            )
            if already:
                continue
            label = entry.get("label") or "unknown"
            summary = (entry.get("summary") or "").strip() or "no rationale recorded"
            peer_lines.append(f"- {display_name}: {label} — {summary}")

    degraded = transcript.get("degraded_consensus")
    is_degraded = isinstance(degraded, dict)

    lines: list[str] = [f"Prior council context (run {run_id}):", ""]
    if truncated_question:
        lines.append(f"Question: {truncated_question}")
        lines.append("")
    summary_line = (
        "Recommendations (final round): "
        f"{counts['yes']} yes / {counts['no']} no / "
        f"{counts['tradeoff']} tradeoff / {counts['unknown']} unknown"
    )
    lines.append(summary_line)
    if peer_lines:
        lines.extend(peer_lines)
    else:
        lines.append("- (no participant responses recorded)")
    if is_degraded:
        labeled = degraded.get("labeled_quorum")
        threshold = degraded.get("min_quorum")
        lines.extend(
            [
                "",
                "[Note: prior run was degraded — "
                f"{labeled} of {threshold} required peers labeled.]",
            ]
        )
    if isinstance(remaining, dict) and remaining.get("ran_max_rounds_unresolved"):
        lines.extend(
            [
                "",
                "[Note: prior run reached max deliberation rounds without convergence.]",
            ]
        )
    return "\n".join(lines).rstrip()


def latest_transcript(base_dir: Path, *, suffix: str = ".md") -> Path | None:
    matches = sorted(
        _existing_paths(base_dir.glob(f"*{suffix}")), key=lambda item: item[1]
    )
    return matches[-1][0] if matches else None


def _existing_paths(paths) -> list[tuple[Path, float]]:
    existing = []
    for path in paths:
        try:
            existing.append((path, path.stat().st_mtime))
        except FileNotFoundError:
            continue
    return existing


def iter_run_json(base_dir: Path) -> list[tuple[Path, float, dict]]:
    """Mtime-sorted ``(path, mtime, data)`` for every readable run JSON.

    Shared scan over ``base_dir/*.json`` (mirrors
    ``stats.load_transcript_files``): each file is stat'd via
    ``_existing_paths``, then read with ``json.loads``; unreadable or
    malformed files are skipped. Unlike ``_load_transcript_json`` this
    never raises and does not enforce ``dict`` shape or set ``_path`` —
    those guarantees are intentionally reserved for the by-id loader.
    """

    rows: list[tuple[Path, float, dict]] = []
    for path, mtime in sorted(
        _existing_paths(base_dir.glob("*.json")), key=lambda item: item[1]
    ):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append((path, mtime, data))
    return rows


def transcript_records(base_dir: Path) -> list[dict[str, Any]]:
    # Mirrors stats.load_transcript_files's scan via the shared iter_run_json.
    records: list[dict[str, Any]] = []
    for path, mtime, data in iter_run_json(base_dir):
        results = data.get("results") or []
        records.append(
            {
                "path": str(path),
                "markdown": str(path.with_suffix(".md")),
                "question": data.get("question", ""),
                "mode": data.get("mode", ""),
                "current": data.get("current"),
                "participants": data.get("participants", []),
                "ok": sum(1 for result in results if result.get("ok")),
                "total": len(results),
                "tokens": sum(result.get("total_tokens") or 0 for result in results),
                "cost_usd": sum(result.get("cost_usd") or 0 for result in results),
                "mtime": mtime,
            }
        )
    return records


def result_to_dict(result: ParticipantResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": result.name,
        "ok": result.ok,
        "model": result.model,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "command": result.command,
        "output": result.output,
        "error": result.error,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "cost_usd": result.cost_usd,
    }
    if result.recovered_after_launch_retry:
        payload["recovered_after_launch_retry"] = True
    if result.repair_retry_recovered:
        payload["repair_retry_recovered"] = True
    if result.recovered_after_timeout:
        payload["recovered_after_timeout"] = True
    if result.terse_retry_attempted:
        payload["terse_retry_attempted"] = True
    if getattr(result, "model_fallback_used", None):
        payload["model_fallback_used"] = result.model_fallback_used
    if getattr(result, "recovered_after_quota", False):
        payload["recovered_after_quota"] = True
    if result.section_repair_attempted:
        payload["section_repair_attempted"] = True
    if getattr(result, "is_ranking_round", False):
        payload["is_ranking_round"] = True
    if getattr(result, "tool_call_status", None) is not None:
        payload["tool_call_status"] = result.tool_call_status
    if result.prompt_chars is not None:
        payload["prompt_chars"] = result.prompt_chars
    if result.from_cache:
        payload["from_cache"] = True
        if result.cache_hit_seconds is not None:
            payload["cache_hit_seconds"] = result.cache_hit_seconds
    if result.stance is not None:
        payload["stance"] = result.stance
    # Envelope fields are emitted only when present so transcripts from
    # peers that never supply them stay readable. List fields are emitted
    # when non-empty; scalar fields when not None.
    envelope_lists = {
        "blockers": list(result.blockers or ()),
        "evidence": list(result.evidence or ()),
        "tests_to_run": list(result.tests_to_run or ()),
        "assumptions": list(result.assumptions or ()),
    }
    for field_name in ("effort", "confidence", "risk"):
        value = getattr(result, field_name, None)
        if value is not None:
            payload[field_name] = value
    for field_name, items in envelope_lists.items():
        if items:
            payload[field_name] = items
    # continue_debate (round-2 vote) and evidence_verification_failures
    # (failed [VERIFIED:...] cites) are part of the documented envelope /
    # citations surface — emit them so the transcript JSON matches the
    # docstrings and the MCP structured_results shape.
    if getattr(result, "continue_debate", None) is not None:
        payload["continue_debate"] = result.continue_debate
    if getattr(result, "evidence_verification_failures", None):
        payload["evidence_verification_failures"] = list(
            result.evidence_verification_failures
        )
    from llm_council.adapters import classify_error

    error_kind = classify_error(result.error)
    if error_kind is not None:
        payload["error_kind"] = error_kind
    return payload


def convergence_summary_lines(metadata: dict[str, Any]) -> list[str]:
    """Render per-round convergence tallies as bullet lines for the markdown header.

    Returns an empty list when no convergence data is recorded (i.e. fewer than
    two rounds ran or the orchestrator did not stamp metadata).
    """
    convergence = metadata.get("convergence")
    if not isinstance(convergence, dict) or not convergence:
        return []
    lines: list[str] = []
    for round_key in sorted(convergence.keys(), key=lambda k: int(k)):
        records = convergence.get(round_key) or []
        if not isinstance(records, list) or not records:
            continue
        states = [r.get("state") for r in records if isinstance(r, dict)]
        counts = tally_states(states)
        insufficient = sum(1 for s in states if s == "insufficient")
        classified_total = counts["converged"] + counts["refining"] + counts["diverging"]
        parts = []
        for state in ("converged", "refining", "diverging"):
            if counts[state]:
                parts.append(f"{counts[state]} {state}")
        if insufficient:
            parts.append(f"{insufficient} insufficient")
        summary = ", ".join(parts) if parts else "no signal"
        prefix = ""
        if classified_total > 0 and counts["converged"] == classified_total:
            prefix = "**ALL CONVERGED** — "
        lines.append(f"- Convergence (round {round_key}): {prefix}{summary}")
    return lines


def deliberation_summary(metadata: dict[str, Any]) -> str:
    status = metadata.get("deliberation_status")
    if status == "ran_no_labeled_disagreement":
        return "ran; no labeled disagreement remained"
    if status == "ran_max_rounds_unresolved":
        return "ran; max rounds reached with labeled disagreement"
    if status == "skipped_no_labeled_disagreement":
        return "skipped, no labeled disagreement detected"
    if status == "skipped_max_rounds":
        return "skipped, max rounds is 1"
    if status == "pending":
        return "pending"
    if metadata.get("deliberated"):
        return "ran"
    if metadata.get("deliberation_requested"):
        return "skipped"
    return "not requested"


def result_round(name: str) -> int:
    match = ROUND_SUFFIX_RE.search(name)
    return int(match.group(1)) if match else 1


def final_round_results(results: list[ParticipantResult]) -> list[ParticipantResult]:
    if not results:
        return []
    # v0.9.0 Feature 2: ranking-round results (`peer:rank`) are
    # post-deliberation telemetry only; they are NOT part of the
    # peer-vote final-round view consumed by synthesis, recommendation
    # counts, and the headline label aggregation. Filter them out
    # before computing the final round so a `--cross-rank` run cannot
    # accidentally have the ranking responses double-counted as
    # primary votes.
    primary = [
        r for r in results if not getattr(r, "is_ranking_round", False)
    ]
    if not primary:
        return []
    final_round = max(result_round(result.name) for result in primary)
    return [result for result in primary if result_round(result.name) == final_round]


_RECOMMENDATION_PREFIX_RE = re.compile(
    r"^RECOMMENDATION:\s*(?:yes|no|tradeoff)\s*[-–—:]?\s*",
    re.IGNORECASE,
)


def _strip_recommendation_prefix(line: str) -> str:
    return _RECOMMENDATION_PREFIX_RE.sub("", line, count=1).strip()


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def _participant_disagreement_entry(result: ParticipantResult) -> dict[str, Any]:
    if result.ok:
        label = recommendation_label(result.output)
        summary = _strip_recommendation_prefix(recommendation_line(result.output))
    else:
        label = None
        summary = _first_nonempty_line(result.error or "")
    return {"name": result.name, "ok": result.ok, "label": label, "summary": summary}


def remaining_disagreement_payload(
    final_results: list[ParticipantResult], metadata: dict[str, Any]
) -> dict[str, Any] | None:
    if not metadata.get("final_disagreement_detected"):
        return None
    if not final_results:
        return None
    counts = recommendation_counts(final_results)
    return {
        "status": metadata.get("deliberation_status"),
        "ran_max_rounds_unresolved": metadata.get("deliberation_status")
        == "ran_max_rounds_unresolved",
        "counts": counts,
        "participants": [_participant_disagreement_entry(r) for r in final_results],
    }


def _minority_callout(remaining: dict[str, Any]) -> str | None:
    """Scannable minority note for the remaining-disagreement count line.

    Returns a string like ``minority: codex, gemini held no`` when there is a
    single CLEAR majority trinary label and a non-empty minority of OTHER
    trinary labels. Returns ``None`` (skip the callout) when:

    - there is no clear majority (two or more trinary labels tie for the top),
    - the council is unanimous (no minority), or
    - no trinary label was emitted at all.

    ``unknown`` / ``None`` labels are intentionally excluded from both the
    majority computation and the minority callout — they're already shown in
    the per-peer label list, and surfacing them here would be noise.
    """
    counts = remaining["counts"]
    trinary = {label: counts[label] for label in ("yes", "no", "tradeoff")}
    top = max(trinary.values())
    if top == 0:
        return None
    leaders = [label for label, n in trinary.items() if n == top]
    if len(leaders) != 1:
        # Ambiguous tie among top labels → no single majority.
        return None
    majority = leaders[0]
    minority: list[tuple[str, str]] = []
    for entry in remaining["participants"]:
        label = entry.get("label")
        if label in ("yes", "no", "tradeoff") and label != majority:
            minority.append((entry["name"], label))
    if not minority:
        return None
    # Group minority peers by the label they held so the callout reads
    # naturally even when the minority itself is split across labels.
    by_label: dict[str, list[str]] = {}
    for name, label in minority:
        by_label.setdefault(label, []).append(name)
    parts = [
        f"{', '.join(names)} held {label}" for label, names in by_label.items()
    ]
    return "minority: " + "; ".join(parts)


def _missing_label_reason(result: ParticipantResult) -> str:
    if result.ok:
        if recommendation_label(result.output) == "unknown":
            return "missing label"
        return "labeled"
    if is_timeout_error(result.error):
        return "timeout"
    if is_context_overflow_error(result.error):
        return "context overflow"
    return "failed"


def context_overflow_excluded_names(
    results: list[ParticipantResult],
) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for result in results:
        if result.ok or not is_context_overflow_error(result.error):
            continue
        base = ROUND_SUFFIX_RE.sub("", result.name)
        if base in seen:
            continue
        seen.add(base)
        names.append(base)
    return names


def context_overflow_records(
    results: list[ParticipantResult],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in results:
        if result.ok or not is_context_overflow_error(result.error):
            continue
        records.append(
            {
                "name": result.name,
                "estimated_tokens": result.prompt_tokens,
                "error": result.error,
            }
        )
    return records


def _participant_quorum_entry(result: ParticipantResult) -> dict[str, Any]:
    if result.ok:
        label = recommendation_label(result.output)
        return {
            "name": result.name,
            "ok": True,
            "label": None if label == "unknown" else label,
            "reason": _missing_label_reason(result),
        }
    return {
        "name": result.name,
        "ok": False,
        "label": None,
        "reason": _missing_label_reason(result),
        "error": _first_nonempty_line(result.error or ""),
    }


def quorum_summary(
    final_results: list[ParticipantResult], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Pure helper: derive labeled_quorum / min_quorum / degraded from results.

    Prefers values stamped onto metadata by the orchestrator; falls back to
    recomputing from final_results so transcripts written from older runs (or
    raw test fixtures) remain coherent.
    """
    labeled = metadata.get("labeled_quorum")
    if labeled is None:
        labeled = labeled_quorum_count(final_results)
    threshold = metadata.get("min_quorum")
    if threshold is None:
        threshold = default_min_quorum(len(final_results))
    degraded = metadata.get("degraded")
    if degraded is None:
        degraded = labeled < threshold
    return {
        "labeled_quorum": int(labeled),
        "min_quorum": int(threshold),
        "degraded": bool(degraded),
    }


def degraded_consensus_payload(
    final_results: list[ParticipantResult], metadata: dict[str, Any]
) -> dict[str, Any] | None:
    summary = quorum_summary(final_results, metadata)
    if not summary["degraded"]:
        return None
    missing = [
        _participant_quorum_entry(result)
        for result in final_results
        if _missing_label_reason(result) != "labeled"
    ]
    return {
        "labeled_quorum": summary["labeled_quorum"],
        "min_quorum": summary["min_quorum"],
        "missing": missing,
    }


def write_transcript(
    markdown_path: Path,
    json_path: Path,
    *,
    question: str,
    mode: str,
    current: str | None,
    participants: list[str],
    prompt: str,
    results: list[ParticipantResult],
    transparent: bool = False,
    metadata: dict[str, Any] | None = None,
    parent_run_id: str | None = None,
) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = metadata or {}
    ok_count = sum(1 for result in results if result.ok)
    final_results = final_round_results(results)
    final_ok_count = sum(1 for result in final_results if result.ok)
    elapsed_total = sum(result.elapsed_seconds for result in results)
    token_total = sum(result.total_tokens or 0 for result in results)
    cost_total = sum(result.cost_usd or 0 for result in results)
    recommendations = recommendation_counts(final_results)
    quorum = quorum_summary(final_results, metadata)
    quorum_bullet = (
        f"- Quorum: {quorum['labeled_quorum']} of {len(final_results)} peers "
        f"labeled (min: {quorum['min_quorum']})"
    )
    if quorum["degraded"]:
        quorum_bullet += " — **DEGRADED**"
    overflow_names = context_overflow_excluded_names(final_results)
    overflow_bullet = (
        [f"- Excluded for context overflow: {', '.join(overflow_names)}"]
        if overflow_names
        else []
    )
    lines = [
        "# LLM Council Transcript",
        "",
        f"- Mode: `{mode}`",
        f"- Current agent: `{current or 'unknown'}`",
        f"- Participants: {', '.join(f'`{name}`' for name in participants)}",
        f"- Successful responses: {ok_count}/{len(results)} total",
        f"- Final-round successful responses: {final_ok_count}/{len(final_results)}",
        f"- Participant elapsed total: `{elapsed_total:.1f}s`",
        f"- Tokens reported: `{token_total}`",
        f"- Cost reported: `${cost_total:.6f}`",
        f"- Rounds: `{metadata.get('rounds', 1)}`",
        f"- Deliberation: {deliberation_summary(metadata)}",
        *convergence_summary_lines(metadata),
        *(
            [f"- Parent run: `{parent_run_id}`"]
            if parent_run_id
            else []
        ),
        "- Recommendations (final round): "
        f"`{recommendations['yes']} yes / {recommendations['no']} no / "
        f"{recommendations['tradeoff']} tradeoff / {recommendations['unknown']} unknown`",
        quorum_bullet,
        *overflow_bullet,
        "",
        "## Question",
        "",
        question.strip(),
        "",
    ]

    images = metadata.get("images") or []
    if images:
        lines.extend(["## Images", ""])
        for entry in images:
            label = entry.get("path") or "?"
            mime = entry.get("mime") or "?"
            size = entry.get("size")
            sha = (entry.get("sha256") or "")[:12]
            size_str = f"{size} bytes" if size is not None else "?"
            lines.append(f"- `{label}` ({mime}, {size_str}, sha256:{sha})")
        lines.append("")

    if transparent:
        lines.extend(["## Model Comparison", ""])
        lines.extend(model_comparison(results))
        lines.append("")

    lines.extend(["## Participant Responses", ""])

    for result in results:
        if result.ok:
            status = "ok"
        elif is_timeout_error(result.error):
            status = "timeout"
        elif is_context_overflow_error(result.error):
            status = "excluded"
        else:
            status = "error"
        cache_tag = " [cached]" if result.from_cache else ""
        lines.extend(
            [
                f"### {result.name} ({status}){cache_tag}",
                "",
                f"- Model: `{result.model or 'cli default (unreported)'}`",
                f"- Elapsed: `{result.elapsed_seconds:.1f}s`",
            ]
        )
        if result.total_tokens is not None:
            lines.append(f"- Tokens: `{result.total_tokens}`")
        if result.cost_usd is not None:
            lines.append(f"- Cost: `${result.cost_usd:.6f}`")
        if result.command:
            lines.append(f"- Command: `{command_for_display(result.command)}`")
        lines.append("")
        if result.ok:
            lines.extend([result.output.strip() or "[empty response]", ""])
        else:
            lines.extend(["```", result.error.strip() or "[unknown error]", "```", ""])
            if result.output.strip():
                lines.extend(["Captured output:", "", result.output.strip(), ""])

    remaining = remaining_disagreement_payload(final_results, metadata)
    if remaining is not None:
        counts = remaining["counts"]
        lines.extend(["## Remaining disagreement", ""])
        count_line = (
            "Recommendations (final round): "
            f"{counts['yes']} yes / {counts['no']} no / "
            f"{counts['tradeoff']} tradeoff / {counts['unknown']} unknown"
        )
        minority = _minority_callout(remaining)
        if minority:
            count_line += f" — {minority}"
        lines.append(count_line)
        lines.append("")
        for entry in remaining["participants"]:
            label = entry["label"] or "—"
            summary = entry["summary"] or "—"
            lines.append(f"- {entry['name']}: {label} — {summary}")
        if remaining["ran_max_rounds_unresolved"]:
            rounds_run = metadata.get("rounds")
            rounds_phrase = (
                f" ({rounds_run})" if isinstance(rounds_run, int) else ""
            )
            lines.extend(
                [
                    "",
                    f"Deliberation reached the maximum configured rounds{rounds_phrase} "
                    "without the council converging on a single recommendation.",
                ]
            )
        lines.append("")

    degraded = degraded_consensus_payload(final_results, metadata)
    if degraded is not None:
        lines.extend(["## Degraded consensus", ""])
        if degraded["missing"]:
            lines.append(
                f"**{degraded['labeled_quorum']} of {len(final_results)} peers produced a "
                f"label, below the configured minimum of {degraded['min_quorum']}.** "
                "Treat the recommendation above with caution: the surviving "
                "peer(s) may not be representative of the council."
            )
            lines.append("")
            lines.append("Peers that did not label:")
            lines.append("")
            for entry in degraded["missing"]:
                reason = entry.get("reason") or "—"
                detail = entry.get("error")
                if detail:
                    lines.append(f"- {entry['name']}: {reason} — {detail}")
                else:
                    lines.append(f"- {entry['name']}: {reason}")
            lines.append("")
        else:
            lines.append(
                f"**The configured `min_quorum` of {degraded['min_quorum']} exceeds "
                f"the {degraded['labeled_quorum']} peer(s) that produced a label, "
                "even though every selected peer responded.** This is a configuration "
                "issue, not a participant failure: lower `min_quorum` or add more "
                "peers if you want a non-degraded result."
            )
            lines.append("")

    # H2 independence warning (advisory-only). Rendered near the quorum /
    # degraded summary; only present when the orchestrator fired it. Does
    # NOT affect quorum/degraded — purely informational.
    independence_warning = metadata.get("independence_warning")
    if isinstance(independence_warning, dict):
        distinct = independence_warning.get("distinct_vendors")
        required = independence_warning.get("required")
        families = independence_warning.get("families") or []
        labeled = independence_warning.get("labeled_quorum")
        lines.append(
            f"- ⚠️ Independence warning: all {labeled} labeled vote(s) came "
            f"from {distinct} vendor family/families "
            f"(families: {', '.join(families) if families else '—'}); "
            f"required ≥ {required} distinct. Same-vendor agreement may "
            "overstate independent corroboration."
        )

    finding_matrix_md = metadata.get("finding_matrix")
    if isinstance(finding_matrix_md, dict) and (
        finding_matrix_md.get("consensus_blockers")
        or finding_matrix_md.get("single_peer_concerns")
    ):
        lines.extend(["## Finding Matrix", ""])
        consensus = finding_matrix_md.get("consensus_blockers") or []
        if consensus:
            lines.append("**Consensus blockers** (>=2 peers, overlapping verified ranges):")
            lines.append("")
            for entry in consensus:
                peers = ", ".join(entry.get("peers") or [])
                location = ""
                path = entry.get("path")
                if path:
                    lo = entry.get("start_line")
                    hi = entry.get("end_line")
                    location = f" at `{path}:{lo}-{hi}`"
                lines.append(
                    f"- {entry.get('id')} [{entry.get('severity')}]{location} — {peers}"
                )
                claim = (entry.get("claim") or "").strip()
                if claim:
                    lines.append(f"  - {claim}")
            lines.append("")
        singles = finding_matrix_md.get("single_peer_concerns") or []
        if singles:
            lines.append("**Single-peer concerns:**")
            lines.append("")
            for entry in singles:
                peer = entry.get("peer") or "?"
                location = ""
                path = entry.get("path")
                if path:
                    lo = entry.get("start_line")
                    hi = entry.get("end_line")
                    location = f" at `{path}:{lo}-{hi}`"
                    if entry.get("unverified"):
                        location += " (unverified)"
                elif entry.get("unverified"):
                    location = " (unverified)"
                lines.append(
                    f"- {peer} [{entry.get('severity')}]{location}"
                )
                claim = (entry.get("claim") or "").strip()
                if claim:
                    lines.append(f"  - {claim}")
            lines.append("")

    cross_rank_scores_md = metadata.get("cross_rank_scores")
    anonymization_map_md = metadata.get("anonymization_map")
    if isinstance(cross_rank_scores_md, dict) and cross_rank_scores_md:
        lines.extend(["## Cross-Rank Scores", ""])
        lines.append(
            "Lower mean rank position = ranked higher by peers (1.0 = "
            "unanimously first). Anonymization map persisted below for "
            "de-anonymization."
        )
        lines.append("")
        lines.append("| Peer | Mean Rank Position |")
        lines.append("| --- | --- |")
        for name, score in sorted(
            cross_rank_scores_md.items(), key=lambda kv: kv[1]
        ):
            lines.append(f"| {name} | {score:.2f} |")
        lines.append("")
        if isinstance(anonymization_map_md, dict) and anonymization_map_md:
            lines.append("Anonymization map:")
            lines.append("")
            for name, label in sorted(anonymization_map_md.items()):
                lines.append(f"- {name} → {label}")
            lines.append("")

    synthesis_md = metadata.get("synthesis")
    if (
        isinstance(synthesis_md, dict)
        and synthesis_md.get("ok")
        and (synthesis_md.get("output") or "").strip()
    ):
        # The synthesis chair is an opt-in, PAID extra call; its decision memo
        # was previously preserved only in the JSON transcript, invisible on
        # the human-facing markdown surface. The chair's `output` already
        # carries the structured ## Decision / ## Consensus blockers / ## Dissent
        # sections, so render it verbatim under a header plus the parsed label.
        chair = synthesis_md.get("chair") or "?"
        decision = synthesis_md.get("decision_label") or "unknown"
        lines.extend([f"## Synthesis (chair: {chair})", ""])
        lines.append(f"**Decision:** {decision}")
        lines.append("")
        lines.append(synthesis_md["output"].strip())
        lines.append("")

    fence = markdown_fence(prompt)
    lines.extend(["## Prompt Sent", "", f"{fence}text", prompt, fence, ""])

    deliberation_prompts = metadata.get("deliberation_prompts")
    if isinstance(deliberation_prompts, dict):
        for round_key in sorted(deliberation_prompts.keys()):
            text = deliberation_prompts[round_key]
            if not isinstance(text, str) or not text:
                continue
            round_fence = markdown_fence(text)
            lines.extend(
                [
                    f"## Round {round_key} Prompt",
                    "",
                    f"{round_fence}text",
                    text,
                    round_fence,
                    "",
                ]
            )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    # These keys live at the TOP level of the JSON payload for downstream
    # consumers (eval harness, dashboards). We extract them from `metadata`
    # and remove them there to avoid double-serialization (the same dict
    # appearing under both `metadata.<key>` and `json_payload.<key>`).
    # v0.9.0 Feature 2: cross-rank scores and the anonymization map are
    # lifted alongside finding_matrix; all are omitted entirely when the
    # producing pass (findings / `--cross-rank`) did not run.
    # Shallow copy so the in-memory `metadata` mutation does not surprise
    # the caller (orchestrator continues to use its own reference after
    # `write_transcript` returns).
    metadata = dict(metadata)
    LIFTED_KEYS = (
        "finding_matrix",
        "cross_rank_scores",
        "anonymization_map",
        "anonymization_map_reverse",
        "cross_rank_rankings",
    )
    lifted = {k: metadata.pop(k) for k in LIFTED_KEYS if k in metadata}

    json_payload: dict[str, Any] = {
        "question": question,
        "mode": mode,
        "current": current,
        "participants": participants,
        "prompt": prompt,
        "metadata": metadata,
        "results": [result_to_dict(result) for result in results],
    }
    if parent_run_id:
        json_payload["parent_run_id"] = parent_run_id
    if remaining is not None:
        json_payload["remaining_disagreement"] = remaining
    if degraded is not None:
        json_payload["degraded_consensus"] = degraded
    overflow_records = context_overflow_records(results)
    if overflow_records:
        json_payload["context_overflow_excluded"] = overflow_records
    finding_matrix_payload = lifted.get("finding_matrix")
    if isinstance(finding_matrix_payload, dict) and (
        finding_matrix_payload.get("consensus_blockers")
        or finding_matrix_payload.get("single_peer_concerns")
    ):
        # Mirrors the shape used in MCP `structured_results`.
        json_payload["finding_matrix"] = finding_matrix_payload
    cross_rank_scores_payload = lifted.get("cross_rank_scores")
    if isinstance(cross_rank_scores_payload, dict) and cross_rank_scores_payload:
        json_payload["cross_rank_scores"] = cross_rank_scores_payload
    anonymization_map_payload = lifted.get("anonymization_map")
    if (
        isinstance(anonymization_map_payload, dict)
        and anonymization_map_payload
    ):
        json_payload["anonymization_map"] = anonymization_map_payload
    anonymization_map_reverse_payload = lifted.get("anonymization_map_reverse")
    if (
        isinstance(anonymization_map_reverse_payload, dict)
        and anonymization_map_reverse_payload
    ):
        json_payload["anonymization_map_reverse"] = (
            anonymization_map_reverse_payload
        )
    cross_rank_rankings_payload = lifted.get("cross_rank_rankings")
    if (
        isinstance(cross_rank_rankings_payload, dict)
        and cross_rank_rankings_payload
    ):
        json_payload["cross_rank_rankings"] = cross_rank_rankings_payload
    json_path.write_text(
        json.dumps(json_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    # Generate and write HTML transcript
    html_path = markdown_path.with_suffix(".html")
    html_content = _generate_html_dashboard(
        question=question,
        mode=mode,
        current=current,
        participants=participants,
        results=results,
        metadata=metadata,
        parent_run_id=parent_run_id,
        elapsed_total=elapsed_total,
        token_total=token_total,
        cost_total=cost_total,
        recommendations=recommendations,
        quorum=quorum,
    )
    try:
        html_path.write_text(html_content, encoding="utf-8")
    except OSError:
        pass


def _generate_html_dashboard(
    question: str,
    mode: str,
    current: str | None,
    participants: list[str],
    results: list[ParticipantResult],
    metadata: dict[str, Any],
    parent_run_id: str | None,
    elapsed_total: float,
    token_total: int,
    cost_total: float,
    recommendations: dict[str, int],
    quorum: dict[str, Any],
) -> str:
    import html
    def esc(text: str) -> str:
        return html.escape(text)

    synthesis = metadata.get("synthesis") or {}
    decision = synthesis.get("decision_label") or metadata.get("recommendation") or "unknown"
    decision_badge_class = f"badge-{decision.lower()}" if decision.lower() in ("yes", "no", "tradeoff") else "badge-unknown"

    peers_html = []
    for r in results:
        status = "ok" if r.ok else "error"
        cache_tag = " [cached]" if r.from_cache else ""
        from llm_council.deliberation import recommendation_label
        rec = recommendation_label(r.output) if r.ok else "unknown"
        rec_badge = f'<span class="badge badge-{rec.lower()}">{rec.upper()}</span>' if r.ok else ""
        
        stance = getattr(r, "stance", None)
        stance_class = f"stance-{stance}" if stance in ("for", "against", "neutral") else ""
        stance_label = f"Stance: {stance.upper()}" if stance else "Stance: GENERAL"

        peers_html.append(f"""
        <div class="card response-card {stance_class}">
            <div class="card-title">
                <div>
                    <strong>{esc(r.name)}</strong> 
                    <span style="font-size: 13px; color: var(--text-muted); margin-left: 8px;">
                        ({status}){cache_tag} &bull; {r.elapsed_seconds:.1f}s &bull; {r.total_tokens or 0} tokens &bull; ${r.cost_usd or 0:.6f}
                    </span>
                </div>
                <div>
                    {rec_badge}
                    <span class="badge" style="background-color: rgba(171, 125, 246, 0.15); color: #d3bcf6; border: 1px solid rgba(171, 125, 246, 0.4); margin-left: 8px;">
                        {esc(stance_label)}
                    </span>
                </div>
            </div>
            <pre>{esc(r.output) if r.ok else esc(r.error)}</pre>
        </div>
        """)

    synthesis_html = ""
    if synthesis.get("ok") and (synthesis.get("output") or "").strip():
        synthesis_html = f"""
        <div class="card" style="border-left: 4px solid var(--accent-color);">
            <div class="card-title">
                <strong>Synthesis Chair Report (Chair: {esc(synthesis.get("chair") or "?")})</strong>
                <span class="badge badge-{decision.lower()}">Decision: {esc(decision.upper())}</span>
            </div>
            <pre>{esc(synthesis["output"].strip())}</pre>
        </div>
        """

    quorum_msg = f"{quorum['labeled_quorum']} of {len(results)} peers labeled (min: {quorum['min_quorum']})"
    if quorum.get("degraded"):
        quorum_msg += " — DEGRADED"

    html_str = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Council Transcript Dashboard</title>
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-color: #c9d1d9;
            --text-muted: #8b949e;
            --primary-color: #58a6ff;
            --success-color: #2ea44f;
            --danger-color: #f85149;
            --warning-color: #db6d28;
            --accent-color: #ab7df6;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }}
        h1 {{
            margin: 0;
            font-size: 28px;
            font-weight: 600;
            background: linear-gradient(45deg, var(--primary-color), var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 600;
            border-radius: 2em;
            text-transform: uppercase;
        }}
        .badge-yes {{ background-color: rgba(46, 164, 79, 0.15); color: #56d364; border: 1px solid rgba(46, 164, 79, 0.4); }}
        .badge-no {{ background-color: rgba(248, 81, 73, 0.15); color: #ff7b72; border: 1px solid rgba(248, 81, 73, 0.4); }}
        .badge-tradeoff {{ background-color: rgba(219, 109, 40, 0.15); color: #f0883e; border: 1px solid rgba(219, 109, 40, 0.4); }}
        .badge-unknown {{ background-color: rgba(139, 148, 158, 0.15); color: #c9d1d9; border: 1px solid rgba(139, 148, 158, 0.4); }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .stat-val {{
            font-size: 24px;
            font-weight: 700;
            margin-top: 8px;
            color: var(--primary-color);
        }}
        .stat-label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .tabs {{
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            font-weight: 600;
            color: var(--text-muted);
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }}
        .tab:hover {{
            color: var(--text-color);
        }}
        .tab.active {{
            color: var(--primary-color);
            border-bottom-color: var(--primary-color);
        }}
        
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
        }}
        .card-title {{
            margin-top: 0;
            margin-bottom: 12px;
            font-size: 16px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        pre {{
            background-color: #0d1117;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 16px;
            overflow-x: auto;
            font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
            font-size: 13px;
            white-space: pre-wrap;
            margin: 0;
        }}
        
        .response-card {{
            border-left: 4px solid var(--border-color);
        }}
        .response-card.stance-for {{ border-left-color: var(--success-color); }}
        .response-card.stance-against {{ border-left-color: var(--danger-color); }}
        .response-card.stance-neutral {{ border-left-color: var(--accent-color); }}
        
        .search-box {{
            width: 100%;
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            padding: 10px 16px;
            border-radius: 6px;
            font-size: 14px;
            margin-bottom: 20px;
            box-sizing: border-box;
        }}
        .search-box:focus {{
            border-color: var(--primary-color);
            outline: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>LLM Council Dashboard</h1>
                <div style="font-size: 14px; color: var(--text-muted); margin-top: 4px;">
                    Mode: <code>{esc(mode)}</code> &bull; Current Agent: <code>{esc(current or 'unknown')}</code>
                </div>
            </div>
            <div>
                <span class="badge {decision_badge_class}" style="font-size: 16px; padding: 6px 16px;">
                    Decision: {esc(decision.upper())}
                </span>
            </div>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Elapsed Total</div>
                <div class="stat-val">{elapsed_total:.1f}s</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Tokens Used</div>
                <div class="stat-val">{token_total}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Cost (USD)</div>
                <div class="stat-val">${cost_total:.5f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Quorum</div>
                <div class="stat-val" style="font-size: 16px; margin-top: 16px;">{esc(quorum_msg)}</div>
            </div>
        </div>

        <div class="tabs">
            <div class="tab active" onclick="switchTab('debate')">Debate Timeline</div>
            <div class="tab" onclick="switchTab('summary')">Executive Report</div>
            <div class="tab" onclick="switchTab('prompt')">Prompt & Context</div>
        </div>

        <div id="debate-content" class="tab-content active">
            <input type="text" class="search-box" id="search-input" placeholder="Search responses..." onkeyup="filterResponses()">
            <div id="responses-container">
                {"".join(peers_html)}
            </div>
        </div>

        <div id="summary-content" class="tab-content">
            {synthesis_html}
            <div class="card">
                <h3 style="margin-top: 0;">Vote Summary</h3>
                <p>Yes: <strong>{recommendations.get('yes', 0)}</strong></p>
                <p>No: <strong>{recommendations.get('no', 0)}</strong></p>
                <p>Tradeoff: <strong>{recommendations.get('tradeoff', 0)}</strong></p>
                <p>Unknown: <strong>{recommendations.get('unknown', 0)}</strong></p>
            </div>
        </div>

        <div id="prompt-content" class="tab-content">
            <div class="card">
                <div class="card-title"><strong>Original Prompt</strong></div>
                <pre>{esc(question)}</pre>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabId) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            const tabEl = Array.from(document.querySelectorAll('.tab')).find(t => t.textContent.toLowerCase().includes(tabId === 'prompt' ? 'prompt' : tabId === 'summary' ? 'executive' : 'debate'));
            if (tabEl) tabEl.classList.add('active');
            
            document.getElementById(tabId + '-content').classList.add('active');
        }}

        function filterResponses() {{
            const query = document.getElementById('search-input').value.toLowerCase();
            document.querySelectorAll('.response-card').forEach(card => {{
                const text = card.textContent.toLowerCase();
                if (text.includes(query)) {{
                    card.style.display = 'block';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    return html_str



def markdown_fence(text: str) -> str:
    longest = 0
    for match in re.finditer(r"`+", text):
        longest = max(longest, len(match.group(0)))
    return "`" * max(3, longest + 1)
