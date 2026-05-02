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
    """

    visited: set[str] = set()
    current = run_id
    depth = 0
    while current and depth < max_depth:
        normalized = normalize_run_id(current)
        if normalized in visited:
            break
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
    rounds = []
    for result in results:
        match = ROUND_SUFFIX_RE.search(str(result.get("name") or ""))
        rounds.append(int(match.group(1)) if match else 1)
    final_round = max(rounds) if rounds else 1
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


def transcript_records(base_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path, mtime in sorted(
        _existing_paths(base_dir.glob("*.json")), key=lambda item: item[1]
    ):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
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
    if result.from_cache:
        payload["from_cache"] = True
    if result.stance is not None:
        payload["stance"] = result.stance
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
    final_round = max(result_round(result.name) for result in results)
    return [result for result in results if result_round(result.name) == final_round]


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
                f"- Model: `{result.model or 'cli default'}`",
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
        lines.append(
            "Recommendations (final round): "
            f"{counts['yes']} yes / {counts['no']} no / "
            f"{counts['tradeoff']} tradeoff / {counts['unknown']} unknown"
        )
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
    json_path.write_text(
        json.dumps(json_payload, indent=2) + "\n",
        encoding="utf-8",
    )


def markdown_fence(text: str) -> str:
    longest = 0
    for match in re.finditer(r"`+", text):
        longest = max(longest, len(match.group(0)))
    return "`" * max(3, longest + 1)
