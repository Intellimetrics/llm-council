"""Synthesis-chair phase (Pick B of the post-council-review build plan).

A configured chair reads round-final peer results and emits a short decision
memo (blockers, dissent, verification plan). It is opt-in (``defaults.synthesize``
or ``--synthesize``), runs at most once per council run, and never changes
the headline recommendation: chair output is metadata, not a vote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_council.adapters import (
    CacheContext,
    ParticipantResult,
    run_participant,
)
from llm_council.deliberation import (
    has_disagreement,
    recommendation_label,
)
from llm_council.findings import FindingMatrix


MAX_SYNTHESIS_PROMPT_CHARS_DEFAULT = 60_000
MAX_RATIONALE_CHARS = 320


def should_synthesize(
    synthesize_flag: bool,
    metadata: dict[str, Any],
) -> bool:
    """Decide whether the synthesis chair should be invoked.

    Returns True only when the chair has something useful to do:

    - **Skip on universal abdication** (regardless of ``synthesize_flag``).
      Abdicated peers are excluded from the chair's input, so a chair call
      after universal abdication would receive an empty peer list. The
      ``metadata["universal_abdication"]`` payload (merged blockers) is
      already the deliverable.
    - **Run on explicit ``synthesize_flag``** when not blocked by the
      above. The user opted in; cost is on them.
    - **Auto-run only on unresolved deliberation**
      (``deliberation_status == "ran_max_rounds_unresolved"``). Never
      auto-run on agreement — synthesizing "everyone agreed" wastes a
      peer call.
    """
    if metadata.get("universal_abdication"):
        return False
    if synthesize_flag:
        return True
    status = metadata.get("deliberation_status")
    return status == "ran_max_rounds_unresolved"


def universal_abdication(results: list[ParticipantResult]) -> dict[str, Any] | None:
    """Short-circuit: if every labeled-eligible peer abdicated, return a
    merged-blockers payload instead of paying for round 2 or synthesis.

    Returns ``None`` when at least one peer produced a usable label.
    """
    if not results:
        return None
    eligible = [r for r in results if r.output]
    if not eligible:
        return None
    abdicated = [r for r in eligible if (r.error or "").startswith("AbdicatedResponse:")]
    if len(abdicated) != len(eligible):
        return None
    merged: list[str] = []
    seen: set[str] = set()
    for result in abdicated:
        for item in (result.blockers or ()):
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return {
        "recommendation": "unknown",
        "reason": "all_peers_abdicated",
        "blockers": merged,
        "abdicated_peers": [r.name for r in abdicated],
    }


def select_synthesizer(
    config: dict[str, Any],
    participant_cfg: dict[str, Any],
    *,
    stances: dict[str, str] | None,
    current: str | None,
) -> str:
    """Resolve the chair participant name; fail loudly when unset.

    There is intentionally no silent default — when ``synthesize=True``
    and ``defaults.synthesizer`` is unset, this raises so the user has
    to choose rather than the requester biasing the chair by default.

    Valid values for ``defaults.synthesizer``:

    - **a participant name** in ``participant_cfg`` (e.g. ``"gemini"``).
    - **``"neutral_peer"``**: pick whichever peer was assigned
      ``stance=neutral`` for this run. Requires a stance-assigning mode
      (``consensus``/``deliberate``); raises otherwise.
    - **``"current"``**: use the host CLI. The host MUST also be a
      configured participant for this run. Peer-only modes that exclude
      the host raise rather than silently falling back — keeping
      requester-bias opt-in.
    """
    defaults = config.get("defaults") or {}
    raw = defaults.get("synthesizer")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise ValueError(
            "synthesize=True but defaults.synthesizer is not configured. "
            "Set defaults.synthesizer to a participant name, 'neutral_peer', "
            "or 'current' in .llm-council.yaml — there is no silent default "
            "so the requester does not bias the chair."
        )
    raw = str(raw).strip()
    if raw == "current":
        if not current:
            raise ValueError(
                "defaults.synthesizer='current' but no host CLI was detected "
                "(execute_council current=None). Set synthesizer to a "
                "participant name instead."
            )
        # Host CLI not configured as a participant = peer-only mode (or
        # the host was intentionally excluded). Fall back loudly rather
        # than silently injecting the requester as chair.
        if current not in participant_cfg:
            raise ValueError(
                f"defaults.synthesizer='current' but host CLI '{current}' "
                "is not a configured participant for this run (peer-only "
                "mode, or `current` excluded from the council). Set "
                "synthesizer to a configured participant name."
            )
        return current
    if raw == "neutral_peer":
        if not stances:
            raise ValueError(
                "defaults.synthesizer='neutral_peer' but no stances were "
                "assigned for this run (mode does not assign for/against/"
                "neutral). Pick a specific participant name instead."
            )
        for name, stance in stances.items():
            if stance == "neutral" and name in participant_cfg:
                return name
        raise ValueError(
            "defaults.synthesizer='neutral_peer' but no participant in "
            "this run has stance=neutral. Pick a specific participant name."
        )
    if raw not in participant_cfg:
        raise ValueError(
            f"defaults.synthesizer='{raw}' is not a configured participant. "
            f"Available: {sorted(participant_cfg)}"
        )
    return raw


def _format_envelope_item(item: Any) -> str:
    """Render an envelope-list entry as a human-readable bullet.

    Evidence entries are stored as ``{"text": str, "tag": str | None}``
    dicts (see ``adapters._parse_tagged_entry``); other envelope lists
    (blockers / tests_to_run / assumptions) stay as plain strings. This
    helper handles both so the chair LLM never sees raw stringified
    dict literals like ``{'text': '...', 'tag': 'PUBLISHED'}`` in its
    prompt. Format: ``[TAG] text`` when a tag is present, otherwise
    ``text``.
    """
    if isinstance(item, dict):
        text = item.get("text") or ""
        tag = item.get("tag")
        if tag:
            return f"[{str(tag).upper()}] {text}"
        return str(text)
    return str(item)


def _summary_rationale(output: str) -> str:
    """Compact rationale: the line after RECOMMENDATION, truncated."""
    in_label = False
    parts: list[str] = []
    for raw in (output or "").splitlines():
        line = raw.strip()
        if not in_label:
            lowered = line.lower()
            if lowered.startswith("recommendation:") or lowered.startswith("**recommendation"):
                in_label = True
                rest = line.split(":", 1)[-1].strip(" *_")
                if rest:
                    parts.append(rest)
            continue
        if not line:
            if parts:
                break
            continue
        parts.append(line)
        if sum(len(p) for p in parts) >= MAX_RATIONALE_CHARS:
            break
    text = " ".join(parts).strip()
    if len(text) > MAX_RATIONALE_CHARS:
        text = text[: MAX_RATIONALE_CHARS - 3].rstrip() + "..."
    return text


def _format_finding_matrix_block(matrix: FindingMatrix) -> list[str]:
    """Render the FindingMatrix as appended lines for the chair prompt.

    Returns an empty list when the matrix has no clusters and no
    single-peer concerns — preserves the pre-Phase-F prompt shape when
    no peer emitted a structured FINDINGS block.

    The block is purely informational for the chair: it surfaces which
    findings ≥2 peers anchored to overlapping verified ranges vs which
    findings only one peer named. It must NEVER be fed back to peers
    during deliberation — see the module docstring in findings.py.
    """
    if matrix.is_empty():
        return []

    lines: list[str] = [
        "",
        "## Finding matrix (post-deliberation; synthesis-input only)",
        "",
        "Weight CONSENSUS BLOCKERS more heavily — these are findings >=2 peers",
        "anchored to overlapping verified line ranges. SINGLE-PEER CONCERNS",
        "are worth surfacing but should not block on their own.",
        "",
    ]

    if matrix.clusters:
        lines.append("CONSENSUS BLOCKERS (>=2 peers, overlapping verified ranges):")
        for cluster in matrix.clusters:
            location = ""
            if cluster.verified_path and cluster.verified_range is not None:
                lo, hi = cluster.verified_range
                location = f" at {cluster.verified_path}:{lo}-{hi}"
            peers = ", ".join(cluster.peers)
            lines.append(
                f"  {cluster.id} [{cluster.severity}]{location} — {peers}"
            )
            # Use the first contributing finding's claim as the
            # representative; do not paraphrase across peers.
            if cluster.findings:
                claim = cluster.findings[0].claim.strip()
                if claim:
                    lines.append(f'    "{claim}"')
        lines.append("")
    else:
        lines.append("CONSENSUS BLOCKERS: (none)")
        lines.append("")

    if matrix.single_peer_concerns:
        lines.append("SINGLE-PEER CONCERNS:")
        for finding in matrix.single_peer_concerns:
            location = ""
            if finding.verified_ref is not None:
                ref = finding.verified_ref
                location = f" at {ref.path}:{ref.start_line}-{ref.end_line}"
                if finding.verified is False:
                    location += " (unverified)"
            else:
                location = " (unverified)"
            lines.append(
                f"  {finding.peer} [{finding.severity}]{location}"
            )
            claim = (finding.claim or "").strip()
            if claim:
                lines.append(f'    "{claim}"')
        lines.append("")

    return lines


def build_synthesis_prompt(
    question: str,
    results: list[ParticipantResult],
    convergence: dict[str, Any] | None,
    *,
    max_chars: int = MAX_SYNTHESIS_PROMPT_CHARS_DEFAULT,
    finding_matrix: FindingMatrix | None = None,
) -> str:
    """Compact chair prompt. Cites peers by name; consumes pre-computed
    convergence metadata rather than re-deriving Jaccard drift on
    output text the orchestrator already analyzed."""
    lines: list[str] = [
        "You are the synthesis chair for an llm-council deliberation.",
        "Your job: produce a tight decision memo, NOT another vote. Your output",
        "is recorded as metadata; the council's headline recommendation comes",
        "from the peers' majority label, not from you.",
        "",
        "Be even-handed: you are the moderator, not a third debater. Do not",
        "introduce a new position or favor any peer — including yourself if you",
        "also voted. Cite peers by name and aggregate the structured envelope",
        "fields below instead of paraphrasing — your memo synthesizes the",
        "council, it is not a fresh opinion, and you must not invent blockers,",
        "evidence, or risks that peers did not name.",
        "",
        "Output the following sections, each prefixed by the heading shown:",
        "",
        "### Decision",
        "One paragraph: yes / no / tradeoff, plus the operational implication.",
        "If the peers did not converge, state the verdict AND explicitly name",
        "the genuine remaining disagreement (which peers held the minority view",
        "and on what) rather than papering over it.",
        "",
        "### Consensus blockers",
        "Bullet list of blockers named by 2+ peers (deduplicate). Prefix each",
        "bullet with the peers who raised it, e.g. `claude, gemini: <blocker>`.",
        "Empty list if peers blocked on disjoint things.",
        "",
        "### Single-peer concerns",
        "Bullet list of one-peer findings worth surfacing but not blocking.",
        "Attribute each to the peer name.",
        "",
        "### Dissent",
        "Per-peer line: `peer-name: label - one-sentence position`. Skip",
        "peers without a usable label.",
    ]
    if convergence:
        lines.extend(
            [
                "",
                "### How positions moved",
                "For each peer that changed across rounds, one line: what they",
                "conceded or held and why, grounded ONLY in the per-peer",
                "Convergence states (converging / diverging / stable) and",
                "similarity values in the Convergence block below. Do not invent",
                "concessions or reference round-1 text you were not given; if a",
                "peer did not move, say so or omit it.",
            ]
        )
    lines.extend(
        [
            "",
            "### Verification plan",
            "Numbered list. Pull from peers' TESTS_TO_RUN: where present.",
            "",
            "## Original question",
            "",
            question.strip(),
            "",
            "## Peer responses (final round)",
            "",
        ]
    )
    for result in results:
        if not result.ok or not result.output:
            continue
        label = recommendation_label(result.output)
        stance = f" stance={result.stance}" if result.stance else ""
        lines.append(f"### {result.name} (label={label}{stance})")
        lines.append("")
        rationale = _summary_rationale(result.output)
        if rationale:
            lines.append(f"- rationale: {rationale}")
        if result.effort:
            lines.append(f"- EFFORT: {result.effort}")
        if result.confidence:
            lines.append(f"- CONFIDENCE: {result.confidence}")
        if result.risk:
            lines.append(f"- RISK: {result.risk}")
        for label_key, items in (
            ("BLOCKERS", result.blockers),
            ("EVIDENCE", result.evidence),
            ("TESTS_TO_RUN", result.tests_to_run),
            ("ASSUMPTIONS", result.assumptions),
        ):
            if items:
                lines.append(f"- {label_key}:")
                for item in items:
                    lines.append(f"  - {_format_envelope_item(item)}")
        lines.append("")
    if convergence:
        lines.append("## Convergence (pre-computed; do not re-derive)")
        lines.append("")
        for round_key in sorted(convergence):
            records = convergence[round_key]
            lines.append(f"### Round {round_key}")
            for record in records or ():
                if isinstance(record, dict):
                    lines.append(
                        f"- {record.get('participant')}: state={record.get('state')}"
                        f" similarity={record.get('similarity')}"
                    )
            lines.append("")
    # Append the FindingMatrix block last so the chair has the
    # already-rendered envelope context (BLOCKERS/EVIDENCE/etc) above when
    # it reads the cross-peer cluster summary. When the matrix is None or
    # empty the prompt shape is unchanged from v0.7.x.
    if finding_matrix is not None:
        lines.extend(_format_finding_matrix_block(finding_matrix))
    prompt = "\n".join(lines)
    if len(prompt) > max_chars:
        prompt = prompt[: max_chars - 80].rstrip() + (
            "\n\n[synthesis prompt truncated by llm-council]\n"
        )
    return prompt


async def run_synthesis_chair(
    *,
    question: str,
    results: list[ParticipantResult],
    convergence: dict[str, Any] | None,
    participant_cfg: dict[str, Any],
    cwd: Path,
    chair_name: str,
    max_chars: int = MAX_SYNTHESIS_PROMPT_CHARS_DEFAULT,
    finding_matrix: FindingMatrix | None = None,
) -> dict[str, Any]:
    """Invoke the chair once, return a metadata-only synthesis payload.

    Caching is disabled (each council run is unique). The result is NOT
    appended to ``results`` — quorum and agreement_count must stay derived
    from peer votes only.
    """
    prompt = build_synthesis_prompt(
        question,
        results,
        convergence,
        max_chars=max_chars,
        finding_matrix=finding_matrix,
    )
    cfg = dict(participant_cfg.get(chair_name) or {})
    # Chair output is a decision memo, not a vote. Override
    # require_recommendation so the standard label-validation path does NOT
    # reject the chair's structured ## Decision / ## Consensus blockers /
    # ## Dissent sections, and so the label-only repair retry does not fire.
    cfg["require_recommendation"] = False
    cfg["retry_on_missing_label"] = False
    cache_ctx = CacheContext(cwd=cwd, cache_disabled=True)
    chair_result = await run_participant(
        chair_name, cfg, prompt, cwd, cache_ctx=cache_ctx
    )
    return {
        "chair": chair_name,
        "ok": chair_result.ok,
        "output": chair_result.output,
        "error": chair_result.error,
        "decision_label": (
            recommendation_label(chair_result.output) if chair_result.ok else "unknown"
        ),
        "blockers": list(chair_result.blockers),
        "evidence": list(chair_result.evidence),
        "tests_to_run": list(chair_result.tests_to_run),
        "elapsed_seconds": round(chair_result.elapsed_seconds, 3),
        "model": chair_result.model,
        "total_tokens": chair_result.total_tokens,
        "cost_usd": chair_result.cost_usd,
        "consumed_convergence": bool(convergence),
        "prompt_chars": len(prompt),
    }


__all__ = [
    "MAX_SYNTHESIS_PROMPT_CHARS_DEFAULT",
    "build_synthesis_prompt",
    "has_disagreement",
    "run_synthesis_chair",
    "select_synthesizer",
    "should_synthesize",
    "universal_abdication",
]
