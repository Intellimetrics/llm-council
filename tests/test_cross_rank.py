"""v0.9.0 Feature 2 — Anonymized cross-ranking flag.

Covers:
- Anonymization map (stable, deterministic, persisted)
- Ranking prompt builder (self-exclusion, label rendering, terse format)
- FINAL RANKING parser (multiple form variants, validity check)
- Orchestrator integration: opt-in gate, ranking-round tagging,
  metadata persistence, MAD-literature "no leak to round 2" guard
- Stats: rank_position_mean surfaces in aggregate_reliability
- Cache: is_ranking_round=True round-trips

Mirrors the orchestrator fixture style from
``tests/test_continue_debate.py`` and the dataclass/parser cadence
of ``tests/test_findings.py``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_council.adapters import (
    ParticipantResult,
    _result_from_cache_payload,
)
from llm_council.cache import build_payload as cache_build_payload
from llm_council.context import (
    CROSS_RANK_MIN_PEERS,
    build_anonymization_map,
    build_ranking_prompt,
    compute_rank_position_means,
    parse_final_ranking,
)
from llm_council.deliberation import build_deliberation_prompt


# --- Anonymization map ---------------------------------------------------

def test_anonymization_map_is_deterministic_sorted():
    """Names sorted alphabetically before label assignment."""
    m1 = build_anonymization_map(["gemini", "claude", "codex"])
    m2 = build_anonymization_map(["codex", "claude", "gemini"])
    assert m1 == m2
    assert m1["claude"] == "Response A"
    assert m1["codex"] == "Response B"
    assert m1["gemini"] == "Response C"


def test_anonymization_map_reverse_lookup_works():
    """Operator should be able to de-anonymize from the persisted map."""
    m = build_anonymization_map(["beta", "alpha", "gamma"])
    reverse = {v.replace("Response ", ""): k for k, v in m.items()}
    assert reverse["A"] == "alpha"
    assert reverse["B"] == "beta"
    assert reverse["C"] == "gamma"


def test_anonymization_map_empty_input():
    assert build_anonymization_map([]) == {}


def test_anonymization_map_handles_excel_column_style_overflow():
    """27 peers should not collide — label rolls over to AA."""
    names = [f"peer{i:02d}" for i in range(27)]
    m = build_anonymization_map(names)
    assert m["peer00"] == "Response A"
    assert m["peer25"] == "Response Z"
    assert m["peer26"] == "Response AA"


# --- Ranking prompt builder ---------------------------------------------

def test_ranking_prompt_excludes_own_response():
    """Peer's own response should NOT appear in their ranking bundle."""
    anon = {"a": "Response A", "b": "Response B", "c": "Response C"}
    prompt = build_ranking_prompt(
        peer_name="a",
        own_response="A's own response — should NEVER appear",
        other_peers={"b": "B's text body", "c": "C's text body"},
        anonymization_map=anon,
        question="Should we ship?",
    )
    assert "A's own response" not in prompt
    assert "B's text body" in prompt
    assert "C's text body" in prompt
    assert "Response B:" in prompt
    assert "Response C:" in prompt
    # The peer's own label should not appear as a section header either.
    assert "Response A:" not in prompt


def test_ranking_prompt_asks_for_final_ranking_terse():
    """Prompt must ask for `FINAL RANKING:` and a short reply."""
    anon = {"a": "Response A", "b": "Response B"}
    prompt = build_ranking_prompt(
        peer_name="a",
        own_response="X",
        other_peers={"b": "Y"},
        anonymization_map=anon,
        question="?",
    )
    assert "FINAL RANKING:" in prompt
    # Encourage terseness — the literal hint should be present.
    assert "short" in prompt.lower() or "one line" in prompt.lower()


def test_ranking_prompt_echoes_question():
    anon = {"a": "Response A", "b": "Response B"}
    prompt = build_ranking_prompt(
        peer_name="a",
        own_response="X",
        other_peers={"b": "Y"},
        anonymization_map=anon,
        question="Original question text marker.",
    )
    assert "Original question text marker." in prompt


# --- FINAL RANKING parser -----------------------------------------------

def test_parse_final_ranking_space_separated():
    out = parse_final_ranking("FINAL RANKING: B A C", {"A", "B", "C"})
    assert out == ["B", "A", "C"]


def test_parse_final_ranking_comma_separated():
    out = parse_final_ranking(
        "Some preamble\n**FINAL RANKING:** B, A, C\n", {"A", "B", "C"}
    )
    assert out == ["B", "A", "C"]


def test_parse_final_ranking_numbered_block():
    out = parse_final_ranking(
        "FINAL RANKING:\n1. B\n2. A\n3. C\n", {"A", "B", "C"}
    )
    assert out == ["B", "A", "C"]


def test_parse_final_ranking_arrow_separated():
    out = parse_final_ranking(
        "FINAL RANKING: B -> A -> C", {"A", "B", "C"}
    )
    assert out == ["B", "A", "C"]


def test_parse_final_ranking_rejects_unknown_label():
    out = parse_final_ranking(
        "FINAL RANKING: B A Z", {"A", "B", "C"}
    )
    # Z is not a valid label — parser refuses the whole thing.
    assert out is None


def test_parse_final_ranking_rejects_duplicate():
    out = parse_final_ranking(
        "FINAL RANKING: B A B", {"A", "B", "C"}
    )
    assert out is None


def test_parse_final_ranking_missing_returns_none():
    out = parse_final_ranking(
        "I prefer response B, then A, then C.", {"A", "B", "C"}
    )
    assert out is None


def test_parse_final_ranking_empty_output():
    assert parse_final_ranking("", {"A"}) is None
    assert parse_final_ranking("   ", {"A"}) is None


def test_parse_final_ranking_subset_accepted():
    """Peer ranks n-1 items (excludes own response) — that's a valid subset."""
    out = parse_final_ranking("FINAL RANKING: A C", {"A", "B", "C"})
    assert out == ["A", "C"]


# --- compute_rank_position_means ----------------------------------------

def test_compute_rank_position_means_simple_three_peers():
    """3 peers; b is unanimously first, c last; verify mean positions."""
    anon = {"a": "Response A", "b": "Response B", "c": "Response C"}
    # a ranks: B (1st), C (2nd)  — a does not rank itself
    # b ranks: A (1st), C (2nd)
    # c ranks: B (1st), A (2nd)
    rankings = {
        "a": ["B", "C"],
        "b": ["A", "C"],
        "c": ["B", "A"],
    }
    means = compute_rank_position_means(anon, rankings)
    # A: ranked 1st by b, 2nd by c → mean 1.5
    assert means["a"] == 1.5
    # B: ranked 1st by a, 1st by c → mean 1.0
    assert means["b"] == 1.0
    # C: ranked 2nd by a, 2nd by b → mean 2.0
    assert means["c"] == 2.0


def test_compute_rank_position_means_drops_peers_with_no_data():
    """Peer that no one rank-evaluated gets omitted, not a default value."""
    anon = {"a": "Response A", "b": "Response B", "c": "Response C"}
    # Only one ranking — covers A and B, NOT C.
    rankings = {"a": ["B"]}  # a ranks just B
    means = compute_rank_position_means(anon, rankings)
    assert "c" not in means
    assert "a" not in means  # no one ranked a here
    assert means["b"] == 1.0


def test_compute_rank_position_means_ignores_self_rank():
    """Self-rank tokens should be silently dropped (defensive).

    The peer's own label is excluded from accumulation; the position
    indices of OTHER labels are preserved as emitted (no re-ranking).
    With ``rankings={"a": ["A", "B"]}`` and a's self-rank dropped, B
    keeps its emitted position 2 (peer's choice, not a re-shift).
    """
    anon = {"a": "Response A", "b": "Response B"}
    rankings = {"a": ["A", "B"]}  # a self-ranks; that A is ignored
    means = compute_rank_position_means(anon, rankings)
    assert means["b"] == 2.0
    assert "a" not in means


# --- Cache round-trip ---------------------------------------------------

def test_cache_round_trip_preserves_is_ranking_round():
    """A ranking-round result must rehydrate with the flag set."""
    payload = cache_build_payload(
        participant_name="a:rank",
        prompt="ranking q",
        key="k",
        output="FINAL RANKING: B A",
        recommendation_label=None,
        elapsed_seconds=1.0,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cost_usd=None,
        model=None,
        command=None,
        is_ranking_round=True,
    )
    assert payload.get("is_ranking_round") is True
    rehydrated = _result_from_cache_payload("a:rank", payload)
    assert rehydrated.is_ranking_round is True


def test_cache_round_trip_omits_flag_when_false():
    """Default False should be tight in the payload (absent rather than written)."""
    payload = cache_build_payload(
        participant_name="a",
        prompt="q",
        key="k",
        output="RECOMMENDATION: yes - ok",
        recommendation_label="yes",
        elapsed_seconds=1.0,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cost_usd=None,
        model=None,
        command=None,
    )
    assert "is_ranking_round" not in payload
    rehydrated = _result_from_cache_payload("a", payload)
    assert rehydrated.is_ranking_round is False


# --- Orchestrator integration --------------------------------------------

def _round1_result(name, *, label, output=None):
    text = output or f"RECOMMENDATION: {label} - reason for {name}"
    return ParticipantResult(
        name=name,
        ok=True,
        output=text,
        error="",
        elapsed_seconds=1.0,
    )


def _ranking_response_for(peer: str) -> str:
    # Synthesize a plausible FINAL RANKING line. Each peer ranks the
    # OTHER peers' labels. With participants [a, b, c], each peer's
    # valid labels are {A, B, C} minus their own. We hand-pick rankings
    # so b is universally favored, c is least favored.
    table = {
        "a": "FINAL RANKING: B C",      # a ranks B over C
        "b": "FINAL RANKING: A C",      # b ranks A over C
        "c": "FINAL RANKING: B A",      # c ranks B over A
    }
    return table[peer]


def _run_orchestrator_with_cross_rank(
    cross_rank: bool,
    *,
    deliberate: bool = False,
    max_rounds: int = 1,
    factory=None,
):
    """Run `execute_council` with mocked round-1 + ranking pass."""
    import llm_council.orchestrator as orch_module

    captured_prompts: dict[str, str] = {}
    participants = ["a", "b", "c"]
    labels = {"a": "yes", "b": "no", "c": "tradeoff"}

    if factory is None:
        def _default_factory(name):
            return _round1_result(name, label=labels[name])
        factory = _default_factory

    call_count = {"n": 0}
    second_call_results = {}

    async def fake_run_participants(selected, *args, **kwargs):
        call_count["n"] += 1
        return [factory(name) for name in selected]

    async def fake_preflight(*args, **kwargs):
        return {}

    async def fake_run_participant(name, cfg, prompt, cwd, **kwargs):
        # Capture the prompt sent during the ranking pass so the test
        # can assert anonymization happened.
        captured_prompts[name] = prompt
        return ParticipantResult(
            name=name,
            ok=True,
            output=_ranking_response_for(name),
            error="",
            elapsed_seconds=0.5,
        )

    with patch.object(
        orch_module, "run_participants", side_effect=fake_run_participants
    ), patch.object(
        orch_module,
        "preflight_local_participants",
        side_effect=fake_preflight,
    ), patch(
        "llm_council.adapters.run_participant",
        side_effect=fake_run_participant,
    ):
        results, metadata = asyncio.run(
            orch_module.execute_council(
                participants=participants,
                participant_cfg={n: {"type": "cli"} for n in participants},
                prompt="orig prompt",
                cwd=Path("."),
                config={"defaults": {}},
                deliberate=deliberate,
                max_rounds=max_rounds,
                cross_rank=cross_rank,
                question="Should we ship the change?",
            )
        )
    return results, metadata, captured_prompts


def test_cross_rank_disabled_no_ranking_pass():
    """cross_rank=False (default) → no ranking pass, no metadata fields."""
    results, metadata, captured_prompts = _run_orchestrator_with_cross_rank(
        cross_rank=False
    )
    assert "cross_rank_scores" not in metadata
    assert "anonymization_map" not in metadata
    # No ranking-round results in the results list.
    assert not any(getattr(r, "is_ranking_round", False) for r in results)
    assert captured_prompts == {}


def test_cross_rank_below_min_peers_noop():
    """cross_rank=True but only 1 labeled peer → no-op."""
    import llm_council.orchestrator as orch_module

    async def fake_run_participants(selected, *args, **kwargs):
        # Only one peer labeled; the other two abdicate.
        out = []
        for name in selected:
            if name == "a":
                out.append(_round1_result("a", label="yes"))
            else:
                out.append(
                    ParticipantResult(
                        name=name,
                        ok=False,
                        output="",
                        error="AbdicatedResponse: not enough info",
                        elapsed_seconds=0.5,
                    )
                )
        return out

    async def fake_preflight(*args, **kwargs):
        return {}

    with patch.object(
        orch_module, "run_participants", side_effect=fake_run_participants
    ), patch.object(
        orch_module,
        "preflight_local_participants",
        side_effect=fake_preflight,
    ):
        _, metadata = asyncio.run(
            orch_module.execute_council(
                participants=["a", "b", "c"],
                participant_cfg={n: {"type": "cli"} for n in ["a", "b", "c"]},
                prompt="p",
                cwd=Path("."),
                config={"defaults": {}},
                cross_rank=True,
                question="?",
            )
        )
    assert "cross_rank_scores" not in metadata
    assert "anonymization_map" not in metadata


def test_cross_rank_enabled_with_three_peers_populates_metadata():
    """cross_rank=True with 3 labeled peers → map + scores in metadata."""
    results, metadata, captured_prompts = _run_orchestrator_with_cross_rank(
        cross_rank=True
    )

    # Anonymization map is persisted, deterministic, and complete.
    anon = metadata["anonymization_map"]
    assert anon == {
        "a": "Response A",
        "b": "Response B",
        "c": "Response C",
    }
    # Reverse map also persisted for operator de-anonymization.
    reverse = metadata["anonymization_map_reverse"]
    assert reverse == {"A": "a", "B": "b", "C": "c"}

    # Cross-rank scores populated for at least the peers that got
    # ranked. With our fixture:
    #   a → B(1) C(2)
    #   b → A(1) C(2)
    #   c → B(1) A(2)
    # A is ranked by b(1) and c(2) → mean 1.5
    # B is ranked by a(1) and c(1) → mean 1.0
    # C is ranked by a(2) and b(2) → mean 2.0
    scores = metadata["cross_rank_scores"]
    assert scores["b"] == pytest.approx(1.0)
    assert scores["a"] == pytest.approx(1.5)
    assert scores["c"] == pytest.approx(2.0)

    # cross_rank_complete event fired.
    events = [
        e for e in metadata["progress_events"]
        if e.get("event") == "cross_rank_complete"
    ]
    assert len(events) == 1


def test_cross_rank_results_tagged_is_ranking_round():
    """Ranking-pass results must be tagged is_ranking_round=True."""
    results, metadata, _ = _run_orchestrator_with_cross_rank(cross_rank=True)
    ranking_results = [
        r for r in results if getattr(r, "is_ranking_round", False)
    ]
    assert len(ranking_results) == 3
    # Names use `:rank` suffix to distinguish from `:round2` deliberation.
    assert {r.name for r in ranking_results} == {"a:rank", "b:rank", "c:rank"}


def test_cross_rank_one_ranking_failure_does_not_abort_council():
    """A ranking-pass task that RAISES must not abort the whole council. The
    gather uses return_exceptions=True, so: round-1 results survive, the
    failed ranker is dropped, the surviving rankers still produce scores, and
    a `cross_rank_peer_error` progress event is emitted. Guards the fault path
    of the semaphore+return_exceptions cross-rank fix."""
    import llm_council.orchestrator as orch_module

    participants = ["a", "b", "c"]
    labels = {"a": "yes", "b": "no", "c": "tradeoff"}

    async def fake_run_participants(selected, *args, **kwargs):
        return [_round1_result(name, label=labels[name]) for name in selected]

    async def fake_preflight(*args, **kwargs):
        return {}

    async def fake_run_participant(name, cfg, prompt, cwd, **kwargs):
        if name == "a":
            raise RuntimeError("ranking subprocess blew up")
        return ParticipantResult(
            name=name,
            ok=True,
            output=_ranking_response_for(name),
            error="",
            elapsed_seconds=0.5,
        )

    with patch.object(
        orch_module, "run_participants", side_effect=fake_run_participants
    ), patch.object(
        orch_module, "preflight_local_participants", side_effect=fake_preflight
    ), patch(
        "llm_council.adapters.run_participant", side_effect=fake_run_participant
    ):
        results, metadata = asyncio.run(
            orch_module.execute_council(
                participants=participants,
                participant_cfg={n: {"type": "cli"} for n in participants},
                prompt="orig prompt",
                cwd=Path("."),
                config={"defaults": {}},
                cross_rank=True,
                question="Should we ship the change?",
            )
        )

    # The council did NOT abort: all three primary (round-1) results survive.
    primary = [r for r in results if not getattr(r, "is_ranking_round", False)]
    assert {r.name for r in primary} == {"a", "b", "c"}
    # The raising ranker ('a') is dropped; the two survivors remain.
    rank_rows = [r for r in results if getattr(r, "is_ranking_round", False)]
    assert {r.name for r in rank_rows} == {"b:rank", "c:rank"}
    # Scores are still computed from the surviving rankers.
    assert metadata.get("cross_rank_scores")
    # The new per-failure progress event names the error.
    errs = [
        e
        for e in metadata.get("progress_events", [])
        if e.get("event") == "cross_rank_peer_error"
    ]
    assert len(errs) == 1
    assert "RuntimeError" in errs[0]["error"]


def test_cross_rank_ranking_prompt_anonymizes_peer_names():
    """The ranking prompt sent to peer 'a' should not name 'b' or 'c' directly."""
    results, metadata, captured_prompts = _run_orchestrator_with_cross_rank(
        cross_rank=True
    )
    prompt_for_a = captured_prompts["a"]
    # Per the anonymization map, b -> Response B and c -> Response C.
    assert "Response B" in prompt_for_a
    assert "Response C" in prompt_for_a
    # Other peers' RAW names should not appear as section headers.
    assert "## b" not in prompt_for_a
    assert "## c" not in prompt_for_a
    # Plus a's own output is NOT included.
    assert "reason for a" not in prompt_for_a


def test_cross_rank_does_not_leak_into_round2_deliberation_prompt():
    """Critical MAD-literature guard: ranking outputs must NOT appear in round-2 prompt."""
    # Use deliberate=True, max_rounds=2 so the deliberation builder runs.
    # The round-1 factory returns labels that DISAGREE so deliberation
    # triggers; the cross-rank pass runs in between.
    import llm_council.orchestrator as orch_module
    import llm_council.deliberation as deliberation_mod

    participants = ["a", "b", "c"]
    labels = {"a": "yes", "b": "no", "c": "tradeoff"}

    ranking_marker = "RANKING_PASS_OUTPUT_THAT_MUST_NOT_LEAK"
    round1_marker = "ROUND_ONE_LEGITIMATE_OUTPUT"

    captured_deliberation_inputs = []
    original_build = deliberation_mod.build_deliberation_prompt

    def spy_build(original_prompt, results):
        captured_deliberation_inputs.append(list(results))
        return original_build(original_prompt, results)

    call_count = {"n": 0}

    async def fake_run_participants(selected, *args, **kwargs):
        call_count["n"] += 1
        # Round 1 returns peers with labels including a distinctive
        # marker so we can grep the deliberation prompt for it.
        return [
            ParticipantResult(
                name=name,
                ok=True,
                output=(
                    f"RECOMMENDATION: {labels[name]} - reason "
                    f"{round1_marker}"
                ),
                error="",
                elapsed_seconds=1.0,
            )
            for name in selected
        ]

    async def fake_preflight(*args, **kwargs):
        return {}

    async def fake_run_participant(name, cfg, prompt, cwd, **kwargs):
        # The ranking-pass response carries a distinctive marker —
        # leaking into round-2 deliberation should be detectable.
        return ParticipantResult(
            name=name,
            ok=True,
            output=f"FINAL RANKING: B A C\n# {ranking_marker}",
            error="",
            elapsed_seconds=0.5,
        )

    with patch.object(
        orch_module, "run_participants", side_effect=fake_run_participants
    ), patch.object(
        orch_module,
        "preflight_local_participants",
        side_effect=fake_preflight,
    ), patch(
        "llm_council.adapters.run_participant",
        side_effect=fake_run_participant,
    ), patch.object(
        orch_module, "build_deliberation_prompt", side_effect=spy_build
    ):
        results, metadata = asyncio.run(
            orch_module.execute_council(
                participants=participants,
                participant_cfg={n: {"type": "cli"} for n in participants},
                prompt="orig prompt",
                cwd=Path("."),
                config={"defaults": {}},
                deliberate=True,
                max_rounds=2,
                cross_rank=True,
                question="Should we ship?",
            )
        )

    # The round-2 deliberation prompt must NOT contain the ranking marker.
    assert "deliberation_prompts" in metadata
    for round_no, round_prompt in metadata["deliberation_prompts"].items():
        assert ranking_marker not in round_prompt, (
            f"Round {round_no} deliberation prompt leaked the ranking "
            "marker — round-2 must NOT see ranking-pass outputs."
        )
        # Sanity: round-1 marker SHOULD be present (legitimate content).
        assert round1_marker in round_prompt
    # And the spy that captured the input list to build_deliberation_prompt:
    # NONE of the results passed in should be is_ranking_round=True.
    assert captured_deliberation_inputs, "build_deliberation_prompt was never called"
    for results_seen in captured_deliberation_inputs:
        for result in results_seen:
            assert not getattr(result, "is_ranking_round", False), (
                "Ranking-round result reached the deliberation builder — "
                "the orchestrator's round_results split is broken."
            )


def test_build_deliberation_prompt_filters_ranking_round_results():
    """Belt-and-braces: even if a ranking result reaches the builder, it's filtered."""
    primary = ParticipantResult(
        name="a",
        ok=True,
        output="RECOMMENDATION: yes - primary content marker",
        error="",
        elapsed_seconds=1.0,
    )
    primary_b = ParticipantResult(
        name="b",
        ok=True,
        output="RECOMMENDATION: no - primary B marker",
        error="",
        elapsed_seconds=1.0,
    )
    ranking = ParticipantResult(
        name="a:rank",
        ok=True,
        output="FINAL RANKING: B - RANKING_LEAK_CANARY",
        error="",
        elapsed_seconds=0.5,
        is_ranking_round=True,
    )
    prompt, _ = build_deliberation_prompt(
        "orig", [primary, ranking, primary_b]
    )
    assert "primary content marker" in prompt
    assert "primary B marker" in prompt
    assert "RANKING_LEAK_CANARY" not in prompt


# --- Stats aggregation ---------------------------------------------------

def _write_transcript_with_cross_rank(
    transcripts_dir: Path,
    run_id: str,
    cross_rank_scores: dict[str, float],
):
    """Write a minimal transcript JSON that aggregate_reliability can parse."""
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "question": "Q?",
        "mode": "review",
        "current": None,
        "participants": sorted(cross_rank_scores.keys()),
        "prompt": "p",
        "metadata": {},
        "results": [
            {"name": name, "ok": True, "output": f"RECOMMENDATION: yes - r"}
            for name in cross_rank_scores
        ],
        "cross_rank_scores": cross_rank_scores,
        "anonymization_map": {
            name: f"Response {chr(ord('A') + i)}"
            for i, name in enumerate(sorted(cross_rank_scores.keys()))
        },
    }
    (transcripts_dir / f"{run_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_aggregate_reliability_surfaces_rank_position_mean(tmp_path):
    from llm_council.stats import aggregate_reliability

    runs_dir = tmp_path / ".llm-council" / "runs"
    _write_transcript_with_cross_rank(
        runs_dir, "run1", {"claude": 1.0, "codex": 2.0, "gemini": 3.0}
    )
    _write_transcript_with_cross_rank(
        runs_dir, "run2", {"claude": 2.0, "codex": 1.0, "gemini": 3.0}
    )
    reliability = aggregate_reliability(tmp_path, transcripts_dir=runs_dir)
    by_name = {row["name"]: row for row in reliability["peers"]}
    assert by_name["claude"]["rank_position_mean"] == pytest.approx(1.5)
    assert by_name["codex"]["rank_position_mean"] == pytest.approx(1.5)
    assert by_name["gemini"]["rank_position_mean"] == pytest.approx(3.0)


def test_aggregate_reliability_rank_position_none_when_no_cross_rank(tmp_path):
    """Peer that never participated in --cross-rank gets rank_position_mean=None."""
    from llm_council.stats import aggregate_reliability

    runs_dir = tmp_path / ".llm-council" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    # A transcript with cross_rank_scores for claude only;
    # codex appears but has no rank-position data.
    payload = {
        "question": "Q",
        "mode": "review",
        "current": None,
        "participants": ["claude", "codex"],
        "prompt": "p",
        "metadata": {},
        "results": [
            {
                "name": "claude",
                "ok": True,
                "output": "RECOMMENDATION: yes - r",
                "evidence": [
                    {
                        "tag": "verified",
                        "verified": True,
                        "text": "stub",
                        "path": "a.py",
                        "start_line": 1,
                        "end_line": 2,
                    }
                ],
            },
            {
                "name": "codex",
                "ok": True,
                "output": "RECOMMENDATION: no - r",
                "evidence": [
                    {
                        "tag": "verified",
                        "verified": False,
                        "text": "stub",
                        "path": "b.py",
                        "start_line": 1,
                        "end_line": 2,
                    }
                ],
            },
        ],
        # Only claude in cross_rank_scores.
        "cross_rank_scores": {"claude": 1.5},
    }
    (runs_dir / "run1.json").write_text(json.dumps(payload), encoding="utf-8")

    reliability = aggregate_reliability(tmp_path, transcripts_dir=runs_dir)
    by_name = {row["name"]: row for row in reliability["peers"]}
    assert by_name["claude"]["rank_position_mean"] == pytest.approx(1.5)
    assert by_name["codex"]["rank_position_mean"] is None
