"""Eval runner (Phase B — v0.8 plan).

Loads fixtures (`prompt.md` + `expected_blockers.json`) from a directory,
runs a council against each prompt, and produces JSON scorecards that
feed `llm_council.stats.aggregate_eval_runs`.

The runner accepts the council-call as an injected callable so unit
tests can stub it without touching `llm_council.orchestrator`. Real CLI
wiring uses `functools.partial` to bind the orchestrator's
`execute_council`.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from llm_council import __version__ as _COUNCIL_VERSION
from llm_council.eval.metrics import compute_all

# An execute_council_fn returns a list of ParticipantResult-like objects.
# Kept as `Any` so tests can pass simple stub dataclasses.
ExecuteCouncilFn = Callable[..., Any]


@dataclass
class Fixture:
    """A single eval case loaded from disk."""

    id: str
    prompt: str
    expected_blockers: list[dict[str, Any]]


@dataclass
class PeerScore:
    """Per-peer metrics for a single fixture run.

    Only the fields consumed downstream by `stats.aggregate_eval_runs`
    and `stats._peer_block` are surfaced. `ok` / `error` / `from_cache`
    used to live here but were populated and never read past the
    serialization layer — they have been dropped to keep the scorecard
    JSON minimal. `cache_miss` is still derived per-fixture from raw
    results in `run_fixture`.
    """

    name: str
    blockers: list[str] = field(default_factory=list)
    evidence: list[Any] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FixtureScorecard:
    fixture_id: str
    peers: list[PeerScore] = field(default_factory=list)
    aggregate_metrics: dict[str, Any] = field(default_factory=dict)
    cache_miss: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "peers": [p.to_dict() for p in self.peers],
            "aggregate_metrics": dict(self.aggregate_metrics),
            "cache_miss": self.cache_miss,
        }


@dataclass
class SuiteScorecard:
    """Top-level eval scorecard for a whole fixture suite.

    `aggregate_metrics` carries the suite-level rollup. The key names
    (`blocker_recall`, `signal_to_noise_ratio`, ...) collide with
    per-peer metric names but their **semantics differ by scope** —
    see `_aggregate_suite` for the full rules. Briefly:

    - ``aggregate_metrics["blocker_recall"]`` is the **mean across
      fixtures of each fixture's per-peer MAX recall**, not a raw peer
      value.
    - ``aggregate_metrics["false_blocker_rate" | "signal_to_noise_ratio"
      | "evidence_density"]`` are **mean-of-mean** rollups.
    - ``aggregate_metrics["citation_accuracy"]`` is the same mean-of-
      mean, skipping fixtures where no peer emitted a VERIFIED entry.
    """

    mode: str
    council_version: str
    timestamp: str
    fixtures: list[FixtureScorecard] = field(default_factory=list)
    aggregate_metrics: dict[str, Any] = field(default_factory=dict)
    cache_only: bool = False
    cache_misses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "council_version": self.council_version,
            "timestamp": self.timestamp,
            "cache_only": self.cache_only,
            "cache_misses": list(self.cache_misses),
            "aggregate_metrics": dict(self.aggregate_metrics),
            "fixtures": [f.to_dict() for f in self.fixtures],
        }


# ---------- fixture I/O ----------------------------------------------------


def load_fixture(path: Path) -> Fixture:
    """Read `prompt.md` + `expected_blockers.json` from a fixture directory.

    The fixture's `id` is the directory name. Missing files raise
    `FileNotFoundError`. Malformed JSON raises `ValueError`.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"fixture directory not found: {path}")
    prompt_path = path / "prompt.md"
    expected_path = path / "expected_blockers.json"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"fixture missing prompt.md: {path}")
    if not expected_path.is_file():
        raise FileNotFoundError(f"fixture missing expected_blockers.json: {path}")
    prompt = prompt_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(expected_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"fixture expected_blockers.json malformed at {path}: {exc}") from exc
    blockers = payload.get("blockers") if isinstance(payload, dict) else None
    if not isinstance(blockers, list):
        raise ValueError(
            f"fixture {path} expected_blockers.json must have a top-level "
            f"`blockers: [...]` list"
        )
    return Fixture(id=path.name, prompt=prompt, expected_blockers=blockers)


def iter_fixture_dirs(fixtures_dir: Path) -> Iterable[Path]:
    """Yield direct subdirectories of `fixtures_dir` that contain a
    `prompt.md`. Ignored: dotfiles, hidden dirs, the package's own
    `__pycache__`.
    """
    fixtures_dir = Path(fixtures_dir)
    if not fixtures_dir.is_dir():
        return
    for child in sorted(fixtures_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name == "__pycache__":
            continue
        if (child / "prompt.md").is_file():
            yield child


# ---------- runner ---------------------------------------------------------


def _maybe_await(value: Any) -> Any:
    """If `value` is a coroutine, run it to completion. Otherwise return it.

    Used so test stubs can be either sync or async. The CLI wrapper
    always passes an `asyncio.run(...)`-wrapped sync callable.
    """
    if inspect.iscoroutine(value):
        import asyncio

        return asyncio.run(value)
    return value


def _score_peer(
    result: Any,
    expected_blockers: list[dict[str, Any]],
) -> PeerScore:
    blockers = list(getattr(result, "blockers", []) or [])
    evidence = list(getattr(result, "evidence", []) or [])
    metrics = compute_all(
        emitted_blockers=blockers,
        expected_blockers=expected_blockers,
        evidence_entries=evidence,
    )
    return PeerScore(
        name=str(getattr(result, "name", "")),
        blockers=blockers,
        evidence=evidence,
        metrics=metrics,
    )


def _aggregate_fixture(peers: list[PeerScore]) -> dict[str, Any]:
    """Aggregate peer metrics for one fixture into a single block.

    `blocker_recall` is reported as the max across peers (did ANY peer
    catch the bug?) — emitted as `blocker_recall_max`, which the
    suite-level rollup keys off. `false_blocker_rate`,
    `signal_to_noise_ratio`, and `evidence_density` are averaged across
    peers (per-peer noise floor). `citation_accuracy` is averaged only
    over peers that emitted a VERIFIED entry — None inputs are skipped.
    """
    if not peers:
        return {}
    recalls = [p.metrics.get("blocker_recall", 0.0) for p in peers]
    snrs = [p.metrics.get("signal_to_noise_ratio", 0.0) for p in peers]
    fbrs = [p.metrics.get("false_blocker_rate", 0.0) for p in peers]
    densities = [p.metrics.get("evidence_density", 0.0) for p in peers]
    citations = [
        p.metrics.get("citation_accuracy") for p in peers
        if p.metrics.get("citation_accuracy") is not None
    ]
    return {
        "blocker_recall_max": max(recalls) if recalls else 0.0,
        "false_blocker_rate_mean": sum(fbrs) / len(fbrs) if fbrs else 0.0,
        "signal_to_noise_ratio_mean": sum(snrs) / len(snrs) if snrs else 0.0,
        "evidence_density_mean": sum(densities) / len(densities) if densities else 0.0,
        "citation_accuracy_mean": (
            sum(citations) / len(citations) if citations else None
        ),
        "peer_count": len(peers),
    }


def _aggregate_suite(scorecards: list[FixtureScorecard]) -> dict[str, Any]:
    """Aggregate across fixtures into the suite-level metric block.

    The keys emitted here intentionally drop the per-fixture
    ``_max`` / ``_mean`` suffix even though their meaning differs from
    the per-peer metrics that share the same name. The aggregation
    rules at this level are:

    - ``blocker_recall``: mean across fixtures of each fixture's
      ``blocker_recall_max`` (i.e. "fraction of fixtures where AT LEAST
      ONE peer caught the bug, averaged"). This is **NOT** the raw
      per-peer recall — the per-peer view is in
      ``fixtures[].peers[].metrics.blocker_recall``.
    - ``false_blocker_rate``: mean across fixtures of each fixture's
      ``false_blocker_rate_mean``. Mean-of-mean; one peer being
      especially noisy on one fixture is diluted both within and across.
    - ``signal_to_noise_ratio``: mean across fixtures of each fixture's
      ``signal_to_noise_ratio_mean``. Same mean-of-mean shape.
    - ``evidence_density``: mean-of-mean (peer-level avg evidence per
      finding, averaged across fixtures).
    - ``citation_accuracy``: mean across fixtures of each fixture's
      ``citation_accuracy_mean``, skipping fixtures where no peer
      emitted a VERIFIED entry (the per-fixture mean is ``None`` in
      that case).

    So the **same key name carries a different aggregation rule at
    suite vs. per-peer scope**:
      - per-peer ``metrics.blocker_recall``: raw recall for that peer.
      - per-fixture ``aggregate_metrics.blocker_recall_max``: max
        across peers (for that one fixture).
      - per-suite ``aggregate_metrics.blocker_recall``: **mean across
        fixtures** of the per-fixture max.

    The promotion gate in ``check_promotion_gate`` reads the suite-level
    ``blocker_recall`` and ``signal_to_noise_ratio`` keys. The trend
    aggregator in ``llm_council.stats.aggregate_eval_runs`` reads the
    same suite-level keys. Downstream readers should treat these as
    **aggregate** rollups, never raw peer values.
    """
    if not scorecards:
        return {}
    recalls = [s.aggregate_metrics.get("blocker_recall_max", 0.0) for s in scorecards]
    fbrs = [s.aggregate_metrics.get("false_blocker_rate_mean", 0.0) for s in scorecards]
    snrs = [s.aggregate_metrics.get("signal_to_noise_ratio_mean", 0.0) for s in scorecards]
    densities = [
        s.aggregate_metrics.get("evidence_density_mean", 0.0) for s in scorecards
    ]
    citations = [
        s.aggregate_metrics.get("citation_accuracy_mean") for s in scorecards
        if s.aggregate_metrics.get("citation_accuracy_mean") is not None
    ]
    return {
        "blocker_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "false_blocker_rate": sum(fbrs) / len(fbrs) if fbrs else 0.0,
        "signal_to_noise_ratio": sum(snrs) / len(snrs) if snrs else 0.0,
        "evidence_density": sum(densities) / len(densities) if densities else 0.0,
        "citation_accuracy": (
            sum(citations) / len(citations) if citations else None
        ),
        "fixture_count": len(scorecards),
    }


def run_fixture(
    fixture: Fixture,
    *,
    mode: str,
    execute_council_fn: ExecuteCouncilFn,
    cache_only: bool = False,
) -> FixtureScorecard:
    """Execute one fixture and return its scorecard.

    `execute_council_fn(prompt, mode)` must return an iterable of
    `ParticipantResult`-like objects. Coroutine returns are awaited.
    """
    raw = _maybe_await(execute_council_fn(fixture.prompt, mode))
    results = list(raw or [])
    peers = [_score_peer(r, fixture.expected_blockers) for r in results]
    # `from_cache` was dropped from the per-peer JSON surface, so the
    # cache-miss signal is derived directly from the raw result objects
    # rather than from PeerScore.
    cache_miss = cache_only and any(
        not bool(getattr(r, "from_cache", False)) for r in results
    )
    scorecard = FixtureScorecard(
        fixture_id=fixture.id,
        peers=peers,
        aggregate_metrics=_aggregate_fixture(peers),
        cache_miss=cache_miss,
    )
    return scorecard


def run_suite(
    fixtures_dir: Path,
    *,
    mode: str,
    execute_council_fn: ExecuteCouncilFn,
    cache_only: bool = False,
) -> SuiteScorecard:
    """Iterate over every fixture in `fixtures_dir` and aggregate scorecards."""
    fixtures_dir = Path(fixtures_dir)
    fixture_dirs = list(iter_fixture_dirs(fixtures_dir))
    fixture_scorecards: list[FixtureScorecard] = []
    cache_misses: list[str] = []
    for fdir in fixture_dirs:
        fixture = load_fixture(fdir)
        scorecard = run_fixture(
            fixture,
            mode=mode,
            execute_council_fn=execute_council_fn,
            cache_only=cache_only,
        )
        fixture_scorecards.append(scorecard)
        if scorecard.cache_miss:
            cache_misses.append(scorecard.fixture_id)
    suite = SuiteScorecard(
        mode=mode,
        council_version=_COUNCIL_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        fixtures=fixture_scorecards,
        aggregate_metrics=_aggregate_suite(fixture_scorecards),
        cache_only=cache_only,
        cache_misses=cache_misses,
    )
    return suite


def to_json(scorecard: SuiteScorecard | FixtureScorecard, *, indent: int = 2) -> str:
    """Serialize a scorecard to a JSON string with stable key ordering."""
    return json.dumps(scorecard.to_dict(), indent=indent, sort_keys=False, default=str)


# ---------- promotion gate (Phase E — v0.8 plan) ---------------------------
#
# The `review-with-tools` mode ships `experimental: true` until the eval
# harness demonstrates that directing CLI peers to use their tools (a)
# materially lifts blocker recall and (b) does NOT collapse the signal-
# to-noise ratio. The literature (SWE-PRBench arxiv 2603.26130 + CR-Bench
# SNR) is hostile to "more context" for code review specifically; the
# promotion gate prevents shipping a regression.
#
# Thresholds default to:
#   recall_lift     = 0.05   (≥5pp absolute lift in blocker_recall)
#   snr_floor_ratio = 0.85   (candidate SNR ≥ 85% of baseline SNR)


@dataclass
class PromotionResult:
    """Outcome of `check_promotion_gate`. Promotion is BOTH conditions or nothing.

    `snr_ratio` is `None` when the baseline SNR was 0 (infinite improvement
    over nothing — the candidate "trivially passes" the floor check), so
    it does not collide with `snr_ratio == 0.0` meaning "candidate is
    completely noisy". Downstream consumers should treat `None` as
    "ratio undefined; SNR floor trivially satisfied".
    """

    promoted: bool
    reasons: list[str] = field(default_factory=list)
    recall_delta: float = 0.0
    snr_ratio: float | None = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_promotion_gate(
    baseline: SuiteScorecard,
    candidate: SuiteScorecard,
    *,
    recall_lift: float = 0.05,
    snr_floor_ratio: float = 0.85,
    cross_rank_correlation_floor: float | None = None,
    cross_rank_correlation: float | None = None,
) -> PromotionResult:
    """Compare a baseline scorecard against a candidate scorecard.

    Returns `promoted=True` only when BOTH conditions hold:
      1. ``candidate.blocker_recall - baseline.blocker_recall >= recall_lift``
      2. ``candidate.signal_to_noise_ratio >= baseline.signal_to_noise_ratio
         * snr_floor_ratio``

    The decision is intentionally conservative — either gate failing
    blocks promotion. `reasons` is a human-readable list explaining
    every gate outcome (pass or fail), one entry per criterion.
    """
    a_agg = baseline.aggregate_metrics or {}
    b_agg = candidate.aggregate_metrics or {}

    a_recall = float(a_agg.get("blocker_recall") or 0.0)
    b_recall = float(b_agg.get("blocker_recall") or 0.0)
    recall_delta = b_recall - a_recall

    a_snr = float(a_agg.get("signal_to_noise_ratio") or 0.0)
    b_snr = float(b_agg.get("signal_to_noise_ratio") or 0.0)
    # When the baseline SNR is exactly zero, the ratio is undefined
    # (any non-negative candidate SNR is "infinite improvement"). We
    # surface that as `snr_ratio=None` in the result so it does not
    # collide with `snr_ratio=0.0` (which has the opposite meaning:
    # candidate produced no signal). The gate treats `None` as
    # trivially passing the SNR floor.
    snr_ratio: float | None
    if a_snr > 0:
        snr_ratio = b_snr / a_snr
    else:
        snr_ratio = None

    reasons: list[str] = []
    recall_ok = recall_delta >= recall_lift
    if recall_ok:
        reasons.append(
            f"recall_lift_met: delta={recall_delta:.4f} >= {recall_lift:.4f}"
        )
    else:
        reasons.append(
            f"recall_lift_missing: delta={recall_delta:.4f} < {recall_lift:.4f} "
            f"(baseline={a_recall:.4f}, candidate={b_recall:.4f})"
        )

    # `None` ratio (baseline SNR=0) reads as "infinite improvement"
    # and trivially passes the floor.
    snr_ok = snr_ratio is None or b_snr >= a_snr * snr_floor_ratio
    if snr_ok:
        if snr_ratio is None:
            reasons.append(
                f"snr_floor_met: baseline_snr=0, candidate_snr={b_snr:.4f} "
                "(trivially passes)"
            )
        else:
            reasons.append(
                f"snr_floor_met: ratio={snr_ratio:.4f} >= {snr_floor_ratio:.4f}"
            )
    else:
        # snr_ratio is non-None here (a_snr > 0 → ratio is computable).
        reasons.append(
            f"snr_collapse: ratio={snr_ratio:.4f} < {snr_floor_ratio:.4f} "
            f"(baseline={a_snr:.4f}, candidate={b_snr:.4f})"
        )

    # v0.9.0 Feature 2 (experimental): optional cross-rank correlation
    # gate. When `cross_rank_correlation_floor` is supplied, the caller
    # asserts that the candidate's per-peer `rank_position_mean` must
    # correlate at >= floor with per-peer `useful_count` from outcome
    # data. Failing the gate blocks promotion. The default `None` keeps
    # the gate untouched for callers that didn't opt in. The
    # `cross_rank_correlation` is supplied directly by the caller
    # (computed from `stats.aggregate_reliability` output) rather than
    # re-derived here, to keep this runner pure and side-effect-free.
    cross_rank_ok = True
    if cross_rank_correlation_floor is not None:
        if cross_rank_correlation is None:
            cross_rank_ok = False
            reasons.append(
                "cross_rank_correlation_missing: no correlation supplied "
                f"with floor={cross_rank_correlation_floor:.4f}"
            )
        elif cross_rank_correlation >= cross_rank_correlation_floor:
            reasons.append(
                f"cross_rank_correlation_met: "
                f"r={cross_rank_correlation:.4f} >= "
                f"{cross_rank_correlation_floor:.4f}"
            )
        else:
            cross_rank_ok = False
            reasons.append(
                f"cross_rank_correlation_low: "
                f"r={cross_rank_correlation:.4f} < "
                f"{cross_rank_correlation_floor:.4f}"
            )

    return PromotionResult(
        promoted=recall_ok and snr_ok and cross_rank_ok,
        reasons=reasons,
        recall_delta=recall_delta,
        snr_ratio=snr_ratio,
    )


__all__ = [
    "ExecuteCouncilFn",
    "Fixture",
    "FixtureScorecard",
    "PeerScore",
    "PromotionResult",
    "SuiteScorecard",
    "check_promotion_gate",
    "iter_fixture_dirs",
    "load_fixture",
    "run_fixture",
    "run_suite",
    "to_json",
]
