"""MCP server wrapper for llm-council."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import os
import re
from pathlib import Path
from typing import Any

from llm_council import __version__
from llm_council.budget import (
    DEFAULT_MCP_MAX_PROMPT_CHARS,
    DEFAULT_IMAGE_MAX_BYTES,
    DEFAULT_IMAGE_TOTAL_MAX_BYTES,
    apply_preflight_cost_to_mcp_budget,
    enforce_mcp_budget,
    image_attachment_violations,
    mcp_budget_report,
    summarize_preflight_caps,
)
from llm_council.context import IMAGE_MIME_ALLOWLIST, MAX_CONTEXT_FILES
from llm_council.defaults import DEFAULT_CONFIG
from llm_council.config import (
    apply_tier_override,
    canonical_mode_name,
    config_warnings,
    detect_current_agent,
    find_config,
    load_config,
    resolve_config_data,
    select_participants,
)
from llm_council.context import (
    MAX_PROMPT_CHARS,
    apply_per_peer_directives,
    build_image_manifest,
    build_prompt,
    resolve_acceptance_contract,
)
from llm_council import display
from llm_council.doctor import check_environment, checks_to_dict
from llm_council.env import load_project_env, project_env_context
from llm_council.estimate import (
    DEFAULT_COMPLETION_TOKENS,
    IMAGE_TOKEN_HEURISTIC,
    cross_rank_prompt_char_bounds,
    deliberation_prompt_char_bounds,
    estimate_council,
    synthesis_prompt_char_bound,
)
from llm_council.model_catalog import fetch_openrouter_models
from llm_council.orchestrator import (
    apply_contextual_persona_recruitment,
    execute_council,
)
from llm_council import policy
from llm_council.recommend_judge import grade_difficulty
from llm_council.stats import aggregate_reliability, compute_stats
from llm_council.transcript import (
    continuation_depth_limit_error,
    find_transcript_by_id,
    format_prior_council_context,
    latest_transcript,
    inspect_transcript_permissions,
    normalize_run_id,
    transcript_dir_within_root,
    transcript_paths,
    write_transcript,
)
from llm_council.update_check import check_for_update


def _mode_description() -> str:
    names = ", ".join(sorted(DEFAULT_CONFIG["modes"]))
    return f"Council mode. Built-in choices: {names}."


def _working_directory_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "description": (
            "Absolute project directory for this call. It must exist inside "
            "this server's configured LLM_COUNCIL_MCP_ROOT; omit it to use "
            "that root. Relative paths are refused."
        ),
    }


def council_run_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "question": {"type": "string", "minLength": 1},
            "mode": {
                "type": "string",
                "description": _mode_description(),
            },
            "current": {
                "type": "string",
                "enum": ["claude", "codex", "gemini", "antigravity"],
            },
            "participants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit participants. Overrides mode routing.",
            },
            "include": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra participants to add to mode routing.",
            },
            "context_files": {
                "type": "array",
                "maxItems": MAX_CONTEXT_FILES,
                "items": {"type": "string"},
                "description": "Files to include as read-only context.",
            },
            "image_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Repo-relative paths to image files (PNG/JPEG/WebP/GIF) the host has staged for council review. CLI participants Read them with their own tools.",
            },
            "images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Base64-encoded image bytes."},
                        "mime": {
                            "type": "string",
                            "enum": sorted(IMAGE_MIME_ALLOWLIST),
                        },
                        "name": {"type": "string"},
                    },
                    "required": ["data", "mime"],
                    "additionalProperties": False,
                },
                "description": "Inline base64 images. llm-council writes them under .llm-council/inputs/<run-id>/ before participants run. Use only when the host cannot stage to disk; image_paths is preferred.",
            },
            "include_diff": {"type": "boolean", "default": False},
            "open": {
                "type": "boolean",
                "default": False,
                "description": "Automatically open the HTML transcript in the browser.",
            },
            "working_directory": _working_directory_schema(),
            "chunk_strategy": {
                "type": "string",
                "enum": ["fail", "head", "tail", "hash-aware"],
                "default": "fail",
                "description": (
                    "How to fit an oversized git diff into the MCP prompt "
                    "budget. Any dropped content is returned in chunk metadata."
                ),
            },
            "allow_privacy_downgrade": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Explicitly allow a private-local continuation to send "
                    "prior context to non-local participants."
                ),
            },
            "request_timeout_seconds": {
                "type": "number",
                "minimum": 1,
                "maximum": 7200,
                "description": (
                    "Wall-clock deadline for the complete MCP council request. "
                    "Defaults to defaults.mcp_request_timeout_seconds."
                ),
            },
            "dry_run": {"type": "boolean", "default": False},
            "transparent": {"type": "boolean", "default": False},
            "deliberate": {
                "type": "boolean",
                "default": False,
                "description": "Run an expensive second round if first-round responses disagree.",
            },
            "synthesize": {
                "type": "boolean",
                "default": False,
                "description": (
                    "After peers respond, invoke the configured synthesis "
                    "chair (defaults.synthesizer) to produce a decision "
                    "memo. Chair output is metadata; the headline "
                    "recommendation still comes from peer votes. Requires "
                    "defaults.synthesizer to be set (fails loudly if not)."
                ),
            },
            "require_sections": {
                "type": "boolean",
                "description": (
                    "Override defaults.require_sections (default True). "
                    "When True and the prompt contains `PART N — TITLE "
                    "(REQUIRED)` headers, peer responses must reference "
                    "each required section or fail with "
                    "error_kind=incomplete_response."
                ),
            },
            "strict_evidence": {
                "type": "boolean",
                "description": (
                    "Override defaults.strict_evidence (default False). "
                    "When True, every EVIDENCE bullet must carry a "
                    "[PUBLISHED]/[OBSERVABLE]/[INFERRED]/[SPECULATIVE] "
                    "tag or fail with error_kind=untagged_evidence."
                ),
            },
            "max_rounds": {"type": "integer", "minimum": 1, "maximum": 3},
            "min_quorum": {
                "type": "integer",
                "minimum": 1,
                "description": "Minimum label-producing peers in the final round before the result counts as trustworthy. Default: 2 when 2+ peers are configured, else equal to the peer count. If set higher than the configured peer count, the council will always be reported as degraded.",
            },
            "origin_policy": {
                "type": "string",
                "enum": ["any", "us"],
                "description": "Set to 'us' to allow only US-origin participants.",
            },
            "continuation_id": {
                "type": "string",
                "description": (
                    "Run id (timestamp prefix or filename) of a prior council "
                    "transcript whose summary should be prepended to the new "
                    "prompt. The new transcript records this as parent_run_id."
                ),
            },
            "stances": {
                "type": "object",
                "description": (
                    "Override or extend stance assignment for one or more "
                    "peers — keys are participant names, values are "
                    "for/against/neutral. Adds an ethical-override clause to "
                    "the peer's prompt to attack groupthink. Composes with "
                    "any stances already declared by the mode."
                ),
                "additionalProperties": {
                    "type": "string",
                    "enum": ["for", "against", "neutral"],
                },
            },
            "max_cost_usd": {
                "type": "number",
                "minimum": 0,
                "description": (
                    "Hard ceiling on the council's pre-flight estimated cost "
                    "in USD. If exceeded, the run is refused before any "
                    "subprocess or HTTP call."
                ),
            },
            "max_tokens": {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "Hard ceiling on estimated prompt+completion tokens "
                    "across all participants and budgeted rounds."
                ),
            },
            "cost_warn_usd": {
                "type": "number",
                "minimum": 0,
                "description": (
                    "Soft, advisory-only cost-warning threshold in USD. When "
                    "the pre-flight estimate is at or above this value the run "
                    "still proceeds, but a non-fatal `cost_warning` is surfaced "
                    "top-level and in metadata. Never blocks — use max_cost_usd "
                    "for a hard ceiling. Overrides defaults.cost_warn_usd."
                ),
            },
            "tier": {
                "type": "string",
                "description": (
                    "Swap participant models per `defaults.tiers.<name>` "
                    "in .llm-council.yaml (e.g. `deep` for top thinking "
                    "models, `fast` for budget). Missing peers in the "
                    "tier map keep their default model."
                ),
            },
            "cross_rank": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Opt-in anonymized cross-ranking pass (v0.9.0, "
                    "experimental). After round 1, each peer ranks the "
                    "OTHER peers' responses blindly via a stable "
                    "anonymization map. Aggregates as per-peer mean rank "
                    "position in `cross_rank_scores`. Composes with any "
                    "existing mode; ranking outputs are NEVER fed back "
                    "into round-2 deliberation."
                ),
            },
            "focus": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Review-focus bundle names to compose onto the selected "
                    "mode. Bundles live at "
                    ".llm-council/review-skills/<name>/SKILL.md and are INERT "
                    "prompt text only (advisory, read-only — they grant no "
                    "tool or write capability). They shape WHAT peers "
                    "scrutinize, compose with any mode, and persist across "
                    "rounds. Unknown names fail the call before any peer is "
                    "launched."
                ),
            },
            "acceptance_contract": {
                "type": "string",
                "description": (
                    "Acceptance criteria to anchor the review (advisory-only). "
                    "Either literal text or a path to a file inside the working "
                    "directory. Peers treat a finding as a blocker "
                    "(RECOMMENDATION: no) only when it violates one of the "
                    "numbered criteria; everything else is surfaced as a "
                    "non-blocking concern. Composes with any mode."
                ),
            },
            "independent_review": {
                "type": "boolean",
                "description": (
                    "On a continuation run (continuation_id set), suppress the "
                    "prior council's per-peer labels/rationales so this round "
                    "forms its verdict independently. Advisory: prior_context "
                    "is simply not injected. No effect without a continuation "
                    "or when no prior context was produced. Default False; can "
                    "also be set per-mode or via defaults.independent_review."
                ),
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    }


COUNCIL_RUN_OUTPUT_SCHEMA_VERSION = 8  # v8 = wall time + client/pin eligibility; v7 = model substitutions
COUNCIL_RUN_VALID_STANCES = ("for", "against", "neutral")
COUNCIL_RUN_VALID_ERROR_KINDS = (
    "timeout",
    "context_overflow",
    "prompt_too_large",
    "invalid_response",
    "downstream_error",
    "cli_nonzero_exit",
    "preflight_failed",
    "abdicated",
    "incomplete_response",
    "untagged_evidence",
    "quota_exhausted",
    "model_substituted",
    "pinned_model_unverified",
    "client_ineligible",
    "unknown",
)


def council_run_output_schema() -> dict[str, Any]:
    """JSON schema describing council_run's structured response.

    Advertised so callers can branch on typed fields rather than parsing the
    text content. The shape mirrors what `run_council` returns and is kept
    in sync by the regression tests. Bump COUNCIL_RUN_OUTPUT_SCHEMA_VERSION
    on any breaking shape change.
    """

    return {
        "type": "object",
        "properties": {
            "schema_version": {
                "type": "integer",
                "const": COUNCIL_RUN_OUTPUT_SCHEMA_VERSION,
                "description": (
                    "Output-schema version. Bump when the shape changes in "
                    "a way that downstream consumers must adapt to."
                ),
            },
            "recommendation": {
                "type": "string",
                "enum": ["yes", "no", "tradeoff", "unknown"],
                "description": (
                    "Unique leading label across the final round. `unknown` "
                    "when no peer produced a usable label or the top labels "
                    "are tied."
                ),
            },
            "agreement_count": {
                "type": "integer",
                "description": "Final-round peers that match `recommendation`.",
            },
            "total_labeled": {
                "type": "integer",
                "description": "Final-round peers that produced any label.",
            },
            "degraded": {
                "type": "boolean",
                "description": (
                    "True when fewer than `min_quorum` peers labeled, so the "
                    "headline recommendation should be treated with caution."
                ),
            },
            "rounds": {"type": "integer"},
            "deliberated": {"type": "boolean"},
            "mode": {"type": "string"},
            "current": {"type": ["string", "null"]},
            "participants": {"type": "array", "items": {"type": "string"}},
            "transcript": {
                "type": "string",
                "description": "Filesystem path to the markdown transcript.",
            },
            "json": {
                "type": "string",
                "description": "Filesystem path to the JSON transcript.",
            },
            "html": {
                "type": "string",
                "description": "Filesystem path to the HTML transcript.",
            },
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "ok": {"type": "boolean"},
                        "label": {
                            "type": ["string", "null"],
                            "enum": ["yes", "no", "tradeoff", "unknown", None],
                        },
                        "stance": {
                            "type": ["string", "null"],
                            "enum": [*COUNCIL_RUN_VALID_STANCES, None],
                        },
                        "elapsed_seconds": {"type": "number"},
                        "wall_elapsed_seconds": {"type": ["number", "null"]},
                        "error": {"type": "string"},
                        "error_kind": {
                            "type": ["string", "null"],
                            "enum": [*COUNCIL_RUN_VALID_ERROR_KINDS, None],
                        },
                        "model": {"type": ["string", "null"]},
                        "total_tokens": {"type": ["integer", "null"]},
                        "cost_usd": {"type": ["number", "null"]},
                        "from_cache": {"type": "boolean"},
                        "cache_hit_seconds": {
                            "type": ["number", "null"],
                            "description": (
                                "Wall-clock seconds the cache lookup took on a "
                                "hit. `null` for non-cached runs. `elapsed_seconds` "
                                "always preserves the original run's timing so "
                                "callers can see true cost; this field documents "
                                "actual cache-hit latency for speedup analysis."
                            ),
                        },
                        "recovered_after_launch_retry": {"type": "boolean"},
                        "repair_retry_recovered": {"type": "boolean"},
                        "recovered_after_timeout": {
                            "type": "boolean",
                            "description": (
                                "True when the original call timed out and the "
                                "60s terse-retry recovered with a valid response. "
                                "Distinguished from `recovered_after_launch_retry` "
                                "(launch failure) and `repair_retry_recovered` "
                                "(missing label) so stats can attribute recovery "
                                "to the right mechanism."
                            ),
                        },
                        "model_fallback_used": {
                            "type": ["string", "null"],
                            "description": (
                                "v0.11.6 Phase 2. Next-in-chain model that ran "
                                "after a quota_exhausted error fired and the "
                                "adapter retried with `cfg.fallback_chain`. "
                                "`null` on the common path (no fallback). "
                                "Always `null` for the Claude family because "
                                "Claude's CLI handles overload natively via "
                                "`--fallback-model` and the swap is invisible "
                                "to llm-council."
                            ),
                        },
                        "recovered_after_quota": {
                            "type": "boolean",
                            "description": (
                                "True when the quota-fallback retry succeeded. "
                                "Set in tandem with `model_fallback_used`. A "
                                "fallback that also fails leaves this False "
                                "and `error_kind=quota_exhausted` so the peer "
                                "still drops from quorum."
                            ),
                        },
                        "tool_call_status": {
                            "type": ["string", "null"],
                            "enum": ["absent", "ok", "malformed", None],
                            "description": (
                                "v0.9.0 tool-call voting telemetry. `null` when "
                                "the mode does not enable `tool_call_voting`; "
                                "`absent` when extraction ran and found no "
                                "`record_recommendation` payload (regex "
                                "fallback canonical); `ok` when a structured "
                                "tool call was parsed and used for the "
                                "envelope; `malformed` when a tool-call shape "
                                "was detected but the payload was unparseable "
                                "(regex fallback still ran). Distinct telemetry "
                                "for `absent` vs `malformed` so parser bugs "
                                "are visible instead of silently masked as "
                                "'fallback succeeded'."
                            ),
                        },
                        "terse_retry_attempted": {
                            "type": "boolean",
                            "description": (
                                "True when the peer timed out and a terse-retry "
                                "was attempted (set on both the recovered-success "
                                "and the annotated-failure paths)."
                            ),
                        },
                        "section_repair_attempted": {
                            "type": "boolean",
                            "description": (
                                "True when a `(REQUIRED)` section was missing and "
                                "a section-repair retry was attempted."
                            ),
                        },
                        "is_ranking_round": {
                            "type": "boolean",
                            "description": (
                                "True for `--cross-rank` ranking-pass results, "
                                "which are post-deliberation telemetry and are not "
                                "primary votes."
                            ),
                        },
                        "continue_debate": {
                            "type": ["string", "null"],
                            "enum": ["yes", "no", None],
                            "description": (
                                "Per-peer round-1 vote on whether round-2 "
                                "deliberation is worthwhile; `null` when the peer "
                                "did not emit the optional `CONTINUE_DEBATE:` line."
                            ),
                        },
                        "evidence_verification_failures": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "`path:start-end` references from `[VERIFIED:...]` "
                                "evidence cites that failed mechanical verification. "
                                "The entries are kept (coverage > filtering); this "
                                "list records which ones could not be verified."
                            ),
                        },
                        "prompt_chars": {
                            "type": ["integer", "null"],
                            "description": (
                                "Length of the assembled prompt for this peer. "
                                "Populated on every real adapter call (success or "
                                "failure); `null` only for cache hits and the "
                                "unsupported-type fallback. Lets stats bucket "
                                "timeouts by prompt size without re-parsing the "
                                "error string."
                            ),
                        },
                        "effort": {
                            "type": ["string", "null"],
                            "description": (
                                "Self-reported analysis depth: full|limited|blocked. "
                                "Parsed from the peer's optional response envelope. "
                                "`blocked` without non-empty `blockers` is treated "
                                "as abdication and dropped from quorum."
                            ),
                        },
                        "confidence": {
                            "type": ["string", "null"],
                            "description": "Self-reported confidence: low|medium|high.",
                        },
                        "risk": {
                            "type": ["string", "null"],
                            "description": "Self-reported risk: low|medium|high|critical.",
                        },
                        "blockers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Concrete missing artifacts (file, command output, "
                                "policy doc) that prevented full analysis."
                            ),
                        },
                        "evidence": {
                            "type": "array",
                            "items": {
                                # v3 (v0.7.0): primary shape is `{text, tag}`
                                # produced by `_parse_tagged_entry`. The
                                # string branch is retained so legacy/
                                # external producers (or a future opt-out
                                # of tag parsing) cannot crash strict MCP
                                # clients on output validation. The
                                # ParticipantResult.evidence field is
                                # typed `list[Any]` for the same reason.
                                "oneOf": [
                                    {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"},
                                            "tag": {
                                                "type": ["string", "null"],
                                                "enum": [
                                                    "published",
                                                    "observable",
                                                    "inferred",
                                                    "speculative",
                                                    "verified",
                                                    None,
                                                ],
                                            },
                                            "path": {"type": "string"},
                                            "start_line": {"type": "integer"},
                                            "end_line": {"type": "integer"},
                                            "verified": {"type": ["boolean", "null"]},
                                        },
                                        "required": ["text"],
                                    },
                                    {"type": "string"},
                                ]
                            },
                            "description": (
                                "Structured evidence entries (schema v3). "
                                "Each item is `{text, tag}` where `tag` is one "
                                "of `published|observable|inferred|speculative|"
                                "verified` or `null` for untagged entries. When "
                                "`tag` is `verified` (from a `[VERIFIED:path:start-end]` "
                                "citation), the item also carries `path`, `start_line`, "
                                "`end_line`, and `verified` (mechanical-check result, "
                                "may be null pre-verification). Plain strings are also "
                                "accepted for legacy/external producers. `text` carries "
                                "the path:line or section reference."
                            ),
                        },
                        "tests_to_run": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Verification commands the peer suggests.",
                        },
                        "assumptions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Stated assumptions underpinning the answer.",
                        },
                    },
                    "required": ["name", "ok", "label"],
                },
            },
            "consensus_blockers": {
                "type": "array",
                "description": (
                    "Per-finding agreement matrix (Phase F). Findings >=2 peers "
                    "anchored to overlapping `[VERIFIED:path:start-end]` ranges. "
                    "Omitted entirely (along with `single_peer_concerns`) when "
                    "no peer emitted FINDINGS or no cluster met the consensus "
                    "threshold. Mechanical clustering only — no fuzzy prose "
                    "match. Surfaces for synthesis input only; peers in round "
                    "2 never see this."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["blocker", "medium", "nit"],
                        },
                        "peers": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "claim": {"type": "string"},
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                    },
                    "required": ["id", "severity", "peers", "claim"],
                },
            },
            "single_peer_concerns": {
                "type": "array",
                "description": (
                    "Findings only one peer raised — either no verified ref "
                    "or no overlapping peer ref. Worth surfacing but not "
                    "consensus-grade. Omitted entirely (along with "
                    "`consensus_blockers`) when no peer emitted FINDINGS."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "peer": {"type": "string"},
                        "id": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["blocker", "medium", "nit"],
                        },
                        "claim": {"type": "string"},
                        "path": {"type": ["string", "null"]},
                        "start_line": {"type": ["integer", "null"]},
                        "end_line": {"type": ["integer", "null"]},
                        "unverified": {"type": "boolean"},
                    },
                    "required": ["peer", "severity", "claim"],
                },
            },
            "cross_rank_scores": {
                "type": "object",
                "description": (
                    "v0.9.0 Feature 2 (experimental). Per-peer mean rank "
                    "position from the opt-in anonymized cross-ranking "
                    "pass. Keys are peer names; values are floats where "
                    "1.0 = unanimously ranked first (lower is better). "
                    "Omitted entirely when `--cross-rank` was not set or "
                    "fewer than 2 peers produced labeled responses."
                ),
                "additionalProperties": {"type": "number"},
            },
            "anonymization_map": {
                "type": "object",
                "description": (
                    "v0.9.0 Feature 2 (experimental). Stable map from "
                    "peer name to anonymization label (`Response A`, "
                    "`Response B`, ...) used by the cross-ranking pass. "
                    "Persisted so operators reading the transcript can "
                    "de-anonymize the rank-position scores. Omitted when "
                    "`--cross-rank` was not set."
                ),
                "additionalProperties": {"type": "string"},
            },
            "quota_throttled_peers": {
                "type": "array",
                "description": (
                    "Peers that hit a known quota/rate-limit error during "
                    "this run (`error_kind=quota_exhausted`) AND did NOT "
                    "recover via fallback. Surfaced top-level so the "
                    "calling agent can identify rate-limited peers "
                    "without parsing per-result `error` strings. Omitted "
                    "entirely when no peer was throttled (the common "
                    "case). Peers that recovered via `fallback_chain` "
                    "appear in `quota_recoveries` instead."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "peer": {"type": "string"},
                        "family": {"type": "string"},
                        "model": {"type": ["string", "null"]},
                        "message": {"type": "string"},
                    },
                    "required": ["peer", "family", "message"],
                },
            },
            "quota_recoveries": {
                "type": "array",
                "description": (
                    "Phase 2: peers that hit quota but recovered via "
                    "`cfg.fallback_chain` retry. Each entry names the "
                    "fallback model that succeeded. Omitted entirely "
                    "when no recovery fired. Disjoint from "
                    "`quota_throttled_peers` (a peer is in exactly one "
                    "list per run, keyed on its final state)."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "peer": {"type": "string"},
                        "family": {"type": "string"},
                        "fallback_model": {"type": ["string", "null"]},
                        "model": {"type": ["string", "null"]},
                    },
                    "required": ["peer", "family"],
                },
            },
            "missing_key_peers": {
                "type": "array",
                "description": (
                    "v0.12.0: hosted peers (openrouter / openai_compatible) "
                    "dropped from the run because their `api_key_env` env "
                    "var was unset. Excluded from the quorum denominator "
                    "BEFORE the run starts, so a missing key on one peer "
                    "doesn't flip the whole run to `degraded`. Omitted "
                    "entirely when all configured keys are present."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "peer": {"type": "string"},
                        "family": {"type": "string"},
                        "api_key_env": {"type": "string"},
                    },
                    "required": ["peer", "family", "api_key_env"],
                },
            },
            "model_substituted_peers": {
                "type": "array",
                "description": (
                    "v0.16.0: peers with `require_pinned_model: true` whose "
                    "turn was served by a model other than the pinned one "
                    "(`error_kind=model_substituted`) — e.g. Claude Fable 5 "
                    "refused and the Claude Code surface silently fell back "
                    "to Opus. The peer's answer was dropped so the "
                    "substituted model is never counted as the pinned "
                    "model's vote. `served_by` is the model the CLI "
                    "actually reported; `ranking_round: true` marks a swap "
                    "during the --cross-rank ranking pass. Omitted entirely "
                    "when no substitution was detected (the common case)."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "peer": {"type": "string"},
                        "requested": {"type": ["string", "null"]},
                        "served_by": {"type": ["string", "null"]},
                        "ranking_round": {"type": "boolean"},
                    },
                    "required": ["peer", "served_by"],
                },
            },
            "applied_focus": {
                "type": "array",
                "description": (
                    "Operator-authored review-focus bundles applied to this "
                    "run (M11 provenance). Each entry is the bundle name + "
                    "the hex sha256 of its (inert, advisory-only) directive "
                    "body. Bundles compose with any mode and grant no tool "
                    "or write capability. Omitted entirely when no focus "
                    "was applied (the common no-focus path)."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "sha256": {"type": "string"},
                    },
                    "required": ["name", "sha256"],
                },
            },
            "metadata": {"type": "object"},
            "summary_markdown": {
                "type": "string",
                "description": (
                    "Markdown payload designed to survive host-agent rendering: "
                    "`**Council**` heading + per-peer table + blockquoted "
                    "transcript path. Agents that quote from tool output "
                    "preserve markdown blockquotes/tables verbatim, giving "
                    "users a visual anchor for council-sourced content."
                ),
            },
        },
        "required": [
            "schema_version",
            "recommendation",
            "agreement_count",
            "total_labeled",
            "degraded",
            "rounds",
            "participants",
            "results",
        ],
    }


def last_transcript_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "working_directory": _working_directory_schema(),
            "format": {
                "type": "string",
                "enum": ["markdown", "json"],
                "default": "markdown",
            },
        },
        "additionalProperties": False,
    }


def recommend_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task": {"type": "string", "minLength": 1},
            "failed_attempts": {"type": "integer", "minimum": 0, "default": 0},
            "files_touched": {"type": "integer", "minimum": 0, "default": 0},
            "risk": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "default": "medium",
            },
            "working_directory": _working_directory_schema(),
        },
        "required": ["task"],
        "additionalProperties": False,
    }


def estimate_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "question": {"type": "string", "minLength": 1},
            "mode": {
                "type": "string",
                "description": _mode_description(),
            },
            "current": {
                "type": "string",
                "enum": ["claude", "codex", "gemini", "antigravity"],
            },
            "participants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit participants. Overrides mode routing.",
            },
            "include": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra participants to add to mode routing.",
            },
            "context_files": {
                "type": "array",
                "maxItems": MAX_CONTEXT_FILES,
                "items": {"type": "string"},
                "description": "Files to include as read-only context.",
            },
            "image_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Repo-relative paths to image files (PNG/JPEG/WebP/GIF). Counted against prompt-size guard as text references only.",
            },
            "images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"},
                        "mime": {"type": "string", "enum": sorted(IMAGE_MIME_ALLOWLIST)},
                        "name": {"type": "string"},
                    },
                    "required": ["data", "mime"],
                    "additionalProperties": False,
                },
                "description": "Inline base64 images. Estimate stages them to .llm-council/inputs/<run-id>/ before computing prompt size.",
            },
            "include_diff": {"type": "boolean", "default": False},
            "working_directory": _working_directory_schema(),
            "chunk_strategy": {
                "type": "string",
                "enum": ["fail", "head", "tail", "hash-aware"],
                "default": "fail",
            },
            "deliberate": {"type": "boolean", "default": False},
            "max_rounds": {"type": "integer", "minimum": 1, "maximum": 3},
            "completion_tokens": {
                "type": "integer",
                "minimum": 0,
                "default": 1500,
                "description": "Assumed output tokens per participant per round.",
            },
            "openrouter_models": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra OpenRouter model IDs to price without editing config.",
            },
            "origin_policy": {
                "type": "string",
                "enum": ["any", "us"],
                "description": "Set to 'us' to allow only US-origin participants.",
            },
            "no_cache": {"type": "boolean", "default": False},
            "tier": {
                "type": "string",
                "description": (
                    "Swap participant models per `defaults.tiers.<name>` in "
                    ".llm-council.yaml before estimating, so the per-peer "
                    "cost reflects the tier you'd actually run."
                ),
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    }


def doctor_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "working_directory": _working_directory_schema(),
            "probe_openrouter": {"type": "boolean", "default": False},
            "probe_ollama": {"type": "boolean", "default": False},
            "probe_native": {"type": "boolean", "default": False},
            "repair_transcript_permissions": {"type": "boolean", "default": False},
            "check_update": {"type": "boolean", "default": False},
        },
        "additionalProperties": False,
    }


def stats_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "working_directory": _working_directory_schema(),
            "since_days": {
                "type": "integer",
                "minimum": 1,
                "description": "Only consider transcripts within the last N days.",
            },
            "participant": {
                "type": "string",
                "description": "Filter participant metrics to one participant.",
            },
        },
        "additionalProperties": False,
    }


def models_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "filter": {"type": "string"},
            "origin": {"type": "string", "enum": ["us", "china", "unknown"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            "no_cache": {"type": "boolean", "default": False},
        },
        "additionalProperties": False,
    }


def query_transcripts_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 5,
            },
            "working_directory": _working_directory_schema(),
        },
        "required": ["query"],
        "additionalProperties": False,
    }


def _build_mcp_progress_callback(
    session: Any,
    progress_token: Any,
    *,
    planned_total: float,
):
    """Build a sync `progress` callback that emits MCP progress notifications.

    The orchestrator's `emit` is sync; `session.send_progress_notification`
    is async. We bridge via `asyncio.create_task(...)` so we never block
    round progression on a slow client, and we swallow transport errors so
    a wedged client cannot fail the council run.

    Returns a callable suitable for `execute_council(..., progress=cb)`,
    or `None` when no token is set / quiet mode is active (caller falls
    back to no progress callback).

    The event loop only holds **weak** references to tasks created via
    `asyncio.create_task`, so without a strong reference the GC can
    collect a task mid-flight and the notification silently disappears
    (CPython `asyncio` docs warning). We keep a closure-local set of
    pending tasks and discard each on completion — references live just
    long enough for `_send` to finish.
    """
    if progress_token is None or session is None:
        return None
    if display.wants_quiet():
        return None

    counter = {"value": 0.0}
    _pending_tasks: set[asyncio.Task[Any]] = set()

    async def _send(progress: float, message: str) -> None:
        try:
            await session.send_progress_notification(
                progress_token,
                progress,
                total=planned_total,
                message=message,
            )
        except Exception:
            # Transport errors must never break the council run. The
            # final tool-result envelope still ships every event in
            # metadata.progress_events, so nothing is permanently lost.
            pass

    def callback(event: dict[str, Any]) -> None:
        kind = event.get("event")
        # Counter logic runs FIRST so suppressed-message events still
        # advance progress where appropriate. `preflight_failed` peers
        # never emit `participant_finish` (they're stripped from
        # `run_targets`), so without this they'd never tick the counter
        # and the bar would stall until `council_finish` clamps.
        if kind in display.PROGRESS_ADVANCING_EVENTS:
            counter["value"] += 1
        elif kind == "council_finish":
            counter["value"] = planned_total
        message = display.format_progress_message(event)
        if message is None:
            return
        try:
            task = asyncio.create_task(_send(counter["value"], message))
        except RuntimeError:
            # No running loop (sync test contexts). Drop silently —
            # progress notifications are advisory.
            return
        # Hold a strong ref until `_send` finishes; without this the
        # event loop's weak-ref policy lets the task get GC'd mid-await
        # and the notification vanishes silently.
        _pending_tasks.add(task)
        task.add_done_callback(_pending_tasks.discard)

    return callback


async def run_council(
    arguments: dict[str, Any],
    *,
    mcp_session: Any | None = None,
    progress_token: Any | None = None,
) -> dict[str, Any]:
    cwd = _resolve_working_directory(arguments)
    # MCP servers are long-lived and may serve sibling projects concurrently.
    # Keep dotenv values in a ContextVar overlay for the entire async request;
    # adapters/tasks inherit it without mutating or racing over os.environ.
    with project_env_context(cwd, stop_at=_mcp_root()):
        config = load_config(find_config(cwd, stop_at=_mcp_root()), search=False)
        configured_timeout = (
            config.get("defaults", {}).get("mcp_request_timeout_seconds")
            or 1200
        )
        request_timeout = float(
            arguments.get("request_timeout_seconds") or configured_timeout
        )
        if request_timeout <= 0 or request_timeout > 7200:
            raise ValueError(
                "request_timeout_seconds must be greater than 0 and no more "
                "than 7200"
            )
        try:
            async with asyncio.timeout(request_timeout):
                return await _run_council_scoped(
                    arguments,
                    cwd=cwd,
                    config=config,
                    mcp_session=mcp_session,
                    progress_token=progress_token,
                )
        except TimeoutError as exc:
            raise ValueError(
                "CouncilRequestTimeout: the complete MCP council request "
                f"exceeded {request_timeout:g}s. Raise "
                "request_timeout_seconds/defaults.mcp_request_timeout_seconds, "
                "choose fewer participants, or reduce rounds/context."
            ) from exc


async def _run_council_scoped(
    arguments: dict[str, Any],
    *,
    cwd: Path,
    config: dict[str, Any],
    mcp_session: Any | None = None,
    progress_token: Any | None = None,
) -> dict[str, Any]:
    # Retain the call for compatibility with existing instrumentation. Inside
    # project_env_context it is intentionally a no-op (see env.py).
    load_project_env(cwd, stop_at=_mcp_root())
    warnings = config_warnings(config)

    question = arguments["question"]
    transcripts_root = transcript_dir_within_root(cwd, config, root=_mcp_root())
    parent_run_id: str | None = None
    prior_context: str | None = None
    prior_transcript: dict[str, Any] | None = None
    continuation_id = arguments.get("continuation_id")
    if continuation_id:
        normalize_run_id(continuation_id)
        _depth_err = continuation_depth_limit_error(
            config, transcripts_root, continuation_id
        )
        if _depth_err:
            raise ValueError(_depth_err)
        prior_transcript = find_transcript_by_id(transcripts_root, continuation_id)
        prior_path = prior_transcript.get("_path")
        parent_run_id = (
            Path(str(prior_path)).stem
            if prior_path
            else normalize_run_id(continuation_id)
        )

    inherited_mode: str | None = None
    if arguments.get("mode") is None and prior_transcript is not None:
        from llm_council.privacy import transcript_was_private_local

        if transcript_was_private_local(prior_transcript):
            inherited_mode = str(prior_transcript.get("mode") or "private-local")
    mode = canonical_mode_name(
        config,
        arguments.get("mode")
        or inherited_mode
        or config.get("defaults", {}).get("mode", "quick"),
    )
    from llm_council.config import apply_smart_routing

    apply_smart_routing(config, mode, cwd)
    # An explicit tier is an operator pin and therefore has higher precedence
    # than automatic low-risk routing. Applying it last prevents `tier: deep`
    # from being silently stepped down to a cheaper model.
    tier = arguments.get("tier")
    if tier:
        apply_tier_override(config, str(tier))
    current = arguments.get("current") or detect_current_agent()
    participants = select_participants(
        config,
        mode,
        current,
        explicit=arguments.get("participants"),
        include=arguments.get("include"),
        origin_policy=arguments.get("origin_policy"),
    )

    if prior_transcript is not None:
        from llm_council.privacy import privacy_downgrade_error

        downgrade_error = privacy_downgrade_error(
            prior_transcript,
            participants=participants,
            participant_cfg=config.get("participants", {}) or {},
            allow_privacy_downgrade=bool(
                arguments.get("allow_privacy_downgrade")
            ),
        )
        if downgrade_error:
            raise ValueError(downgrade_error)
        prior_context = format_prior_council_context(
            prior_transcript, run_id=parent_run_id
        )

    # Transcript identifiers are opaque; never place user question text in a
    # filesystem path, even before secret scanning runs.
    md_path, json_path = transcript_paths(transcripts_root, "")
    sweep_old_inline_inputs(cwd)
    inline_staged = _stage_inline_images(arguments.get("images"), cwd, md_path.stem)
    image_path_inputs = list(arguments.get("image_paths") or []) + inline_staged
    image_manifest = (
        build_image_manifest(image_path_inputs, cwd=cwd, allow_outside_cwd=False)
        if image_path_inputs
        else []
    )
    image_violations = image_attachment_violations(image_manifest)
    if image_violations:
        raise ValueError(
            "Image attachment budget exceeded: "
            + ", ".join(
                f"{item['limit']} {item.get('actual')} > {item.get('maximum')}"
                for item in image_violations
            )
        )
    participant_cfg_for_prompt = config.get("participants", {})
    if not isinstance(participant_cfg_for_prompt, dict):
        participant_cfg_for_prompt = {}
    mode_cfg = config.get("modes", {}).get(mode, {})
    if not isinstance(mode_cfg, dict):
        mode_cfg = {}
    # Independent-review isolation (advisory). Resolution order:
    # MCP arg > per-mode independent_review > defaults.independent_review.
    # Default OFF. When ON and a prior_context WOULD have been injected from a
    # continuation, drop it so this round forms its verdict without anchoring
    # on prior verdicts. Surfaced post-run as a metadata flag (and top-level).
    # None-aware precedence (NOT an `or` chain): an explicit per-call
    # `independent_review: false` must override a true mode/default, and a
    # mode's explicit false must override a true global default. Walk
    # highest-priority first, taking the first non-None layer. (codex WU5
    # review.)
    _iv = arguments.get("independent_review")
    if _iv is None:
        _iv = mode_cfg.get("independent_review")
    if _iv is None:
        _iv = config.get("defaults", {}).get("independent_review")
    independent_review = bool(_iv)
    prior_context_suppressed = False
    if independent_review and prior_context:
        prior_context = None
        prior_context_suppressed = True
    # Acceptance contract (advisory). Resolve <text|path>: read the file only
    # when the value names an existing regular file inside cwd; otherwise treat
    # it as literal contract text. A failed in-cwd path check raises.
    acceptance_contract = resolve_acceptance_contract(
        arguments.get("acceptance_contract"), cwd=cwd, allow_outside_cwd=False
    )
    mode_stances = mode_cfg.get("stances")
    arg_stances = arguments.get("stances")
    if isinstance(arg_stances, dict) and arg_stances:
        base = dict(mode_stances) if isinstance(mode_stances, dict) else {}
        for peer, stance in arg_stances.items():
            if not isinstance(peer, str) or not isinstance(stance, str):
                continue
            base[peer] = stance.lower()
        mode_stances = base
    from llm_council.config import balance_stances
    mode_stances = balance_stances(participants, mode_stances)
    default_max = (
        config.get("defaults", {}).get("max_prompt_chars") or MAX_PROMPT_CHARS
    )
    mcp_max = (
        config.get("defaults", {}).get("mcp_max_prompt_chars")
        or DEFAULT_MCP_MAX_PROMPT_CHARS
    )
    peer_caps = [
        int(participant_cfg_for_prompt.get(name, {}).get("max_prompt_chars"))
        for name in participants
        if isinstance(participant_cfg_for_prompt.get(name), dict)
        and participant_cfg_for_prompt.get(name, {}).get("max_prompt_chars")
    ]
    effective_max = min([int(default_max), int(mcp_max), *peer_caps])
    safe_context = bool(
        (config.get("modes", {}) or {}).get(mode, {}).get("safe_context")
    )
    chunk_events: list[dict[str, Any]] = []
    prompt = build_prompt(
        question,
        mode=mode,
        cwd=cwd,
        context_paths=arguments.get("context_files") or [],
        include_diff=bool(arguments.get("include_diff")),
        stdin_text=None,
        allow_outside_cwd=False,
        max_prompt_chars=effective_max,
        image_manifest=image_manifest or None,
        stances=mode_stances if isinstance(mode_stances, dict) else None,
        participants=participant_cfg_for_prompt or None,
        prior_context=prior_context,
        acceptance_contract=acceptance_contract,
        safe_context=safe_context,
        chunk_strategy=str(arguments.get("chunk_strategy") or "fail"),
        chunk_progress=chunk_events.append,
    )
    from llm_council.safety import apply_secret_scan_policy, redact_secrets

    _defaults_cfg = config.get("defaults", {}) or {}
    _scan_policy = str(_defaults_cfg.get("secret_scan") or "warn").lower()
    _scan_allowlist = str(
        _defaults_cfg.get("secret_scan_allowlist")
        or ".llm-council-secrets-allow"
    )
    secret_scan_payload = apply_secret_scan_policy(
        prompt,
        policy=_scan_policy,
        cwd=cwd,
        allowlist_filename=_scan_allowlist,
    )
    # redact policy: swap in the masked prompt before it reaches peers OR the
    # transcript. Pop it so the redacted prompt isn't duplicated into the
    # metadata.secret_scan payload below.
    _redacted_prompt = secret_scan_payload.pop("redacted_prompt", None)
    persisted_question = question
    if _redacted_prompt is not None:
        prompt = _redacted_prompt
        persisted_question, _question_findings = redact_secrets(
            question,
            cwd=cwd,
            allowlist_filename=_scan_allowlist,
        )
    transparent = bool(
        arguments.get("transparent") or config.get("defaults", {}).get("transparent")
    )
    deliberate = bool(arguments.get("deliberate") or mode_cfg.get("deliberate"))
    synthesize = bool(
        arguments.get("synthesize")
        or mode_cfg.get("synthesize")
        or config.get("defaults", {}).get("synthesize")
    )
    # MCP per-call toggles for v0.7 validators. None means "use config";
    # an explicit bool overrides config["defaults"] for this run.
    _mcp_require_sections = arguments.get("require_sections")
    _mcp_strict_evidence = arguments.get("strict_evidence")
    if _mcp_require_sections is not None:
        config.setdefault("defaults", {})["require_sections"] = bool(_mcp_require_sections)
    if _mcp_strict_evidence is not None:
        config.setdefault("defaults", {})["strict_evidence"] = bool(_mcp_strict_evidence)
    max_rounds = int(
        arguments.get("max_rounds")
        or mode_cfg.get("max_rounds")
        or config.get("defaults", {}).get("max_deliberation_rounds")
        or 2
    )
    min_quorum_arg = arguments.get("min_quorum")
    if min_quorum_arg is None:
        min_quorum_arg = mode_cfg.get("min_quorum")
    min_quorum_value = int(min_quorum_arg) if min_quorum_arg is not None else None

    # Resolve review-focus bundles before estimating so their per-peer framing
    # is included in hard token/cost caps as well as the actual calls.
    resolved_focus = None
    focus_directive = ""
    _focus_arg = arguments.get("focus")
    if _focus_arg:
        from llm_council import review_skills as _review_skills

        _focus_names = [str(name) for name in _focus_arg]
        try:
            resolved_focus, _ = _review_skills.resolve_focus(_focus_names, cwd)
        except _review_skills.FocusNotFound as exc:
            raise ValueError(str(exc)) from exc
        focus_directive = _review_skills.render_focus_directive(resolved_focus)

    tool_call_voting = bool(mode_cfg.get("tool_call_voting"))
    apply_contextual_persona_recruitment(
        participants,
        participant_cfg_for_prompt,
        cwd,
        stances=mode_stances if isinstance(mode_stances, dict) else None,
    )

    resolved_synthesizer: str | None = None
    if synthesize:
        from llm_council.synthesis import select_synthesizer

        selected_participant_cfg = {
            name: participant_cfg_for_prompt[name]
            for name in participants
            if name in participant_cfg_for_prompt
        }
        resolved_synthesizer = select_synthesizer(
            config,
            selected_participant_cfg,
            stances=mode_stances if isinstance(mode_stances, dict) else None,
            current=current,
        )

    participant_prompts: dict[str, str] = {}
    for name in participants:
        peer_cfg = participant_cfg_for_prompt.get(name) or {}
        assigned_stance = (
            mode_stances.get(name)
            if isinstance(mode_stances, dict)
            else peer_cfg.get("stance")
        )
        participant_prompts[name] = apply_per_peer_directives(
            prompt,
            mode=mode,
            family=peer_cfg.get("family"),
            tool_call_voting=tool_call_voting,
            stance=assigned_stance,
            persona=peer_cfg.get("persona"),
            persona_prompt=peer_cfg.get("persona_prompt"),
            focus_directive=focus_directive,
        )
    budget_prompt_chars = max(
        [len(prompt), *(len(text) for text in participant_prompts.values())]
    )
    participant_prompt_char_counts = {
        name: len(text) for name, text in participant_prompts.items()
    }
    deliberation_prompt_bounds = (
        deliberation_prompt_char_bounds(
            participants=participants,
            participant_cfg=participant_cfg_for_prompt,
            mode=mode,
            tool_call_voting=tool_call_voting,
            stances=mode_stances if isinstance(mode_stances, dict) else None,
            focus_directive=focus_directive,
        )
        if deliberate and max_rounds > 1
        else {}
    )
    image_prompt_tokens = {
        name: len(image_manifest) * IMAGE_TOKEN_HEURISTIC
        if (participant_cfg_for_prompt.get(name) or {}).get("vision")
        else 0
        for name in participants
    }
    cross_rank_enabled = bool(arguments.get("cross_rank"))
    ranking_prompt_bounds = (
        cross_rank_prompt_char_bounds(
            participants=participants,
            participant_cfg=participant_cfg_for_prompt,
            question=persisted_question or prompt,
            completion_tokens=DEFAULT_COMPLETION_TOKENS,
            focus_directive=focus_directive,
            safe_context=safe_context,
        )
        if cross_rank_enabled and len(participants) >= 2
        else {}
    )
    synthesis_prompt_bound = (
        synthesis_prompt_char_bound(
            participant_cfg_for_prompt.get(resolved_synthesizer, {})
        )
        if resolved_synthesizer is not None
        else None
    )
    budget = mcp_budget_report(
        config=config,
        participants=participants,
        prompt_chars=budget_prompt_chars,
        deliberate=deliberate,
        max_rounds=max_rounds,
        cross_rank=cross_rank_enabled,
        synthesize=synthesize,
        synthesizer_name=resolved_synthesizer,
        participant_prompt_chars=participant_prompt_char_counts,
        deliberation_prompt_chars=deliberation_prompt_bounds,
        cross_rank_prompt_chars=ranking_prompt_bounds,
        synthesis_prompt_chars=synthesis_prompt_bound,
        image_prompt_tokens=image_prompt_tokens,
    )

    max_cost_usd = arguments.get("max_cost_usd")
    max_tokens = arguments.get("max_tokens")
    # M6 soft cost-warning threshold. None-aware precedence (NOT an `or`
    # chain, so an explicit 0 at either layer is respected): MCP arg >
    # defaults.cost_warn_usd.
    cost_warn_usd = arguments.get("cost_warn_usd")
    if cost_warn_usd is None:
        cost_warn_usd = config.get("defaults", {}).get("cost_warn_usd")
    # Compute the pre-flight estimate once and reuse it for the hard budget
    # gate (when a cap is set), the M6 soft warning, and the L7 compact echo.
    # allow_network=False keeps it cheap (cached catalog only) so the
    # always-on L7 echo adds no meaningful latency.
    preflight: dict[str, Any] | None = None
    cost_warning_payload: dict[str, Any] | None = None
    cost_estimate_block: dict[str, Any] | None = None
    try:
        preflight = estimate_council(
            config=config,
            cwd=cwd,
            question=persisted_question,
            mode=mode,
            current=current,
            explicit=arguments.get("participants"),
            include=arguments.get("include"),
            origin_policy=arguments.get("origin_policy"),
            context_paths=arguments.get("context_files") or [],
            include_diff=bool(arguments.get("include_diff")),
            allow_outside_cwd=False,
            deliberate=deliberate,
            max_rounds=max_rounds,
            use_cache=True,
            allow_network=False,
            image_paths=image_path_inputs or None,
            prepared_prompt=prompt,
            prepared_participants=participants,
            participant_prompts=participant_prompts,
            cross_rank=cross_rank_enabled,
            synthesize=synthesize,
            synthesizer_name=resolved_synthesizer,
            focus_directive=focus_directive,
            cross_rank_prompt_chars=ranking_prompt_bounds,
            synthesis_prompt_chars=synthesis_prompt_bound,
            deliberation_prompt_chars=deliberation_prompt_bounds,
        )
    except (OSError, ValueError) as exc:
        # Preserve any cheap prompt/input violation first. Otherwise explicit
        # caps and the default hosted-run ceiling fail closed when the richer
        # whole-run estimate (including outputs) cannot be computed.
        enforce_mcp_budget(budget)
        if (
            max_cost_usd is not None
            or max_tokens is not None
            or budget.get("paid_hosted_participants")
        ):
            raise ValueError(
                f"failed to compute pre-flight estimate: {exc}"
            ) from exc
    if preflight is not None:
        from llm_council.estimate import compact_cost_estimate

        apply_preflight_cost_to_mcp_budget(budget, preflight)
        cost_estimate_block = compact_cost_estimate(preflight)
        cost_total, token_total, unpriced_paid = summarize_preflight_caps(preflight)
        if max_cost_usd is not None and unpriced_paid:
            raise ValueError(
                "Pre-flight estimate cannot enforce max_cost_usd: non-local "
                f"peer(s) without complete pricing: {', '.join(unpriced_paid)}. "
                "Configure input_per_million and output_per_million (or, for "
                "OpenRouter, confirm the model id against `council_models`), "
                "or drop these peers before relying on the cost cap."
            )
        if max_cost_usd is not None and cost_total > float(max_cost_usd):
            raise ValueError(
                f"Pre-flight estimate ${cost_total:.6f} (with worst-case "
                f"outer-retry headroom) exceeds max_cost_usd "
                f"${float(max_cost_usd):.6f}; refused before any participant "
                "was invoked."
            )
        if max_tokens is not None and token_total > int(max_tokens):
            raise ValueError(
                f"Pre-flight estimate {token_total} tokens exceeds max_tokens "
                f"{int(max_tokens)}; refused before any participant was invoked."
            )
        # M6: soft warning off the SAME reduction the hard gate uses, so soft
        # and hard numbers can never drift. Only reached when the run proceeds
        # (a tripped hard gate above raises before this point).
        if cost_warn_usd is not None and cost_total >= float(cost_warn_usd):
            cost_warning_payload = {
                "estimated_usd": cost_total,
                "threshold_usd": float(cost_warn_usd),
            }
    # Both real and dry runs honor the default MCP ceiling. Dry-run means
    # "invoke no peers", not "preview a run the server would refuse".
    enforce_mcp_budget(budget)

    if arguments.get("dry_run"):
        # Strict MCP clients reject any council_run response that doesn't
        # satisfy the advertised outputSchema. Build a schema-valid envelope
        # with sentinel values for fields a real run would populate, and keep
        # the fully enforced preview details in metadata.
        participant_models = {
            name: (config.get("participants", {}).get(name, {}) or {}).get("model")
            for name in participants
        }
        from llm_council.display import render_summary_markdown

        dry_summary = render_summary_markdown(
            mode=mode,
            ok_count=0,
            total=len(participants),
            elapsed_seconds=0.0,
            recommendation="unknown",
            per_peer_rows=[
                {
                    "name": name,
                    "label": "(dry-run)",
                    "stance": None,
                    "elapsed_seconds": 0.0,
                }
                for name in participants
            ],
            transcript_path=None,
            deliberated=False,
            rounds=0,
        )
        dry_metadata: dict[str, Any] = {
            "dry_run": True,
            "prompt_chars": len(prompt),
            "deliberate": deliberate,
            "max_rounds": max_rounds,
            "budget": budget,
            "participant_models": participant_models,
            "images": [
                _public_image_entry(entry, cwd) for entry in image_manifest
            ],
            "config_warnings": warnings,
        }
        if cost_estimate_block is not None:
            dry_metadata["cost_estimate"] = cost_estimate_block
        if cost_warning_payload is not None:
            dry_metadata["cost_warning"] = cost_warning_payload
        if chunk_events:
            dry_metadata["chunk_events"] = list(chunk_events)
        return {
            "schema_version": COUNCIL_RUN_OUTPUT_SCHEMA_VERSION,
            "recommendation": "unknown",
            "agreement_count": 0,
            "total_labeled": 0,
            "degraded": True,
            "rounds": 0,
            "deliberated": False,
            "mode": mode,
            "current": current,
            "participants": participants,
            "results": [],
            "metadata": dry_metadata,
            "summary_markdown": dry_summary,
        }

    cfg = config.get("participants", {})
    # Stash warnings now so they survive into the post-run response. We
    # populate the metadata field once execute_council has returned its
    # own dict, but hold the list locally so it doesn't fall out of scope.
    _pending_config_warnings = warnings
    # Plan §5 progress semantics: `total = peers * effective_rounds + 1`
    # (the +1 reserves headroom for synthesis or cross-rank). Clamps
    # apply on `council_finish`. Skipped entirely when no progressToken
    # was set by the client (silent no-op fallback).
    _effective_rounds = max_rounds if deliberate else 1
    _planned_total = float(len(participants) * _effective_rounds + 1)
    _progress_cb = _build_mcp_progress_callback(
        mcp_session, progress_token, planned_total=_planned_total
    )
    results, metadata = await execute_council(
        participants,
        cfg,
        prompt,
        cwd,
        config,
        deliberate=deliberate,
        max_rounds=max_rounds,
        progress=_progress_cb,
        image_manifest=image_manifest or None,
        min_quorum=min_quorum_value,
        mode=mode,
        stances=mode_stances if isinstance(mode_stances, dict) else None,
        synthesize=synthesize,
        synthesizer_name=resolved_synthesizer,
        current=current,
        question=persisted_question,
        cross_rank=bool(arguments.get("cross_rank")),
        focus=resolved_focus,
    )
    if image_manifest:
        metadata["images"] = [
            _public_image_entry(entry, cwd) for entry in image_manifest
        ]
    if secret_scan_payload.get("detected_count") or _scan_policy != "off":
        metadata["secret_scan"] = secret_scan_payload
    if chunk_events:
        metadata["chunk_events"] = list(chunk_events)
        metadata["diff_chunking"] = dict(chunk_events[-1])
        progress_events = metadata.setdefault("progress_events", [])
        if isinstance(progress_events, list):
            progress_events.extend(chunk_events)
    # Record the pre-run independent-review suppression (only when it actually
    # occurred). Surfaced as a metadata flag rather than a mid-run progress
    # event because the decision precedes execute_council.
    if prior_context_suppressed:
        metadata["prior_context_suppressed_for_independence"] = True
    metadata["config_warnings"] = _pending_config_warnings
    # L7: compact cost-estimate echo so a caller who skipped `council_estimate`
    # still sees the cost signal. M6: non-fatal soft cost-warning (omitted when
    # not triggered).
    if cost_estimate_block is not None:
        metadata["cost_estimate"] = cost_estimate_block
    if cost_warning_payload is not None:
        metadata["cost_warning"] = cost_warning_payload
    from llm_council.adapters import classify_error
    from llm_council.deliberation import (
        recommendation_label,
        summarize_recommendations,
    )
    from llm_council.transcript import final_round_results

    final = final_round_results(results)
    vote_summary = summarize_recommendations(final)
    recommendation = vote_summary.recommendation
    agreement = vote_summary.agreement_count
    labeled_total = vote_summary.total_labeled
    # Persist the exact same final-round vote reduction returned to the MCP
    # caller. Transcript consumers must not have to recompute it (and risk
    # drifting on tie semantics) from cumulative multi-round results.
    metadata["recommendation"] = recommendation
    metadata["agreement_count"] = agreement
    metadata["total_labeled"] = labeled_total
    metadata["recommendation_counts"] = dict(vote_summary.counts)
    metadata["recommendation_tied"] = vote_summary.tied
    write_transcript(
        md_path,
        json_path,
        question=persisted_question,
        mode=mode,
        current=current,
        participants=participants,
        prompt=prompt,
        results=results,
        transparent=transparent,
        metadata=metadata,
        parent_run_id=parent_run_id,
    )
    structured_results = []
    for result in results:
        label = (
            recommendation_label(result.output)
            if result.ok and result.output
            else None
        )
        structured_results.append(
            {
                "name": result.name,
                "ok": result.ok,
                "label": label,
                "stance": result.stance,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "wall_elapsed_seconds": (
                    round(result.wall_elapsed_seconds, 3)
                    if result.wall_elapsed_seconds is not None
                    else None
                ),
                "error": result.error,
                "error_kind": classify_error(result.error),
                "model": result.model,
                "total_tokens": result.total_tokens,
                "cost_usd": result.cost_usd,
                "from_cache": bool(result.from_cache),
                "cache_hit_seconds": result.cache_hit_seconds,
                "recovered_after_launch_retry": bool(
                    result.recovered_after_launch_retry
                ),
                "repair_retry_recovered": bool(result.repair_retry_recovered),
                "recovered_after_timeout": bool(result.recovered_after_timeout),
                "model_fallback_used": getattr(result, "model_fallback_used", None),
                "recovered_after_quota": bool(
                    getattr(result, "recovered_after_quota", False)
                ),
                "tool_call_status": getattr(result, "tool_call_status", None),
                "terse_retry_attempted": bool(
                    getattr(result, "terse_retry_attempted", False)
                ),
                "section_repair_attempted": bool(
                    getattr(result, "section_repair_attempted", False)
                ),
                "is_ranking_round": bool(getattr(result, "is_ranking_round", False)),
                "continue_debate": getattr(result, "continue_debate", None),
                "evidence_verification_failures": list(
                    getattr(result, "evidence_verification_failures", None) or []
                ),
                "prompt_chars": result.prompt_chars,
                "effort": result.effort,
                "confidence": result.confidence,
                "risk": result.risk,
                "blockers": list(result.blockers),
                "evidence": list(result.evidence),
                "tests_to_run": list(result.tests_to_run),
                "assumptions": list(result.assumptions),
            }
        )
    from llm_council.display import render_summary_markdown

    # Surface a markdown payload host agents tend to preserve verbatim
    # when quoting tool output. Use only the final-round results so the
    # per-peer table reflects each peer's last position rather than
    # listing every round separately. The blockquoted transcript path
    # gives the user a copy-pasteable pointer to the full record.
    final_names = {r.name for r in final}
    final_peer_rows = [
        {
            "name": row["name"].split(":round", 1)[0],
            "label": row.get("label"),
            "stance": row.get("stance"),
            "elapsed_seconds": row.get("elapsed_seconds") or 0,
            "wall_elapsed_seconds": row.get("wall_elapsed_seconds"),
        }
        for row in structured_results
        if row["name"] in final_names
    ]
    summary_markdown = render_summary_markdown(
        mode=mode,
        ok_count=sum(1 for r in final if r.ok),
        total=len(final),
        elapsed_seconds=sum(r.elapsed_seconds for r in results),
        wall_elapsed_seconds=metadata.get("run_wall_elapsed_seconds"),
        recommendation=recommendation,
        per_peer_rows=final_peer_rows,
        transcript_path=str(md_path),
        deliberated=bool(metadata.get("deliberated")),
        rounds=int(metadata.get("rounds") or 1),
    )
    # Shared "pop key from metadata and lift its value to the top-level
    # payload" pattern. Returns the (possibly copied) metadata plus the
    # lifted value, preserving the historical `metadata.pop(key) or default`
    # fallback so a key whose value is falsy collapses to `default`.
    def _lift(meta: Any, key: str, default: Any) -> tuple[Any, Any]:
        if isinstance(meta, dict) and key in meta:
            meta = dict(meta)
            return meta, (meta.pop(key) or default)
        return meta, default

    # `finding_matrix` is lifted to the top-level `consensus_blockers` /
    # `single_peer_concerns` keys below. Strip it from `metadata` before
    # surfacing the payload so the same data is not serialized in two
    # places (metadata.finding_matrix AND the top-level lists).
    metadata, finding_matrix_payload = _lift(metadata, "finding_matrix", {})
    consensus_blockers = list(finding_matrix_payload.get("consensus_blockers") or [])
    single_peer_concerns = list(finding_matrix_payload.get("single_peer_concerns") or [])
    # Mirror the finding-matrix pattern for quota-throttled peers: lift
    # the cumulative list from `metadata.quota_throttled_peers` to a
    # top-level payload key so a calling agent (e.g. Claude Code) can
    # spot rate-limited peers without parsing per-result `error_kind`
    # fields. Strip from metadata to avoid double-serialization.
    metadata, quota_throttled_peers = _lift(metadata, "quota_throttled_peers", [])
    # Phase 2 parallel: quota recoveries. Lifted with the same pattern so
    # the operator can see "this peer hit quota but recovered via fallback
    # to <model>" without parsing per-result fields.
    metadata, quota_recoveries = _lift(metadata, "quota_recoveries", [])
    # v0.12.0: peers dropped before the run because their api_key_env was
    # unset. Lifted top-level so the operator gets a "you forgot to set
    # X env var" signal without parsing per-result errors (the peer
    # never produced a result — it was excluded pre-run).
    metadata, missing_key_peers = _lift(metadata, "missing_key_peers", [])
    # v0.16.0: pinned-model substitutions (e.g. Claude Fable 5 refused and the
    # Claude Code surface silently served Opus). Lifted with the same pattern
    # as quota_throttled_peers so the calling agent sees the swap at the
    # advertised top-level location instead of parsing metadata.
    metadata, model_substituted_peers = _lift(
        metadata, "model_substituted_peers", []
    )
    # M11 provenance: lift applied review-focus bundles top-level (name +
    # short content hash) so a calling agent sees which inert focus
    # directives shaped the run without parsing metadata. Absent entirely
    # when no --focus / focus was applied (default no-focus path).
    metadata, applied_focus = _lift(metadata, "applied_focus", [])
    payload: dict[str, Any] = {
        "schema_version": COUNCIL_RUN_OUTPUT_SCHEMA_VERSION,
        "recommendation": recommendation,
        "agreement_count": agreement,
        "total_labeled": labeled_total,
        "degraded": bool(metadata.get("degraded")),
        "rounds": int(metadata.get("rounds") or 1),
        "deliberated": bool(metadata.get("deliberated")),
        "mode": mode,
        "current": current,
        "participants": participants,
        "metadata": metadata,
        "transcript": str(md_path),
        "json": str(json_path),
        "html": str(md_path.with_suffix(".html")),
        "results": structured_results,
        "summary_markdown": summary_markdown,
    }
    # Mirror the transcript JSON precedent: only emit the per-finding
    # matrix keys when at least one of them has content. Runs without a
    # FINDINGS envelope leave the keys absent rather than emitting empty
    # arrays.
    if consensus_blockers or single_peer_concerns:
        payload["consensus_blockers"] = consensus_blockers
        payload["single_peer_concerns"] = single_peer_concerns
    # Same omit-when-empty convention for quota_throttled_peers: the
    # common case has no quota issues and we don't want to ship an
    # empty array on every run.
    if quota_throttled_peers:
        payload["quota_throttled_peers"] = quota_throttled_peers
    if quota_recoveries:
        payload["quota_recoveries"] = quota_recoveries
    if missing_key_peers:
        payload["missing_key_peers"] = missing_key_peers
    if model_substituted_peers:
        payload["model_substituted_peers"] = model_substituted_peers
    if applied_focus:
        payload["applied_focus"] = applied_focus
    # v0.9.0 Feature 2: lift cross-rank fields to the top-level payload
    # mirroring finding_matrix. Strip them from metadata to avoid
    # double-serialization (same data appearing under metadata.* AND
    # top-level keys).
    metadata, cross_rank_scores_out = _lift(metadata, "cross_rank_scores", {})
    metadata, anonymization_map_out = _lift(metadata, "anonymization_map", {})
    # L7 / M6: lift the compact cost-estimate echo and the soft cost-warning
    # to top-level so a calling agent sees the cost signal without parsing
    # metadata. Omit-when-absent (the common no-cap-no-warn path leaves both
    # unset). `cost_estimate` is left in metadata too (mirrors `degraded`),
    # so re-anchor below picks it up; `cost_warning` is lifted out.
    metadata, cost_warning_out = _lift(metadata, "cost_warning", {})
    # Anchor the payload's metadata reference to the fully-popped dict.
    # `metadata` may have been copied/mutated by any of the lifts above
    # (pre- and post-payload); a single assignment here keeps
    # `payload["metadata"]` in sync without per-key re-anchors.
    payload["metadata"] = metadata
    if cross_rank_scores_out:
        payload["cross_rank_scores"] = cross_rank_scores_out
    if anonymization_map_out:
        payload["anonymization_map"] = anonymization_map_out
    # L7: surface the compact cost-estimate block top-level (still present in
    # metadata too). M6: surface the soft cost-warning top-level when triggered.
    if isinstance(metadata, dict) and metadata.get("cost_estimate"):
        payload["cost_estimate"] = metadata["cost_estimate"]
    if cost_warning_out:
        payload["cost_warning"] = cost_warning_out
    # H2 independence warning (advisory-only): surfaced top-level so a
    # calling agent can spot single-vendor quorums without parsing
    # metadata. Mirrors the omit-when-absent convention — the common case
    # has the feature off (key never set by the orchestrator). Left in
    # metadata too (like `degraded`); never overloads quorum/degraded.
    if isinstance(metadata, dict) and metadata.get("independence_warning"):
        payload["independence_warning"] = metadata["independence_warning"]
    # Independent-review suppression flag (advisory): mirror the
    # omit-when-absent convention so a calling agent can see that the prior
    # council context was intentionally withheld. Left in metadata too.
    if isinstance(metadata, dict) and metadata.get(
        "prior_context_suppressed_for_independence"
    ):
        payload["prior_context_suppressed_for_independence"] = True

    # Auto-open HTML transcript in browser if configured or requested
    auto_open = False
    if arguments.get("open"):
        auto_open = True
    else:
        defaults_cfg = config.get("defaults", {})
        if isinstance(defaults_cfg, dict) and defaults_cfg.get("auto_open_browser"):
            auto_open = True

    if auto_open:
        html_path = md_path.with_suffix(".html")
        if html_path.is_file():
            import webbrowser
            webbrowser.open(html_path.resolve().as_uri())

    return payload


def last_transcript(arguments: dict[str, Any]) -> dict[str, Any]:
    cwd = _resolve_working_directory(arguments)
    load_project_env(cwd, stop_at=_mcp_root())
    config = load_config(find_config(cwd, stop_at=_mcp_root()), search=False)
    out_dir = transcript_dir_within_root(cwd, config, root=_mcp_root())
    path = latest_transcript(out_dir, suffix=".json" if arguments.get("format") == "json" else ".md")
    if path is None:
        return {"found": False, "path": None, "content": ""}
    return {"found": True, "path": str(path), "content": path.read_text(encoding="utf-8")}


def run_doctor(arguments: dict[str, Any]) -> dict[str, Any]:
    cwd = _resolve_working_directory(arguments)
    load_project_env(cwd, stop_at=_mcp_root())
    config = load_config(find_config(cwd, stop_at=_mcp_root()), search=False)
    result: dict[str, Any] = {
        "config_warnings": config_warnings(config),
        "checks": checks_to_dict(
            check_environment(
                config,
                probe_openrouter=bool(arguments.get("probe_openrouter")),
                probe_ollama=bool(arguments.get("probe_ollama")),
                probe_native=bool(arguments.get("probe_native")),
                probe_cwd=cwd,
            )
        )
    }
    out_dir = transcript_dir_within_root(cwd, config, root=_mcp_root())
    if arguments.get("repair_transcript_permissions"):
        result["transcript_permissions"] = (
            inspect_transcript_permissions(out_dir, repair=True)
            if out_dir.is_dir()
            else {"directory": str(out_dir), "status": "not_created"}
        )
    result["server"] = {
        "version": __version__,
        "project_root": str(_mcp_root()),
        "working_directory": str(cwd),
        "config_path": str(find_config(cwd, stop_at=_mcp_root()) or ""),
        "project_scoped": True,
    }
    result["version"] = __version__
    if arguments.get("check_update"):
        result["update"] = check_for_update(__version__).to_dict()
    return result


def _peers_to_consider_dropping(cwd: Path, config: dict[str, Any]) -> list[str]:
    """L6 advisory: peer names whose recorded reliability suggests they be
    reconsidered. Defensive — never raises; returns [] on any failure or
    when there is no data. Advisory only: does NOT change participant
    selection or the run.
    """
    try:
        reliability = aggregate_reliability(
            cwd,
            transcripts_dir=transcript_dir_within_root(
                cwd, config, root=_mcp_root()
            ),
        )
        return policy.peers_to_consider_dropping(reliability)
    except Exception:
        return []


async def _run_recommend(arguments: dict[str, Any]) -> dict[str, Any]:
    """`council_recommend` handler.

    Primary verdict is always the mechanical, zero-cost `policy.recommend`
    output (M10). Two advisory enrichments layer on top — both are loaded
    from config but neither changes the council run or participant selection:

      - L6: `peers_to_consider_dropping` from recorded reliability.
      - M9: an optional LLM difficulty `judge` (default OFF, fail-open),
        attached as `result["judge"]` only when `defaults.recommend_judge`
        is set and the call succeeds. The mechanical verdict is NEVER
        overridden by the judge.
    """
    task = arguments["task"]
    result = policy.recommend(
        task,
        failed_attempts=int(arguments.get("failed_attempts") or 0),
        files_touched=int(arguments.get("files_touched") or 0),
        risk=arguments.get("risk") or "medium",
    )
    # Config is needed for both L6 and the optional M9 judge. A config-load
    # failure must not break the always-on mechanical verdict.
    config: dict[str, Any] | None = None
    try:
        cwd = _resolve_working_directory(arguments)
        load_project_env(cwd, stop_at=_mcp_root())
        config = load_config(find_config(cwd, stop_at=_mcp_root()), search=False)
    except Exception:
        config = None

    if config is not None:
        result["peers_to_consider_dropping"] = _peers_to_consider_dropping(cwd, config)
        # M9 judge: only when explicitly enabled via defaults.recommend_judge.
        defaults_cfg = config.get("defaults") or {}
        if defaults_cfg.get("recommend_judge"):
            judge = await grade_difficulty(task, config)
            if judge is not None:
                result["judge"] = {
                    "difficulty": judge.get("difficulty"),
                    "rationale": judge.get("rationale"),
                    "suggested_mode": judge.get("suggested_mode"),
                }
    else:
        result["peers_to_consider_dropping"] = []
    return result


def estimate_run(arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        cwd = _resolve_working_directory(arguments)
        load_project_env(cwd, stop_at=_mcp_root())
        config = load_config(find_config(cwd, stop_at=_mcp_root()), search=False)
        warnings = config_warnings(config)
        mode = canonical_mode_name(
            config,
            arguments.get("mode")
            or config.get("defaults", {}).get("mode", "quick"),
        )
        from llm_council.config import apply_smart_routing

        # Match `council_run`: automatic low-risk routing first, explicit tier
        # last so an operator pin cannot be silently downgraded.
        apply_smart_routing(config, mode, cwd)
        tier = arguments.get("tier")
        if tier:
            apply_tier_override(config, str(tier))
        defaults_cfg = config.setdefault("defaults", {})
        defaults_cfg["max_prompt_chars"] = min(
            int(defaults_cfg.get("max_prompt_chars") or MAX_PROMPT_CHARS),
            int(
                defaults_cfg.get("mcp_max_prompt_chars")
                or DEFAULT_MCP_MAX_PROMPT_CHARS
            ),
        )
        current = arguments.get("current") or detect_current_agent()
        completion_tokens = (
            1500
            if arguments.get("completion_tokens") is None
            else int(arguments["completion_tokens"])
        )
        from datetime import datetime

        estimate_slug = (
            "estimate-" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        sweep_old_inline_inputs(cwd)
        inline_staged = _stage_inline_images(
            arguments.get("images"), cwd, estimate_slug
        )
        image_path_inputs = list(arguments.get("image_paths") or []) + inline_staged
        chunk_events: list[dict[str, Any]] = []
        estimate = estimate_council(
            config=config,
            cwd=cwd,
            question=arguments["question"],
            mode=mode,
            current=current,
            explicit=arguments.get("participants"),
            include=arguments.get("include"),
            origin_policy=arguments.get("origin_policy"),
            context_paths=arguments.get("context_files") or [],
            include_diff=bool(arguments.get("include_diff")),
            stdin_text=None,
            allow_outside_cwd=False,
            deliberate=bool(arguments.get("deliberate")),
            max_rounds=arguments.get("max_rounds"),
            completion_tokens=completion_tokens,
            openrouter_models=arguments.get("openrouter_models") or [],
            use_cache=not bool(arguments.get("no_cache")),
            image_paths=image_path_inputs or None,
            chunk_strategy=str(arguments.get("chunk_strategy") or "fail"),
            chunk_progress=chunk_events.append,
        )
        if chunk_events:
            estimate["chunk_events"] = chunk_events
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "config_warnings": warnings,
        "peers_to_consider_dropping": _peers_to_consider_dropping(cwd, config),
        **estimate,
    }


def list_models(arguments: dict[str, Any]) -> dict[str, Any]:
    models = fetch_openrouter_models(use_cache=not bool(arguments.get("no_cache")))
    if arguments.get("filter"):
        needle = str(arguments["filter"]).lower()
        models = [
            model
            for model in models
            if needle in model["id"].lower() or needle in model["name"].lower()
        ]
    if arguments.get("origin"):
        prefix = {"us": "US /", "china": "China /", "unknown": "Unknown"}[
            arguments["origin"]
        ]
        models = [model for model in models if str(model["origin"]).startswith(prefix)]
    limit = int(arguments.get("limit") or 40)
    return {"models": models[:limit]}


def run_stats(arguments: dict[str, Any]) -> dict[str, Any]:
    cwd = _resolve_working_directory(arguments)
    load_project_env(cwd, stop_at=_mcp_root())
    config = load_config(find_config(cwd, stop_at=_mcp_root()), search=False)
    out_dir = transcript_dir_within_root(cwd, config, root=_mcp_root())
    since_days = arguments.get("since_days")
    if since_days is not None:
        since_days = int(since_days)
        if since_days <= 0:
            raise ValueError("since_days must be a positive integer")
    participant = arguments.get("participant")
    if participant is not None and not isinstance(participant, str):
        raise ValueError("participant must be a string")
    return compute_stats(
        out_dir,
        participant=participant or None,
        since_days=since_days,
    )


def list_modes(arguments: dict[str, Any]) -> dict[str, Any]:
    cwd = _resolve_working_directory(arguments)
    load_project_env(cwd, stop_at=_mcp_root())
    config = load_config(find_config(cwd, stop_at=_mcp_root()), search=False)
    return {
        "modes": config.get("modes", {}),
        "participants": list(config.get("participants", {}).keys()),
    }


def query_transcripts(arguments: dict[str, Any]) -> dict[str, Any]:
    """Semantic search over recorded transcripts (Jaccard MVP).

    Mirrors the read-only / project-rooted resolution used by every other
    transcript-reading tool. Returns ``{matches: [...]}`` even when empty
    so the consumer can rely on the shape.
    """
    from llm_council.query import search_similar

    cwd = _resolve_working_directory(arguments)
    load_project_env(cwd, stop_at=_mcp_root())
    config = load_config(find_config(cwd, stop_at=_mcp_root()), search=False)
    out_dir = transcript_dir_within_root(cwd, config, root=_mcp_root())
    query_text = arguments.get("query")
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query must be a non-empty string")
    top_k_raw = arguments.get("top_k")
    top_k = 5 if top_k_raw is None else int(top_k_raw)
    if top_k < 1 or top_k > 50:
        raise ValueError("top_k must be between 1 and 50")
    matches = search_similar(query_text, top_k=top_k, runs_dir=out_dir)
    return {
        "matches": [
            {
                "run_id": match.run_id,
                "similarity": match.similarity,
                "question_excerpt": match.question_excerpt,
                "recommendation_label": match.recommendation_label,
                "timestamp": match.timestamp,
            }
            for match in matches
        ]
    }


def config_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get", "set"],
                "description": "Whether to get or set a config option.",
            },
            "key": {
                "type": "string",
                "description": "The dot-notation key path to retrieve or update (e.g., defaults.auto_open_browser).",
            },
            "value": {
                "type": "string",
                "description": "The value to set (used only for action='set'). Strings like 'true'/'false' will be parsed appropriately.",
            },
            "working_directory": _working_directory_schema(),
        },
        "required": ["action", "key"],
        "additionalProperties": False,
    }


def run_config(arguments: dict[str, Any]) -> dict[str, Any]:
    """Get or set configuration options programmatically via MCP."""
    from llm_council.cli import _get_nested_val, _set_nested_val, _parse_config_value
    import yaml

    cwd = _resolve_working_directory(arguments)
    load_project_env(cwd, stop_at=_mcp_root())
    cfg_file = find_config(cwd, stop_at=_mcp_root())
    if not cfg_file:
        raise ValueError("Configuration file not found. Run setup first.")
    cfg_path = Path(cfg_file)

    try:
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise ValueError(f"Failed to read configuration: {e}")

    action = arguments.get("action")
    key = arguments.get("key")
    if not key:
        raise ValueError("Key path is required")

    if action == "get":
        val = _get_nested_val(config, key)
        return {
            "key": key,
            "value": val,
            "success": True
        }
    elif action == "set":
        value_str = arguments.get("value")
        if value_str is None:
            raise ValueError("Value is required for set action")
        parsed_val = _parse_config_value(value_str)
        _set_nested_val(config, key, parsed_val)

        try:
            resolve_config_data(config)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Refusing to write invalid configuration: {exc}"
            ) from exc

        import tempfile

        temp_path: Path | None = None
        try:
            payload = yaml.safe_dump(config, sort_keys=False)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=cfg_path.parent,
                prefix=f".{cfg_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(payload)
                temp_path = Path(handle.name)
            temp_path.chmod(cfg_path.stat().st_mode)
            os.replace(temp_path, cfg_path)
        except Exception as exc:
            raise ValueError(f"Failed to write configuration: {exc}") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return {
            "key": key,
            "value": parsed_val,
            "success": True
        }
    else:
        raise ValueError(f"Unknown action: {action}")


async def _serve() -> None:
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
    except Exception as exc:  # pragma: no cover - depends on environment install
        raise SystemExit(
            "The 'mcp' Python package is required for MCP server mode. "
            "Install project requirements first."
        ) from exc

    app = Server("llm-council")

    def _tool(**kwargs: Any) -> Tool:
        """Build a Tool, dropping outputSchema if the installed mcp SDK is too old.

        outputSchema landed in MCP spec 2025-06; older `mcp` packages reject
        the kwarg. We try first with outputSchema, then fall back so old envs
        still get the tool, just without the typed advertisement.
        """
        try:
            return Tool(**kwargs)
        except TypeError:
            kwargs.pop("outputSchema", None)
            return Tool(**kwargs)

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            _tool(
                name="council_run",
                description="Run a read-only multi-agent council.",
                inputSchema=council_run_schema(),
                outputSchema=council_run_output_schema(),
            ),
            Tool(
                name="council_recommend",
                description="Recommend whether a task should go to council and which mode to use.",
                inputSchema=recommend_schema(),
            ),
            Tool(
                name="council_estimate",
                description="Estimate prompt size and OpenRouter costs before running council.",
                inputSchema=estimate_schema(),
            ),
            Tool(
                name="council_list_modes",
                description="List configured council modes and participants.",
                inputSchema={
                    "type": "object",
                    "properties": {"working_directory": _working_directory_schema()},
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="council_last_transcript",
                description="Read the latest council transcript from the project.",
                inputSchema=last_transcript_schema(),
            ),
            Tool(
                name="council_doctor",
                description="Diagnose local CLI, OpenRouter, Ollama, and MCP readiness.",
                inputSchema=doctor_schema(),
            ),
            Tool(
                name="council_models",
                description="List cached OpenRouter models with optional filter/origin.",
                inputSchema=models_schema(),
            ),
            Tool(
                name="council_stats",
                description=(
                    "Aggregate per-participant metrics across recorded "
                    "transcripts: run count, success rate, recommendation "
                    "label distribution, tokens, cost, and last-used time."
                ),
                inputSchema=stats_schema(),
            ),
            Tool(
                name="council_query_transcripts",
                description=(
                    "Semantic search across recorded council transcripts. "
                    "Returns the top-k prior runs whose questions overlap "
                    "with the query (Jaccard token similarity). Lets an "
                    "agent check whether council already weighed in on a "
                    "topic before launching a fresh consultation."
                ),
                inputSchema=query_transcripts_schema(),
            ),
            Tool(
                name="council_config",
                description="Get or set configuration keys in the project's .llm-council.yaml config file.",
                inputSchema=config_schema(),
            ),
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "council_run":
            # Grab the per-call MCP context to enable mid-run progress
            # notifications. Both fields are None-safe: a client that
            # didn't set `_meta.progressToken` (the on-the-wire shape)
            # gets the silent no-op fallback documented in
            # `_build_mcp_progress_callback`.
            session: Any = None
            progress_token: Any = None
            try:
                rc = app.request_context
                session = getattr(rc, "session", None)
                meta = getattr(rc, "meta", None)
                progress_token = (
                    getattr(meta, "progressToken", None) if meta is not None else None
                )
            except (LookupError, AttributeError):
                pass
            result = await run_council(
                arguments,
                mcp_session=session,
                progress_token=progress_token,
            )
        elif name == "council_recommend":
            cwd = _resolve_working_directory(arguments)
            with project_env_context(cwd, stop_at=_mcp_root()):
                result = await _run_recommend(arguments)
        elif name == "council_estimate":
            cwd = _resolve_working_directory(arguments)
            with project_env_context(cwd, stop_at=_mcp_root()):
                result = estimate_run(arguments)
        elif name == "council_list_modes":
            cwd = _resolve_working_directory(arguments)
            with project_env_context(cwd, stop_at=_mcp_root()):
                result = list_modes(arguments)
        elif name == "council_last_transcript":
            cwd = _resolve_working_directory(arguments)
            with project_env_context(cwd, stop_at=_mcp_root()):
                result = last_transcript(arguments)
        elif name == "council_doctor":
            cwd = _resolve_working_directory(arguments)
            with project_env_context(cwd, stop_at=_mcp_root()):
                result = run_doctor(arguments)
        elif name == "council_models":
            result = list_models(arguments)
        elif name == "council_stats":
            cwd = _resolve_working_directory(arguments)
            with project_env_context(cwd, stop_at=_mcp_root()):
                result = run_stats(arguments)
        elif name == "council_query_transcripts":
            cwd = _resolve_working_directory(arguments)
            with project_env_context(cwd, stop_at=_mcp_root()):
                result = query_transcripts(arguments)
        elif name == "council_config":
            cwd = _resolve_working_directory(arguments)
            with project_env_context(cwd, stop_at=_mcp_root()):
                result = run_config(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
        text_blocks = [TextContent(type="text", text=json.dumps(result, indent=2))]
        # Tools that advertise an outputSchema MUST return structuredContent
        # alongside the text payload — strict MCP clients refuse the call
        # otherwise. Right now only `council_run` is typed; if more tools
        # gain outputSchema, extend this set.
        if name == "council_run":
            return text_blocks, result
        return text_blocks

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run llm-council MCP server")
    parser.parse_args(argv or [])
    asyncio.run(_serve())
    return 0


_INLINE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
INLINE_INPUTS_RETENTION_DAYS = 7


def sweep_old_inline_inputs(
    cwd: Path, *, retention_days: int = INLINE_INPUTS_RETENTION_DAYS
) -> int:
    """Best-effort cleanup of staged inline-image directories older than
    `retention_days`. Returns the number of directories removed.

    Called opportunistically before staging new inputs, so a long-running
    project doesn't accumulate gigabytes of screenshot artifacts. Failures
    are swallowed: cleanup must never block a council run.
    """

    import shutil
    import time

    inputs_root = cwd / ".llm-council" / "inputs"
    if not inputs_root.is_dir():
        return 0
    cutoff = time.time() - max(0, retention_days) * 86400
    removed = 0
    try:
        candidates = list(inputs_root.iterdir())
    except OSError:
        return 0
    for entry in candidates:
        if not entry.is_dir():
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            continue
        try:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        except OSError:
            continue
    return removed


def _stage_inline_images(
    images: list[dict[str, Any]] | None,
    cwd: Path,
    run_slug: str,
) -> list[str]:
    if not images:
        return []
    if not isinstance(images, list):
        raise ValueError("images must be an array of {data, mime, name?} entries")
    inputs_root = cwd / ".llm-council" / "inputs" / run_slug
    inputs_root.mkdir(parents=True, exist_ok=True)
    staged_relative: list[str] = []
    total_bytes = 0
    for index, entry in enumerate(images):
        if not isinstance(entry, dict):
            raise ValueError("images entry must be an object")
        mime = entry.get("mime")
        if mime not in IMAGE_MIME_ALLOWLIST:
            raise ValueError(
                f"Inline image #{index} mime '{mime}' is not allowed. "
                f"Allowed: {', '.join(sorted(IMAGE_MIME_ALLOWLIST))}."
            )
        data = entry.get("data")
        if not isinstance(data, str) or not data:
            raise ValueError(f"Inline image #{index} missing base64 'data'")
        # Cheap pre-decode size guard: base64 expands ~4/3.
        approx_bytes = (len(data) * 3) // 4
        if approx_bytes > DEFAULT_IMAGE_MAX_BYTES:
            raise ValueError(
                f"Inline image #{index} exceeds per-file budget before decode "
                f"(~{approx_bytes} > {DEFAULT_IMAGE_MAX_BYTES})"
            )
        if total_bytes + approx_bytes > DEFAULT_IMAGE_TOTAL_MAX_BYTES:
            raise ValueError(
                "Inline images exceed total attachment budget before decode "
                f"(~{total_bytes + approx_bytes} > {DEFAULT_IMAGE_TOTAL_MAX_BYTES})"
            )
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Inline image #{index} base64 decode failed: {exc}") from exc
        total_bytes += len(decoded)
        if len(decoded) > DEFAULT_IMAGE_MAX_BYTES:
            raise ValueError(
                f"Inline image #{index} exceeds per-file budget after decode "
                f"({len(decoded)} > {DEFAULT_IMAGE_MAX_BYTES})"
            )
        if total_bytes > DEFAULT_IMAGE_TOTAL_MAX_BYTES:
            raise ValueError(
                "Inline images exceed total attachment budget "
                f"({total_bytes} > {DEFAULT_IMAGE_TOTAL_MAX_BYTES})"
            )
        ext = _MIME_TO_EXT.get(mime, "")
        raw_name = entry.get("name") or f"img-{index:02d}{ext}"
        safe_name = _INLINE_NAME_RE.sub("-", str(raw_name)).strip("-") or f"img-{index:02d}{ext}"
        # Force the extension to match the declared mime so downstream
        # mimetypes.guess_type matches what the host claimed.
        if Path(safe_name).suffix.lower() != ext:
            safe_name = Path(safe_name).stem + ext
        target = inputs_root / safe_name
        # Avoid path collisions if the host reuses names.
        suffix = 0
        while target.exists():
            suffix += 1
            target = inputs_root / f"{Path(safe_name).stem}-{suffix}{ext}"
        target.write_bytes(decoded)
        staged_relative.append(str(target.resolve().relative_to(cwd.resolve())))
    return staged_relative


def _public_image_entry(entry: dict[str, Any], cwd: Path) -> dict[str, Any]:
    return {
        "path": entry.get("relative_path") or entry.get("path"),
        "mime": entry.get("mime"),
        "size": entry.get("size"),
        "sha256": entry.get("sha256"),
    }


def _mcp_root() -> Path:
    return Path(os.environ.get("LLM_COUNCIL_MCP_ROOT") or ".").resolve()


def _resolve_working_directory(arguments: dict[str, Any]) -> Path:
    root = _mcp_root()
    requested = arguments.get("working_directory")
    if requested and not Path(str(requested)).is_absolute():
        raise ValueError(
            "WorkingDirectoryMustBeAbsolute: requested="
            f"{requested!r}; configured_root={root}. Pass an absolute path."
        )
    cwd = Path(requested or root).resolve()
    if not cwd.exists():
        raise ValueError(f"working_directory does not exist: {cwd}")
    if not cwd.is_dir():
        raise ValueError(f"working_directory is not a directory: {cwd}")
    try:
        cwd.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "ProjectRootMismatch: requested="
            f"{cwd}; configured_root={root}. This MCP server is project-scoped; "
            "working_directory must be inside MCP project root. "
            "restart/reconnect it from the target checkout or use that "
            "checkout's .mcp.json."
        ) from exc
    return cwd


if __name__ == "__main__":
    raise SystemExit(main())
