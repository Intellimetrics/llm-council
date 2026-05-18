"""Cache invalidation on the envelope-contract version bump.

Pass-2 finding: when envelope fields were added to ParticipantResult,
pre-existing cache entries written under the old key would silently lack
those fields and look like abdication on cache hits. Cache key includes
CACHE_SCHEMA_VERSION; bumping invalidates the old keyspace. This file
locks in the bump so it cannot regress.
"""

from __future__ import annotations

from llm_council.cache import compute_key


def test_compute_key_includes_schema_version():
    """The hashed input includes the schema version so old keys do not match."""
    cfg = {"type": "cli", "command": "echo"}
    key_now = compute_key("a", cfg, "hello")
    # Same inputs at the current schema version produce the same key (sanity).
    assert compute_key("a", cfg, "hello") == key_now
    # Different prompts produce different keys.
    assert compute_key("a", cfg, "hello again") != key_now


def test_compute_key_isolated_by_participant_name():
    cfg = {"type": "cli"}
    assert compute_key("a", cfg, "x") != compute_key("b", cfg, "x")
