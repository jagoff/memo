"""Regression: orphaned semantic-contradiction conflicts must not freeze
unrelated writes, and deleting a memory must GC the conflicts it belongs to.

Root cause (found during a QA stress run): a native ``semantic_contradiction``
anomaly stores the two conflicting memory ids only inside the conflict's
``topic`` string, and ``active_conflicts`` token-matched the write's topic
against the conflict's prose ``summary`` ("memo contradiction between
memories ..."). Common words (``memo``, ``contradiction``, ``between``,
``memories``) therefore froze a large share of legitimate writes, and the
conflict was never cleared when its subject memories were deleted — leaving an
orphan that froze writes forever.
"""

from __future__ import annotations

from memo.operational import OperationalStore
from memo.write_policy import WritePolicyEngine

# Two real-looking memory ids that a contradiction was detected between.
MEM_A = "4bb2ff4e756b4f35a94e989b7b1e8efb"
MEM_B = "d459004fa4df49dda36d36e455c1a716"
# The exact prose the detector writes into the conflict summary.
SUMMARY = f"memo contradiction between memories {MEM_A[:12]} and {MEM_B[:12]}"


def _open_semantic_conflict(store: OperationalStore, *, a: str = MEM_A, b: str = MEM_B) -> str:
    anomaly_id = "anomaly-test-0001"
    store.record_anomaly(
        {
            "anomaly_id": anomaly_id,
            "kind": "semantic_contradiction",
            "state": "detected",
            "summary": SUMMARY,
            "memory_id_a": a,
            "memory_id_b": b,
            "relationship": "contradiction",
            "evidence_uris": [f"memo://memoria/{a}", f"memo://memoria/{b}"],
            "created_at": "2026-07-25T18:28:41+00:00",
        },
    )
    store.state()  # materialize the projection
    return anomaly_id


def test_semantic_conflict_does_not_freeze_unrelated_prose_writes(tmp_path):
    store = OperationalStore(tmp_path, device_id="device-a")
    _open_semantic_conflict(store)

    # These exact titles froze during the QA run because their tokens are
    # substrings of the conflict summary. None of them are about MEM_A/MEM_B.
    for topic in (
        "memo indexing benchmark results",
        "Notes on contradiction resolution",
        "between two services latency",
        "Five databases in use",
    ):
        assert store.active_conflicts(topic) == [], f"prose write wrongly frozen: {topic!r}"


def test_semantic_conflict_still_matches_by_member_id(tmp_path):
    store = OperationalStore(tmp_path, device_id="device-a")
    _open_semantic_conflict(store)

    # A write/update that actually references a subject memory id still matches.
    hits = store.active_conflicts(f"updating memory {MEM_A} with a new value")
    assert len(hits) == 1
    assert hits[0]["freeze_write"] is True


def test_topic_scoped_conflict_still_freezes_matching_topic(tmp_path):
    store = OperationalStore(tmp_path, device_id="device-a")
    store.open_conflict(
        topic="billing architecture",
        summary="Two incompatible billing designs are active",
    )
    store.state()

    # Manually-opened topic conflicts keep topic-token matching (unchanged).
    assert len(store.active_conflicts("Billing architecture redesign")) == 1
    # ...but an unrelated write is not frozen by a topic conflict either.
    assert store.active_conflicts("gardening tips for tomatoes") == []


def test_gc_conflicts_for_memory_resolves_orphan(tmp_path):
    store = OperationalStore(tmp_path, device_id="device-a")
    _open_semantic_conflict(store)
    assert len(store.active_conflicts(f"touch {MEM_A}")) == 1

    resolved = store.gc_conflicts_for_memory(MEM_A)
    assert resolved == 1
    assert store.active_conflicts(f"touch {MEM_A}") == []
    # Idempotent: a second GC (or GC of the other member) resolves nothing new.
    assert store.gc_conflicts_for_memory(MEM_A) == 0
    assert store.gc_conflicts_for_memory(MEM_B) == 0


def test_write_policy_allows_prose_write_when_only_semantic_conflict_exists(tmp_path):
    store = OperationalStore(tmp_path, device_id="device-a")
    _open_semantic_conflict(store)
    engine = WritePolicyEngine(store)

    decision = engine.preflight(
        title="memo indexing benchmark results",
        content="Benchmarked the index build.",
        tags=None,
        extra=None,
    )
    assert decision.allowed is True
    assert decision.reason == "allowed"


def _open_topic_conflict(store: OperationalStore, *, topic: str, summary: str) -> str:
    """A manually-opened, topic-scoped conflict — carries no subject memory ids."""
    record = store.open_conflict(topic=topic, summary=summary, freeze_write=True)
    store.state()  # materialize the projection
    return record.id


def test_topic_conflict_does_not_freeze_writes_sharing_one_word(tmp_path):
    """Regression: abandoned QA conflicts froze every write mentioning test/mcp.

    The topic branch matched when any >=3-char token of the *write* was a
    substring of the conflict topic, so `test_conflict` swallowed "test
    coverage ..." and `zzz_mcp_qa_probe_conflict` swallowed "mcp server ...".
    Found in the live store on 2026-08-04 with 4 such conflicts open.
    """
    store = OperationalStore(tmp_path, device_id="device-a")
    _open_topic_conflict(store, topic="test_conflict", summary="test conflict")
    _open_topic_conflict(
        store,
        topic="zzz_mcp_qa_probe_conflict",
        summary="Testing memo_conflict_open from MCP QA audit",
    )

    for topic in (
        "test coverage for the recall hook",
        "flaky test in CI",
        "mcp server registration",
        "add mcp tool for graph",
    ):
        assert store.active_conflicts(topic) == [], f"write wrongly frozen: {topic!r}"


def test_topic_conflict_still_freezes_writes_about_its_topic(tmp_path):
    store = OperationalStore(tmp_path, device_id="device-a")
    _open_topic_conflict(store, topic="billing_provider", summary="stripe vs adyen")

    for topic in (
        "billing_provider",
        "billing provider decision reversed",
        "we are switching the billing provider to adyen",
    ):
        assert store.active_conflicts(topic), f"write should be frozen: {topic!r}"


def test_topic_conflict_with_no_significant_tokens_never_freezes(tmp_path):
    """A degenerate topic must not blanket-freeze; resolve it by id instead."""
    store = OperationalStore(tmp_path, device_id="device-a")
    _open_topic_conflict(store, topic="t", summary="t")

    assert store.active_conflicts("anything at all") == []
    assert store.active_conflicts("t") == []
