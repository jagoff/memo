"""dream_edge_verify — graph edges earn confidence from grounded co-use:
evidence pairing, shared verified threshold (recall_assoc reads the same
constant), ledger-backed idempotency (F1), ledger-keyed decay (F3),
reconciliation after a rebuild wipe (F4), dry-run, never-raises."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from memo import dream_edge_verify as dev
from memo.dashboard_logs import append_grounding_log
from memo.graph import GraphStore

NOW = datetime(2026, 7, 20, 3, 0, tzinfo=UTC)

# Full memory ids; grounding.log stores the 8-char prefix.
A = "aaaa1111" + "0" * 24
B = "bbbb2222" + "0" * 24
C = "cccc3333" + "0" * 24

PK_AB = dev.pair_key(A, B)


def _row(sid: str, turn: int, rid: str, used: float) -> dict:
    return {"session_id": sid, "turn": turn, "recall_id": rid[:8], "used_score": used}


def _edge(a: str, b: str, *, confidence: float, created_at: str | None = None) -> dict:
    return {
        "source_kind": "memory",
        "source_id": a,
        "target_kind": "memory",
        "target_id": b,
        "relation": "supports",
        "weight": 1.0,
        "confidence": confidence,
        "evidence_id": a,
        "derived_from": "test.v1",
        "created_at": created_at or NOW.isoformat(),
        "valid_at": None,
        "invalid_at": None,
    }


def _entry(
    *,
    last_evidenced_at: str | None = None,
    curated: float | None = None,
    credited: list[str] | None = None,
) -> dict:
    e: dict = {"credited_turns": credited or []}
    if last_evidenced_at is not None:
        e["last_evidenced_at"] = last_evidenced_at
    if curated is not None:
        e["curated_confidence"] = curated
    return e


# --- co_used_turns: what counts as evidence ----------------------------------


def test_co_used_turns_collects_distinct_turn_keys_where_both_used():
    rows = [
        # turn 1: A and B both used -> 1 evidence turn for (A, B)
        _row("s1", 1, A, 0.9),
        _row("s1", 1, B, 0.85),
        # turn 2: same pair again (plus a duplicate row for A — still 1 turn)
        _row("s1", 2, A, 0.9),
        _row("s1", 2, A, 0.95),
        _row("s1", 2, B, 0.9),
    ]
    assert dev.co_used_turns(rows) == {PK_AB: {"s1|1", "s1|2"}}


def test_co_used_turns_ignores_rows_that_were_not_used():
    rows = [
        _row("s1", 1, A, 0.9),
        _row("s1", 1, B, 0.1),  # recalled but NOT used — no co-use evidence
        _row("s1", 2, C, 0.9),  # used alone — no pair
    ]
    assert dev.co_used_turns(rows) == {}


def test_co_used_turns_does_not_mix_sessions_or_turns():
    rows = [
        _row("s1", 1, A, 0.9),
        _row("s2", 1, B, 0.9),  # different session, same turn number
        _row("s1", 2, B, 0.9),  # same session, different turn
    ]
    assert dev.co_used_turns(rows) == {}


# --- verified: one shared threshold ------------------------------------------


def test_verified_is_the_shared_constant_not_a_retrieval_band():
    assert dev.verified(dev.VERIFIED_CONFIDENCE)
    assert not dev.verified(dev.VERIFIED_CONFIDENCE - 0.01)
    # The strongest deterministic-extractor prior (0.82) is NOT verified:
    # only grounded co-use promotion can clear the bar.
    assert not dev.verified(0.82)
    assert dev.VERIFIED_CONFIDENCE < dev.CONFIDENCE_CAP


# --- decide_edges: promotion --------------------------------------------------


def test_promotion_crosses_the_verified_threshold():
    """One grounded co-use turn lifts a strong extractor prior over the shared
    verified bar recall_assoc renders by."""
    decisions = dev.decide_edges([_edge(A, B, confidence=0.80)], {PK_AB: 1}, {}, now=NOW)
    (d,) = decisions
    assert d["action"] == "promote"
    assert abs(d["confidence"] - 0.90) < 1e-9
    assert d["verified"] is True
    assert d["evidence_turns"] == 1


def test_promotion_scales_with_evidence_and_is_capped():
    decisions = dev.decide_edges([_edge(A, B, confidence=0.9)], {PK_AB: 3}, {}, now=NOW)
    (d,) = decisions
    assert d["action"] == "promote"
    assert d["confidence"] == dev.CONFIDENCE_CAP


def test_edge_already_at_cap_is_held():
    decisions = dev.decide_edges(
        [_edge(A, B, confidence=dev.CONFIDENCE_CAP)], {PK_AB: 2}, {}, now=NOW
    )
    (d,) = decisions
    assert d["action"] == "hold"
    assert d["confidence"] == dev.CONFIDENCE_CAP


def test_promotion_matches_pair_regardless_of_edge_direction():
    # co-use pair keys are canonical (sorted); the edge is stored B -> A.
    decisions = dev.decide_edges([_edge(B, A, confidence=0.5)], {PK_AB: 1}, {}, now=NOW)
    assert decisions[0]["action"] == "promote"


def test_promotion_builds_on_curated_confidence_after_a_wipe():
    """New evidence on a wiped row promotes from the ledger's curated
    baseline, not from the re-extracted prior."""
    ledger = {PK_AB: _entry(last_evidenced_at=NOW.isoformat(), curated=0.90)}
    decisions = dev.decide_edges([_edge(A, B, confidence=0.70)], {PK_AB: 1}, ledger, now=NOW)
    (d,) = decisions
    assert d["action"] == "promote"
    assert d["confidence"] == dev.CONFIDENCE_CAP  # min(cap, 0.90 + 0.1)


# --- decide_edges: decay keyed on the ledger's evidence clock (F3) ------------


def test_edge_past_grace_since_last_evidence_decays():
    old = (NOW - timedelta(days=dev.DECAY_GRACE_DAYS + 1)).isoformat()
    ledger = {PK_AB: _entry(last_evidenced_at=old)}
    decisions = dev.decide_edges([_edge(A, B, confidence=0.8)], {}, ledger, now=NOW)
    (d,) = decisions
    assert d["action"] == "decay"
    assert abs(d["confidence"] - 0.8 * dev.DECAY_FACTOR) < 1e-9


def test_recently_evidenced_edge_is_not_decayed():
    recent = (NOW - timedelta(days=1)).isoformat()
    ledger = {PK_AB: _entry(last_evidenced_at=recent)}
    decisions = dev.decide_edges([_edge(A, B, confidence=0.8)], {}, ledger, now=NOW)
    assert decisions[0]["action"] == "hold"


def test_first_seen_edge_never_decays_regardless_of_created_at():
    """No ledger entry -> no decay clock yet, even for an ancient row (the
    old created_at keying eroded verified edges whose evidence rotated out
    of the capped grounding.log)."""
    ancient = (NOW - timedelta(days=400)).isoformat()
    decisions = dev.decide_edges([_edge(A, B, confidence=0.8, created_at=ancient)], {}, {}, now=NOW)
    (d,) = decisions
    assert d["action"] == "hold"
    assert d["reason"] == "first_seen"


def test_unparseable_last_evidenced_at_is_never_decayed():
    ledger = {PK_AB: _entry(last_evidenced_at="garbage")}
    decisions = dev.decide_edges([_edge(A, B, confidence=0.8)], {}, ledger, now=NOW)
    assert decisions[0]["action"] == "hold"


def test_decay_never_goes_below_the_floor_or_deletes():
    old = (NOW - timedelta(days=400)).isoformat()
    ledger = {
        PK_AB: _entry(last_evidenced_at=old),
        dev.pair_key(A, C): _entry(last_evidenced_at=old),
    }
    decisions = dev.decide_edges(
        [
            _edge(A, B, confidence=dev.CONFIDENCE_FLOOR + 0.001),
            _edge(A, C, confidence=dev.CONFIDENCE_FLOOR),
        ],
        {},
        ledger,
        now=NOW,
    )
    assert decisions[0]["action"] == "decay"
    assert decisions[0]["confidence"] == dev.CONFIDENCE_FLOOR
    # already at the floor -> held, never zeroed / removed
    assert decisions[1]["action"] == "hold"
    assert decisions[1]["confidence"] == dev.CONFIDENCE_FLOOR


# --- decide_edges: reconciliation (F4) ----------------------------------------


def test_wiped_row_is_reconciled_to_curated_confidence():
    """`memo graph relations rebuild` resets rows to extractor priors; the
    ledger's curated confidence restores them without new evidence."""
    ledger = {PK_AB: _entry(last_evidenced_at=NOW.isoformat(), curated=0.90)}
    decisions = dev.decide_edges([_edge(A, B, confidence=0.70)], {}, ledger, now=NOW)
    (d,) = decisions
    assert d["action"] == "reconcile"
    assert abs(d["confidence"] - 0.90) < 1e-9
    assert d["verified"] is True


# --- ledger sidecar -----------------------------------------------------------


def test_load_ledger_missing_or_corrupt_starts_fresh(tmp_path):
    assert dev.load_ledger(tmp_path) == {"version": dev.LEDGER_VERSION, "edges": {}}
    p = dev.ledger_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert dev.load_ledger(tmp_path) == {"version": dev.LEDGER_VERSION, "edges": {}}
    p.write_text('{"version": 999, "edges": {}}', encoding="utf-8")
    assert dev.load_ledger(tmp_path) == {"version": dev.LEDGER_VERSION, "edges": {}}


def test_save_ledger_round_trips(tmp_path):
    ledger = {
        "version": dev.LEDGER_VERSION,
        "edges": {PK_AB: _entry(last_evidenced_at=NOW.isoformat(), curated=0.9, credited=["s1|1"])},
    }
    dev.save_ledger(tmp_path, ledger)
    assert dev.load_ledger(tmp_path) == ledger


def test_credit_turns_bounds_history_and_refreshes_clock():
    entry = _entry(credited=[f"old|{i}" for i in range(dev.LEDGER_MAX_TURNS)])
    out = dev.credit_turns(entry, ["s1|1", "s1|2"], NOW.isoformat())
    assert len(out["credited_turns"]) == dev.LEDGER_MAX_TURNS  # capped
    assert out["credited_turns"][-2:] == ["s1|1", "s1|2"]  # newest retained
    assert out["last_evidenced_at"] == NOW.isoformat()
    # input never mutated; empty credit leaves the clock alone
    assert "last_evidenced_at" not in entry
    assert dev.credit_turns(entry, [], NOW.isoformat()).get("last_evidenced_at") is None


# --- run_edge_verify: store round-trip ---------------------------------------


def _graph(tmp_path) -> GraphStore:
    return GraphStore(tmp_path / "graph.db")


def _seed_edge(graph: GraphStore, a: str, b: str, *, confidence: float) -> None:
    graph.upsert_semantic_relation(
        source_kind="memory",
        source_id=a,
        target_kind="memory",
        target_id=b,
        relation="supports",
        weight=1.0,
        confidence=confidence,
        evidence_id=a,
        derived_from="test.v1",
    )


def _stored(graph: GraphStore, a: str) -> dict:
    row = graph._conn.execute(
        "SELECT * FROM semantic_relations WHERE source_id = ?", (a,)
    ).fetchone()
    assert row is not None
    return dict(row)


def _log_turn(cfg, sid: str, turn: int, *ids: str) -> None:
    for rid in ids:
        append_grounding_log(
            cfg.state_dir,
            session_id=sid,
            turn=turn,
            recall_id=rid,
            used_score=0.9,
            method="lexical",
        )


def test_run_edge_verify_promotes_co_used_edge_and_writes_ledger(tmp_cfg, tmp_path):
    graph = _graph(tmp_path)
    try:
        _seed_edge(graph, A, B, confidence=0.4)
        before = _stored(graph, A)
        _log_turn(tmp_cfg, "s1", 1, A, B)
        mem = SimpleNamespace(graph=graph)
        res = dev.run_edge_verify(tmp_cfg, mem, now=NOW)
        assert res["status"] == "done"
        assert res["pairs_evidenced"] == 1 and res["turns_credited"] == 1
        assert res["promoted"] == 1 and res["decayed"] == 0
        after = _stored(graph, A)
        assert abs(after["confidence"] - 0.5) < 1e-9
        # provenance untouched: created_at / weight / evidence_id preserved
        assert after["created_at"] == before["created_at"]
        assert after["weight"] == before["weight"]
        assert after["evidence_id"] == before["evidence_id"]
        # ledger recorded the credited turn + curation
        entry = dev.load_ledger(tmp_cfg.state_dir)["edges"][PK_AB]
        assert entry["credited_turns"] == ["s1|1"]
        assert entry["last_evidenced_at"] == NOW.isoformat()
        assert abs(entry["curated_confidence"] - 0.5) < 1e-6
    finally:
        graph.close()


def test_run_edge_verify_is_idempotent_over_the_same_log(tmp_cfg, tmp_path):
    """F1: re-running over the same grounding.log must not re-promote."""
    graph = _graph(tmp_path)
    try:
        _seed_edge(graph, A, B, confidence=0.4)
        _log_turn(tmp_cfg, "s1", 1, A, B)
        mem = SimpleNamespace(graph=graph)
        dev.run_edge_verify(tmp_cfg, mem, now=NOW)
        res2 = dev.run_edge_verify(tmp_cfg, mem, now=NOW + timedelta(days=1))
        assert res2["status"] == "done"
        assert res2["promoted"] == 0 and res2["turns_credited"] == 0
        assert abs(_stored(graph, A)["confidence"] - 0.5) < 1e-9
        # only genuinely NEW turns promote further
        _log_turn(tmp_cfg, "s1", 2, A, B)
        res3 = dev.run_edge_verify(tmp_cfg, mem, now=NOW + timedelta(days=2))
        assert res3["promoted"] == 1 and res3["turns_credited"] == 1
        assert abs(_stored(graph, A)["confidence"] - 0.6) < 1e-9
    finally:
        graph.close()


def test_run_edge_verify_decay_clock_survives_log_rotation(tmp_cfg, tmp_path):
    """F3: within-grace nights hold (even once evidence rotated out of the
    capped log); decay starts only past the grace since LAST EVIDENCE."""
    graph = _graph(tmp_path)
    try:
        _seed_edge(graph, A, B, confidence=0.4)
        _log_turn(tmp_cfg, "s1", 1, A, B)
        mem = SimpleNamespace(graph=graph)
        dev.run_edge_verify(tmp_cfg, mem, now=NOW)  # promote -> 0.5
        # simulate the capped log rotating the evidence away entirely
        (tmp_cfg.state_dir / "grounding.log").unlink()
        res = dev.run_edge_verify(tmp_cfg, mem, now=NOW + timedelta(days=5))
        assert res["decayed"] == 0 and res["held"] == 1
        assert abs(_stored(graph, A)["confidence"] - 0.5) < 1e-9
        res = dev.run_edge_verify(tmp_cfg, mem, now=NOW + timedelta(days=dev.DECAY_GRACE_DAYS + 1))
        assert res["decayed"] == 1
        assert abs(_stored(graph, A)["confidence"] - 0.5 * dev.DECAY_FACTOR) < 1e-9
    finally:
        graph.close()


def test_run_edge_verify_reconciles_after_rebuild_wipe(tmp_cfg, tmp_path):
    """F4: delete-by-derived_from + rebuild resets confidence to extractor
    priors; the next night restores the curated value from the ledger."""
    graph = _graph(tmp_path)
    try:
        _seed_edge(graph, A, B, confidence=0.70)
        _log_turn(tmp_cfg, "s1", 1, A, B)
        _log_turn(tmp_cfg, "s1", 2, A, B)
        mem = SimpleNamespace(graph=graph)
        res = dev.run_edge_verify(tmp_cfg, mem, now=NOW)
        assert res["promoted"] == 1
        assert abs(_stored(graph, A)["confidence"] - 0.90) < 1e-6
        # rebuild wipe: row re-created at the extractor prior
        graph._conn.execute("DELETE FROM semantic_relations")
        graph._conn.commit()
        _seed_edge(graph, A, B, confidence=0.70)
        res2 = dev.run_edge_verify(tmp_cfg, mem, now=NOW + timedelta(days=1))
        assert res2["reconciled"] == 1 and res2["promoted"] == 0
        assert abs(_stored(graph, A)["confidence"] - 0.90) < 1e-6
    finally:
        graph.close()


def test_run_edge_verify_keeps_evidence_for_pairs_without_an_edge(tmp_cfg, tmp_path):
    """Evidence for a pair with no edge is NOT credited (burned) — it waits
    for the night the edge exists."""
    graph = _graph(tmp_path)
    try:
        _seed_edge(graph, A, B, confidence=0.4)
        _log_turn(tmp_cfg, "s1", 1, A, C)  # co-use for a pair with NO edge
        mem = SimpleNamespace(graph=graph)
        dev.run_edge_verify(tmp_cfg, mem, now=NOW)
        assert dev.pair_key(A, C) not in dev.load_ledger(tmp_cfg.state_dir)["edges"]
        _seed_edge(graph, A, C, confidence=0.4)
        res = dev.run_edge_verify(tmp_cfg, mem, now=NOW + timedelta(days=1))
        assert res["promoted"] == 1  # the preserved turn finally counts
        row_ac = graph._conn.execute(
            "SELECT confidence FROM semantic_relations WHERE target_id = ?", (C,)
        ).fetchone()
        assert abs(row_ac["confidence"] - 0.5) < 1e-9
        row_ab = graph._conn.execute(
            "SELECT confidence FROM semantic_relations WHERE target_id = ?", (B,)
        ).fetchone()
        assert abs(row_ab["confidence"] - 0.4) < 1e-9  # A->B row untouched
    finally:
        graph.close()


def test_run_edge_verify_dry_run_writes_nothing(tmp_cfg, tmp_path):
    graph = _graph(tmp_path)
    try:
        _seed_edge(graph, A, B, confidence=0.4)
        _log_turn(tmp_cfg, "s1", 1, A, B)
        mem = SimpleNamespace(graph=graph)
        res = dev.run_edge_verify(tmp_cfg, mem, dry_run=True, now=NOW)
        assert res["status"] == "done" and res["promoted"] == 1
        assert _stored(graph, A)["confidence"] == 0.4
        assert not dev.ledger_path(tmp_cfg.state_dir).exists()
    finally:
        graph.close()


def test_run_edge_verify_skips_without_graph_or_edges(tmp_cfg, tmp_path):
    assert dev.run_edge_verify(tmp_cfg, SimpleNamespace(graph=None))["status"] == "skipped"
    graph = _graph(tmp_path)
    try:
        res = dev.run_edge_verify(tmp_cfg, SimpleNamespace(graph=graph))
        assert res["status"] == "skipped" and res["edges_total"] == 0
    finally:
        graph.close()


def test_run_edge_verify_never_raises(tmp_cfg):
    class _BoomConn:
        def execute(self, *a, **kw):
            raise RuntimeError("graph exploded")

    mem = SimpleNamespace(graph=SimpleNamespace(_conn=_BoomConn()))
    res = dev.run_edge_verify(tmp_cfg, mem)
    assert res["status"] == "error" and "graph exploded" in res["error"]
