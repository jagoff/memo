"""Regression: a semantic-contradiction conflict must close when its pair is
judged, and the operational state surfaces must not grow without bound.

Root cause (found during an end-user QA run): ``emit_anomaly`` was only ever
called with ``"open"``. Nothing flipped the operational conflict to
``resolved`` once the underlying relation was judged in the canonical relation
ledger, so ``detected`` conflicts accumulated forever — the SessionStart
briefing listed conflicts that were already settled, and
``memo_operational_state`` grew past the MCP client's token cap (91 KB on the
live corpus, 68 of 111 conflicts already resolved).
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from memo.cli import cli
from memo.operational import ActorIdentity, OperationalStore

MEM_A = "4bb2ff4e756b4f35a94e989b7b1e8efb"
MEM_B = "d459004fa4df49dda36d36e455c1a716"
MEM_C = "9f1c0a2b3d4e5f60718293a4b5c6d7e8"


def _open_semantic_conflict(
    store: OperationalStore,
    *,
    a: str = MEM_A,
    b: str = MEM_B,
    anomaly_id: str = "anomaly-test-0001",
) -> str:
    store.record_anomaly(
        {
            "anomaly_id": anomaly_id,
            "kind": "semantic_contradiction",
            "state": "detected",
            "summary": f"memo contradiction between memories {a[:12]} and {b[:12]}",
            "memory_id_a": a,
            "memory_id_b": b,
            "relationship": "contradiction",
            "evidence_uris": [f"memo://memoria/{a}", f"memo://memoria/{b}"],
            "created_at": "2026-08-05T18:28:41+00:00",
        },
    )
    store.state()
    return anomaly_id


def test_gc_conflicts_for_pair_resolves_the_judged_pair(tmp_path) -> None:
    store = OperationalStore(tmp_path, device_id="device-a")
    _open_semantic_conflict(store)

    # Endpoint order must not matter: the ledger may judge B->A.
    assert store.gc_conflicts_for_pair(MEM_B, MEM_A, reason="relation judged") == 1
    assert store.active_conflicts(f"touch {MEM_A}") == []
    # Idempotent.
    assert store.gc_conflicts_for_pair(MEM_A, MEM_B) == 0


def test_gc_conflicts_for_pair_rejects_a_degenerate_pair(tmp_path) -> None:
    store = OperationalStore(tmp_path, device_id="device-a")
    _open_semantic_conflict(store)

    # A self-pair or a blank endpoint identifies no conflict; closing anything
    # on that basis would resolve records the caller never named.
    assert store.gc_conflicts_for_pair(MEM_A, MEM_A) == 0
    assert store.gc_conflicts_for_pair(MEM_A, "  ") == 0
    assert len(store.active_conflicts(f"touch {MEM_A}")) == 1


def test_mcp_state_tool_defaults_to_open_only(tmp_path) -> None:
    from unittest.mock import MagicMock

    from memo.server_operational import register

    store = OperationalStore(tmp_path, device_id="device-a")
    open_id = _open_semantic_conflict(store)
    closed = store.open_conflict(topic="billing architecture", summary="two designs")
    store.resolve_conflict(
        closed.id,
        resolution="picked design B",
        actor=ActorIdentity(actor_id="fer", actor_kind="human"),
    )

    tools: dict[str, object] = {}
    server = MagicMock()
    server.tool = lambda **_kw: lambda fn: tools.setdefault(fn.__name__, fn)
    memory = MagicMock()
    memory.operational = store
    register(server, memory)

    state_tool = tools["memo_operational_state"]
    assert set(state_tool()["conflicts"]) == {open_id}
    assert set(state_tool(include_closed=True)["conflicts"]) == {open_id, closed.id}


def test_gc_conflicts_for_pair_leaves_other_pairs_open(tmp_path) -> None:
    store = OperationalStore(tmp_path, device_id="device-a")
    _open_semantic_conflict(store)
    _open_semantic_conflict(store, a=MEM_A, b=MEM_C, anomaly_id="anomaly-test-0002")

    assert store.gc_conflicts_for_pair(MEM_A, MEM_B) == 1
    # The A/C conflict is a different pair — a half-overlap must not close it.
    assert len(store.active_conflicts(f"touch {MEM_C}")) == 1


def test_state_hides_closed_items_by_default(tmp_path) -> None:
    store = OperationalStore(tmp_path, device_id="device-a")
    human = ActorIdentity(actor_id="fer", actor_kind="human")

    open_id = _open_semantic_conflict(store)
    closed = store.open_conflict(topic="billing architecture", summary="two designs")
    store.resolve_conflict(closed.id, resolution="picked design B", actor=human)

    live = store.create_handoff(project="memo", summary="live handoff", from_actor="a")
    done = store.create_handoff(project="memo", summary="done handoff", from_actor="a")
    store.consume_handoff(done.id, actor_id="b")

    hot = store.add_attention(project="memo", summary="unread", severity="high")
    cold = store.add_attention(project="memo", summary="read", severity="low")
    store.acknowledge_attention(cold.id, actor_id="b")

    state = store.state(include_closed=False)
    assert set(state["conflicts"]) == {open_id}
    assert set(state["handoffs"]) == {live.id}
    assert set(state["attention"]) == {hot.id}

    # The library default stays the raw projection — internal consumers
    # (briefing, outcome ranking) rebuild their own views from it.
    full = store.state()
    assert {open_id, closed.id} <= set(full["conflicts"])
    assert {live.id, done.id} <= set(full["handoffs"])
    assert {hot.id, cold.id} <= set(full["attention"])


def test_state_scoping_composes_with_project_filter(tmp_path) -> None:
    store = OperationalStore(tmp_path, device_id="device-a")
    mine = store.create_handoff(project="memo", summary="mine", from_actor="a")
    other = store.create_handoff(project="other", summary="other", from_actor="a")
    store.consume_handoff(other.id, actor_id="b")

    scoped = store.state(project="memo", include_closed=False)
    assert set(scoped["handoffs"]) == {mine.id}
    assert set(store.state(project="other", include_closed=False)["handoffs"]) == set()
    assert set(store.state(project="other")["handoffs"]) == {other.id}


def test_cli_state_defaults_to_open_only(tmp_path) -> None:
    env = {
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
    }
    store = OperationalStore(tmp_path / "state", device_id="device-a")
    open_id = _open_semantic_conflict(store)
    closed = store.open_conflict(topic="billing architecture", summary="two designs")
    store.resolve_conflict(
        closed.id,
        resolution="picked design B",
        actor=ActorIdentity(actor_id="fer", actor_kind="human"),
    )

    runner = CliRunner()
    default = runner.invoke(cli, ["operational", "state"], env=env)
    assert default.exit_code == 0, default.output
    assert set(json.loads(default.output)["conflicts"]) == {open_id}

    full = runner.invoke(cli, ["operational", "state", "--include-closed"], env=env)
    assert full.exit_code == 0, full.output
    assert set(json.loads(full.output)["conflicts"]) == {open_id, closed.id}


def test_judging_a_relation_closes_its_operational_conflict(mock_memory) -> None:
    older = mock_memory.save(content="deploys run at 03:00", title="deploy window", type_="fact")
    newer = mock_memory.save(content="deploys run at 05:00", title="deploy window v2", type_="fact")
    candidate = mock_memory.store.create_relation_candidate(
        source_id=newer.id,
        target_id=older.id,
        suggested_relation="conflicts_with",
        reason="same subject, different hour",
        confidence=0.95,
        provenance={"generator": "contradiction_scanner"},
    )
    _open_semantic_conflict(mock_memory.operational, a=newer.id, b=older.id)
    assert len(mock_memory.operational.active_conflicts(f"touch {newer.id}")) == 1

    mock_memory.judge_relation(
        str(candidate["id"]),
        "supersedes",
        reason="newer window wins",
        confidence=0.95,
        actor="qa",
        actor_kind="agent",
    )

    assert mock_memory.operational.active_conflicts(f"touch {newer.id}") == []


def test_mcp_state_tool_bounds_an_unbounded_open_backlog(tmp_path) -> None:
    """Open conflicts awaiting human triage must not blow the response budget.

    Detection outruns triage: the nightly scan opens a semantic-contradiction
    conflict per pair, and pairs held back by the supersede support gate stay
    ``detected`` indefinitely. On the live corpus that reached 37 open
    conflicts (~10k tokens per call) and keeps climbing ~9/day, so the tool
    returns the newest slice plus the true totals.
    """
    from unittest.mock import MagicMock

    from memo.server_operational import register

    store = OperationalStore(tmp_path, device_id="device-a")
    for i in range(30):
        _open_semantic_conflict(
            store,
            a=f"{i:032x}",
            b=f"{i + 100:032x}",
            anomaly_id=f"anomaly-bulk-{i:04d}",
        )

    tools: dict[str, object] = {}
    server = MagicMock()
    server.tool = lambda **_kw: lambda fn: tools.setdefault(fn.__name__, fn)
    memory = MagicMock()
    memory.operational = store
    register(server, memory)

    out = tools["memo_operational_state"](limit=10)

    assert len(out["conflicts"]) == 10
    assert out["counts"]["conflicts"] == 30
    assert out["limit"] == 10
    # The full set stays reachable through the store itself.
    assert len(store.state()["conflicts"]) == 30
