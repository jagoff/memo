"""dream_hype — nightly HyPE question-generation pass (Task 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from memo import dream_hype as dh
from memo.store.hype_store import HypeStore


def _fake_embed_query(dims: int = 4):
    """Deterministic fake embedder: same text -> same vector, different
    lengths -> different vectors (good enough for store round-trips)."""

    def _embed(text: str) -> list[float]:
        h = float(len(text) % 7)
        return [h + i for i in range(dims)]

    return _embed


class _FakeStore:
    """Fake mem.store — all_ids/get/get_fts_body only, as consumed by dream_hype."""

    def __init__(self, memories: dict[str, dict]):
        self._memories = memories

    def all_ids(self) -> list[str]:
        return list(self._memories.keys())

    def get(self, id_: str) -> dict | None:
        return self._memories.get(id_)

    def get_fts_body(self, id_: str) -> str:
        return self._memories.get(id_, {}).get("_body", "")


class _FakeEmbedder:
    def __init__(self, dims: int = 4):
        self._fn = _fake_embed_query(dims)

    def embed_query(self, text: str) -> list[float]:
        return self._fn(text)


class _FakeCfg:
    def __init__(self, db_path: Path | None = None, state_dir: Path | None = None):
        self.db_path = db_path
        self.embedder_dims = 4
        self.helper_model = "fake-model"
        self.state_dir = state_dir


class _FakeMem:
    def __init__(self, memories: dict[str, dict], state_dir: Path | None = None):
        self.store = _FakeStore(memories)
        self.embedder = _FakeEmbedder()
        self.cfg = _FakeCfg(state_dir=state_dir)

    def _ensure_chat(self):
        return object()  # never actually invoked; _llm_questions is monkeypatched


def _mem_row(type_="decision", body_hash="h1", title="Decision about X"):
    return {"type": type_, "body_hash": body_hash, "title": title, "_body": "some body text"}


@pytest.fixture
def hype_store(tmp_path):
    store = HypeStore(tmp_path / "memvec.db", dims=4)
    yield store
    store.close()


# --- select_backlog -----------------------------------------------------------


def test_select_backlog_filters_by_durable_types(hype_store, tmp_path):
    memories = {
        "id_decision1": _mem_row(type_="decision"),
        "id_reference1": _mem_row(type_="reference"),  # not durable, excluded
    }
    mem = _FakeMem(memories, state_dir=tmp_path)
    backlog = dh.select_backlog(mem, hype_store, cap=10)
    ids = {item["id"] for item in backlog}
    assert ids == {"id_decision1"}


def test_select_backlog_skips_memories_with_matching_body_hash_watermark(hype_store, tmp_path):
    memories = {"id1": _mem_row(body_hash="matching_hash")}
    hype_store.replace_for_memory("id1", "matching_hash", "m", [("q1", [0.0, 0.0, 0.0, 0.0])])
    mem = _FakeMem(memories, state_dir=tmp_path)
    backlog = dh.select_backlog(mem, hype_store, cap=10)
    assert backlog == []


def test_select_backlog_includes_memory_when_body_hash_changed(hype_store, tmp_path):
    memories = {"id1": _mem_row(body_hash="new_hash")}
    hype_store.replace_for_memory("id1", "old_hash", "m", [("q1", [0.0, 0.0, 0.0, 0.0])])
    mem = _FakeMem(memories, state_dir=tmp_path)
    backlog = dh.select_backlog(mem, hype_store, cap=10)
    assert [item["id"] for item in backlog] == ["id1"]


def test_select_backlog_orders_by_roi_utility_desc(hype_store, tmp_path, monkeypatch):
    memories = {
        "id_low": _mem_row(),
        "id_high": _mem_row(),
        "id_missing": _mem_row(),  # no ROI data -> neutral 0.5
    }
    mem = _FakeMem(memories, state_dir=tmp_path)

    def _fake_compute_utilities(state_dir):
        return {
            "by_prefix": {
                "id_low"[:8]: {"utility": 0.1},
                "id_high"[:8]: {"utility": 0.9},
            }
        }

    monkeypatch.setattr("memo.outcome.compute_utilities", _fake_compute_utilities)
    backlog = dh.select_backlog(mem, hype_store, cap=10)
    ids_in_order = [item["id"] for item in backlog]
    assert ids_in_order.index("id_high") < ids_in_order.index("id_missing")
    assert ids_in_order.index("id_missing") < ids_in_order.index("id_low")


def test_select_backlog_respects_cap(hype_store, tmp_path):
    memories = {f"id{i}": _mem_row() for i in range(5)}
    mem = _FakeMem(memories, state_dir=tmp_path)
    backlog = dh.select_backlog(mem, hype_store, cap=2)
    assert len(backlog) == 2


# --- _llm_questions -------------------------------------------------------------


def test_llm_questions_parses_and_filters_short_questions(monkeypatch):
    """Real test of _llm_questions parsing logic with a stubbed chat_with_timeout."""

    def _fake_chat_with_timeout(chat, *, timeout, **kwargs):
        return {"message": {"content": '["¿Qué decidimos sobre X?", "corta"]'}}

    monkeypatch.setattr("memo.memory.record.chat_with_timeout", _fake_chat_with_timeout)
    mem = _FakeMem({})
    questions = dh._llm_questions(mem, "Title", "Body text", n=3)
    assert questions == ["¿Qué decidimos sobre X?"]


def test_llm_questions_returns_none_on_timeout(monkeypatch):
    monkeypatch.setattr(
        "memo.memory.record.chat_with_timeout", lambda chat, *, timeout, **kwargs: None
    )
    mem = _FakeMem({})
    assert dh._llm_questions(mem, "Title", "Body", n=3) is None


def test_llm_questions_returns_none_on_bad_json(monkeypatch):
    monkeypatch.setattr(
        "memo.memory.record.chat_with_timeout",
        lambda chat, *, timeout, **kwargs: {"message": {"content": "not json"}},
    )
    mem = _FakeMem({})
    assert dh._llm_questions(mem, "Title", "Body", n=3) is None


def test_llm_questions_strips_json_markdown_fence(monkeypatch):
    """LLMs sometimes wrap the JSON array in a ```json ... ``` fence — strip
    it before json.loads instead of failing to parse."""
    fenced = '```json\n["¿Qué decidimos sobre X?", "¿Por qué se eligió Y?"]\n```'
    monkeypatch.setattr(
        "memo.memory.record.chat_with_timeout",
        lambda chat, *, timeout, **kwargs: {"message": {"content": fenced}},
    )
    mem = _FakeMem({})
    questions = dh._llm_questions(mem, "Title", "Body", n=3)
    assert questions == ["¿Qué decidimos sobre X?", "¿Por qué se eligió Y?"]


def test_llm_questions_strips_plain_markdown_fence(monkeypatch):
    """Same as above but with a plain ``` fence (no language tag)."""
    fenced = '```\n["question one here", "question two here"]\n```'
    monkeypatch.setattr(
        "memo.memory.record.chat_with_timeout",
        lambda chat, *, timeout, **kwargs: {"message": {"content": fenced}},
    )
    mem = _FakeMem({})
    questions = dh._llm_questions(mem, "Title", "Body", n=3)
    assert questions == ["question one here", "question two here"]


# --- run_hype_pass ---------------------------------------------------------------


def test_run_hype_pass_generates_and_persists(tmp_path, monkeypatch):
    memories = {"id1": _mem_row(body_hash="h1")}
    mem = _FakeMem(memories, state_dir=tmp_path)
    cfg = _FakeCfg(tmp_path / "memvec.db")

    monkeypatch.setattr(dh, "_llm_questions", lambda mem, title, body, *, n: ["q1?", "q2?"])

    res = dh.run_hype_pass(cfg, mem, questions_per_memory=3, night_cap=400, dry_run=False)

    assert res["status"] == "done"
    assert res["generated"] == 2
    assert res["memories"] == 1
    assert res["errors_items"] == 0

    store = HypeStore(cfg.db_path, dims=4)
    try:
        stats = store.stats()
        assert stats == {"memories": 1, "questions": 2}
    finally:
        store.close()


def test_run_hype_pass_second_run_same_body_hash_is_skipped(tmp_path, monkeypatch):
    memories = {"id1": _mem_row(body_hash="h1")}
    mem = _FakeMem(memories, state_dir=tmp_path)
    cfg = _FakeCfg(tmp_path / "memvec.db")
    monkeypatch.setattr(dh, "_llm_questions", lambda mem, title, body, *, n: ["q1?", "q2?"])

    first = dh.run_hype_pass(cfg, mem, dry_run=False)
    assert first["status"] == "done"

    second = dh.run_hype_pass(cfg, mem, dry_run=False)
    assert second["status"] == "skipped"
    assert second["generated"] == 0


def test_run_hype_pass_skipped_when_backlog_empty(tmp_path):
    mem = _FakeMem({}, state_dir=tmp_path)
    cfg = _FakeCfg(tmp_path / "memvec.db")
    res = dh.run_hype_pass(cfg, mem, dry_run=False)
    assert res["status"] == "skipped"
    assert res["generated"] == 0


def test_run_hype_pass_llm_none_skips_item_without_aborting(tmp_path, monkeypatch):
    memories = {
        "id1": _mem_row(body_hash="h1", title="Decision about X"),
        "id2": _mem_row(body_hash="h2", title="Decision about Y"),
    }
    mem = _FakeMem(memories, state_dir=tmp_path)
    cfg = _FakeCfg(tmp_path / "memvec.db")

    def _fake_llm(mem, title, body, *, n):
        return None if title == "Decision about X" else ["q1?"]

    monkeypatch.setattr(dh, "_llm_questions", _fake_llm)
    res = dh.run_hype_pass(cfg, mem, dry_run=False)

    assert res["status"] == "done"
    assert res["errors_items"] == 1
    assert res["memories"] == 1
    assert res["generated"] == 1


def test_run_hype_pass_llm_empty_list_skips_item(tmp_path, monkeypatch):
    # Single-item backlog where that one item fails is also the
    # all-items-failed case (Fix 4) — status reflects the total wash.
    memories = {"id1": _mem_row(body_hash="h1")}
    mem = _FakeMem(memories, state_dir=tmp_path)
    cfg = _FakeCfg(tmp_path / "memvec.db")
    monkeypatch.setattr(dh, "_llm_questions", lambda mem, title, body, *, n: [])

    res = dh.run_hype_pass(cfg, mem, dry_run=False)
    assert res["status"] == "all_items_failed"
    assert res["errors_items"] == 1
    assert res["generated"] == 0


def test_run_hype_pass_never_raises(tmp_path, monkeypatch):
    mem = _FakeMem({"id1": _mem_row()}, state_dir=tmp_path)
    cfg = _FakeCfg(tmp_path / "memvec.db")

    def _boom(mem, store, *, cap):
        raise RuntimeError("boom")

    monkeypatch.setattr(dh, "select_backlog", _boom)
    res = dh.run_hype_pass(cfg, mem, dry_run=False)
    assert res["status"] == "error"
    assert "boom" in res.get("error", "")


def test_run_hype_pass_dry_run_computes_backlog_without_writing(tmp_path, monkeypatch):
    memories = {"id1": _mem_row(body_hash="h1")}
    mem = _FakeMem(memories, state_dir=tmp_path)
    cfg = _FakeCfg(tmp_path / "memvec.db")
    llm_called = []
    monkeypatch.setattr(
        dh, "_llm_questions", lambda mem, title, body, *, n: llm_called.append(1) or ["q1?"]
    )

    res = dh.run_hype_pass(cfg, mem, dry_run=True)

    assert res["status"] == "done"
    assert res["generated"] == 0
    assert res["backlog_remaining"] == 1
    assert not llm_called

    store = HypeStore(cfg.db_path, dims=4)
    try:
        assert store.stats() == {"memories": 0, "questions": 0}
    finally:
        store.close()


def test_run_hype_pass_real_run_backlog_remaining_reflects_failed_items(tmp_path, monkeypatch):
    """A REAL (non-dry-run) run must populate backlog_remaining honestly:
    items that failed (LLM returned None) were never written to the store, so
    they legitimately remain pending — len(backlog) - memories_succeeded."""
    memories = {
        "id1": _mem_row(body_hash="h1", title="Decision about X"),
        "id2": _mem_row(body_hash="h2", title="Decision about Y"),
        "id3": _mem_row(body_hash="h3", title="Decision about Z"),
    }
    mem = _FakeMem(memories, state_dir=tmp_path)
    cfg = _FakeCfg(tmp_path / "memvec.db")

    def _fake_llm(mem, title, body, *, n):
        return None if title == "Decision about X" else ["q1?"]

    monkeypatch.setattr(dh, "_llm_questions", _fake_llm)

    res = dh.run_hype_pass(cfg, mem, questions_per_memory=1, night_cap=400, dry_run=False)

    assert res["status"] == "done"
    assert res["memories"] == 2
    assert res["errors_items"] == 1
    assert res["backlog_remaining"] == 1


def test_run_hype_pass_all_items_failed_status(tmp_path, monkeypatch):
    """When every backlog item fails (LLM returns None for all), the run must
    still be a normal return (no raise) but with status='all_items_failed'
    instead of 'done', so the nightly receipt is honest about a total wash."""
    memories = {
        "id1": _mem_row(body_hash="h1"),
        "id2": _mem_row(body_hash="h2"),
    }
    mem = _FakeMem(memories, state_dir=tmp_path)
    cfg = _FakeCfg(tmp_path / "memvec.db")
    monkeypatch.setattr(dh, "_llm_questions", lambda mem, title, body, *, n: None)

    res = dh.run_hype_pass(cfg, mem, dry_run=False)

    assert res["status"] == "all_items_failed"
    assert res["errors_items"] == 2
    assert res["memories"] == 0


def test_run_hype_pass_prunes_orphans(tmp_path, monkeypatch):
    """A memory indexed previously but no longer live gets pruned at pass end."""
    cfg = _FakeCfg(tmp_path / "memvec.db")
    store = HypeStore(cfg.db_path, dims=4)
    store.replace_for_memory("stale_id", "old_hash", "m", [("q?", [0.0, 0.0, 0.0, 0.0])])
    store.close()

    memories = {"id1": _mem_row(body_hash="h1")}
    mem = _FakeMem(memories, state_dir=tmp_path)
    monkeypatch.setattr(dh, "_llm_questions", lambda mem, title, body, *, n: ["q1?"])

    res = dh.run_hype_pass(cfg, mem, dry_run=False)
    assert res["status"] == "done"
    assert res["pruned"] == 1

    verify_store = HypeStore(cfg.db_path, dims=4)
    try:
        assert verify_store.stats()["memories"] == 1
    finally:
        verify_store.close()


def test_dream_hype_subcommand_json(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from memo import dream_hype as dh_mod
    from memo.cli import cli

    monkeypatch.setattr(
        dh_mod,
        "run_hype_pass",
        lambda cfg, mem, **kw: {"status": "done", "generated": 5, "memories": 2},
    )
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }
    result = CliRunner().invoke(cli, ["dream", "hype", "--json"], env=env)
    assert result.exit_code == 0, result.output
    assert '"status": "done"' in result.output


def test_run_hype_pass_embed_failure_isolates_one_memory(tmp_path, monkeypatch):
    """Failure to embed questions for ONE memory does not abort the pass.

    Embedder fails for id1 but id2 succeeds → pass completes with status='done',
    only id2 written to store, errors_items=1, memories=1.
    """
    memories = {
        "id1": _mem_row(body_hash="h1", title="Decision with embed failure"),
        "id2": _mem_row(body_hash="h2", title="Decision that works"),
    }
    mem = _FakeMem(memories, state_dir=tmp_path)
    cfg = _FakeCfg(tmp_path / "memvec.db")

    monkeypatch.setattr(dh, "_llm_questions", lambda mem, title, body, *, n: ["q1?"])

    # Embedder that raises for id1's question
    original_embed = mem.embedder.embed_query

    call_count = {"count": 0}

    def _embed_failing_on_id1(text: str):
        call_count["count"] += 1
        if call_count["count"] == 1:  # First call is for id1
            raise RuntimeError("embed failed for id1")
        return original_embed(text)

    mem.embedder.embed_query = _embed_failing_on_id1

    res = dh.run_hype_pass(cfg, mem, questions_per_memory=3, night_cap=400, dry_run=False)

    assert res["status"] == "done", f"Expected 'done', got {res['status']}"
    assert res["errors_items"] == 1, f"Expected errors_items=1, got {res['errors_items']}"
    assert res["memories"] == 1, f"Expected memories=1, got {res['memories']}"
    assert res["generated"] >= 1, f"Expected generated>=1, got {res['generated']}"

    verify_store = HypeStore(cfg.db_path, dims=4)
    try:
        stats = verify_store.stats()
        # Only id2 should be in store (id1 failed)
        assert stats["memories"] == 1, f"Expected 1 memory in store, got {stats['memories']}"
    finally:
        verify_store.close()
