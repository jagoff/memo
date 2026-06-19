"""Tests for Sleep-time Compute dream improvements (B/C/D/E/A)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from memo.store.store import VecStore


def _make_store(tmp_path: Path) -> VecStore:
    return VecStore(tmp_path / "test.db", dims=4)


def _insert_meta(cx, id_: str, type_: str = "fact", days_old: int = 10) -> None:
    cx.execute(
        "INSERT OR REPLACE INTO meta "
        "(id, title, type, tags, path, created, updated, body_hash) "
        "VALUES (?, ?, ?, ?, ?, datetime('now', ? || ' days'), datetime('now', ? || ' days'), ?)",
        (id_, f"title-{id_}", type_, "[]", f"/fake/{id_}.md", f"-{days_old}", f"-{days_old}", f"hash-{id_}"),
    )


# ---------------------------------------------------------------------------
# C: Co-recall graph edges
# ---------------------------------------------------------------------------

class TestCoRecallGraph:
    def test_record_co_recall_creates_pairs(self, tmp_path):
        from memo.graph import GraphStore

        g = GraphStore(tmp_path / "graph.db")
        n = g.record_co_recall(["aaa", "bbb", "ccc"])
        assert n == 3  # (aaa,bbb), (aaa,ccc), (bbb,ccc)

    def test_record_co_recall_less_than_2_is_noop(self, tmp_path):
        from memo.graph import GraphStore

        g = GraphStore(tmp_path / "graph.db")
        assert g.record_co_recall([]) == 0
        assert g.record_co_recall(["only_one"]) == 0

    def test_record_co_recall_increments_count(self, tmp_path):
        from memo.graph import GraphStore

        g = GraphStore(tmp_path / "graph.db")
        g.record_co_recall(["x", "y"])
        g.record_co_recall(["x", "y"])
        pairs = g.top_co_recalled(limit=10)
        assert len(pairs) == 1
        pair = pairs[0]
        assert {pair["id_a"], pair["id_b"]} == {"x", "y"}
        assert pair["count"] == 2

    def test_record_co_recall_order_independent(self, tmp_path):
        from memo.graph import GraphStore

        g = GraphStore(tmp_path / "graph.db")
        g.record_co_recall(["b", "a"])
        g.record_co_recall(["a", "b"])
        pairs = g.top_co_recalled()
        assert len(pairs) == 1
        assert pairs[0]["count"] == 2

    def test_top_co_recalled_sorted_by_count(self, tmp_path):
        from memo.graph import GraphStore

        g = GraphStore(tmp_path / "graph.db")
        g.record_co_recall(["a", "b"])
        g.record_co_recall(["a", "b"])
        g.record_co_recall(["c", "d"])
        pairs = g.top_co_recalled(limit=10)
        assert pairs[0]["count"] == 2
        assert pairs[1]["count"] == 1


# ---------------------------------------------------------------------------
# D: Eviction automation
# ---------------------------------------------------------------------------

class TestDreamEviction:
    def _make_mem_with_corpus(self, tmp_path, n: int):
        """Create a mock Memory object with n non-reference memorias."""
        store = _make_store(tmp_path)
        with store._conn as cx:
            for i in range(n):
                _insert_meta(cx, f"mem{i:03d}", type_="fact", days_old=30)
        mem = MagicMock()
        mem.store = store
        mem.lifecycle = MagicMock()
        return mem

    def test_eviction_skipped_when_under_budget(self, tmp_path):
        from memo.cli_dream import _run_eviction

        mem = self._make_mem_with_corpus(tmp_path, n=5)
        evicted = _run_eviction(mem, max_count=10, dry_run=False)
        assert evicted == []
        mem.lifecycle.archive_memoria.assert_not_called()

    def test_eviction_archives_excess_lfu(self, tmp_path):
        from memo.cli_dream import _run_eviction

        mem = self._make_mem_with_corpus(tmp_path, n=10)
        evicted = _run_eviction(mem, max_count=7, dry_run=False)
        assert len(evicted) == 3
        assert mem.lifecycle.archive_memoria.call_count == 3

    def test_eviction_dry_run_no_archive(self, tmp_path):
        from memo.cli_dream import _run_eviction

        mem = self._make_mem_with_corpus(tmp_path, n=10)
        evicted = _run_eviction(mem, max_count=5, dry_run=True)
        assert len(evicted) == 5
        mem.lifecycle.archive_memoria.assert_not_called()


# ---------------------------------------------------------------------------
# E: Pre-warm cache
# ---------------------------------------------------------------------------

class TestPrewarmQueries:
    def test_prewarm_embeds_recent_queries(self, tmp_path):
        from memo.cli_dream import _run_prewarm_queries

        # Write 5 recall.log entries
        log_path = tmp_path / "recall.log"
        entries = [
            {"ts": "2026-06-18T10:00:00+00:00", "prompt": f"query {i}", "hits": []}
            for i in range(5)
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries))

        cfg = MagicMock()
        cfg.state_dir = tmp_path

        mem = MagicMock()
        mem.embedder = MagicMock()
        mem.embedder.embed_query.return_value = [0.1] * 4

        result = _run_prewarm_queries(cfg, mem, n=3)
        assert result["queries_warmed"] == 3
        assert mem.embedder.embed_query.call_count == 3

    def test_prewarm_handles_empty_log(self, tmp_path):
        from memo.cli_dream import _run_prewarm_queries

        cfg = MagicMock()
        cfg.state_dir = tmp_path  # no recall.log

        mem = MagicMock()
        mem.embedder = MagicMock()

        result = _run_prewarm_queries(cfg, mem, n=10)
        assert result["queries_warmed"] == 0
        mem.embedder.embed_query.assert_not_called()

    def test_prewarm_deduplicates_queries(self, tmp_path):
        from memo.cli_dream import _run_prewarm_queries

        log_path = tmp_path / "recall.log"
        entries = [
            {"ts": "2026-06-18T10:00:00+00:00", "prompt": "same query", "hits": []}
            for _ in range(5)
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries))

        cfg = MagicMock()
        cfg.state_dir = tmp_path
        mem = MagicMock()
        mem.embedder = MagicMock()
        mem.embedder.embed_query.return_value = [0.1] * 4

        result = _run_prewarm_queries(cfg, mem, n=10)
        # 5 entries but all same query → 1 unique
        assert result["queries_warmed"] == 1
        assert mem.embedder.embed_query.call_count == 1


# ---------------------------------------------------------------------------
# B: Verbose compression
# ---------------------------------------------------------------------------

class TestVerboseCompression:
    def test_compress_skips_short_bodies(self, tmp_path):
        from memo.cli_dream import _run_compress

        store = _make_store(tmp_path)
        # Insert FTS entry with short body
        with store._conn as cx:
            _insert_meta(cx, "short01", type_="fact", days_old=10)
            cx.execute("INSERT INTO fts (id, title, tags, body) VALUES (?, ?, ?, ?)",
                       ("short01", "short title", "[]", "short body"))

        mem = MagicMock()
        mem.store = store
        mem.get.return_value = MagicMock(body="short body")

        results = _run_compress(mem, threshold=2000, dry_run=False)
        assert results == []

    def test_compress_dry_run_does_not_update(self, tmp_path):
        from memo.cli_dream import _run_compress

        long_body = "word " * 500  # 2500 chars
        store = _make_store(tmp_path)
        with store._conn as cx:
            _insert_meta(cx, "long01", type_="fact", days_old=10)
            cx.execute("INSERT INTO fts (id, title, tags, body) VALUES (?, ?, ?, ?)",
                       ("long01", "long title", "[]", long_body))

        mock_rec = MagicMock()
        mock_rec.body = long_body

        chat_out = {"message": {"content": "Two sentence summary here."}}
        mock_chat = MagicMock()
        mock_chat.chat.return_value = chat_out

        mem = MagicMock()
        mem.store = store
        mem.get.return_value = mock_rec

        with patch("memo.memory.record.chat_with_timeout", return_value=chat_out):
            results = _run_compress(mem, threshold=100, dry_run=True)

        # dry_run → no update call
        mem.update.assert_not_called()
        assert len(results) == 1
        assert results[0]["id"] == "long01"
        assert results[0]["original_len"] > 100


# ---------------------------------------------------------------------------
# A: Query-prediction pre-synthesis (unit test — mostly integration-free)
# ---------------------------------------------------------------------------

class TestPresynthesisQueries:
    def test_presynthesis_empty_log_returns_empty(self, tmp_path):
        from memo.cli_dream import _run_presynthesis

        cfg = MagicMock()
        cfg.state_dir = tmp_path  # no recall.log

        mem = MagicMock()
        mem.search.return_value = []

        result = _run_presynthesis(cfg, mem, top_n=5, dry_run=True)
        assert result == []

    def test_presynthesis_picks_top_queries(self, tmp_path):
        from memo.cli_dream import _run_presynthesis

        log_path = tmp_path / "recall.log"
        entries = []
        for _ in range(5):
            entries.append({"ts": "2026-06-18T10:00:00+00:00", "prompt": "popular query", "hits": []})
        entries.append({"ts": "2026-06-18T10:00:00+00:00", "prompt": "rare query", "hits": []})
        log_path.write_text("\n".join(json.dumps(e) for e in entries))

        cfg = MagicMock()
        cfg.state_dir = tmp_path

        hits = [MagicMock(id=f"mem{i}") for i in range(5)]
        mem = MagicMock()
        mem.search.return_value = hits
        mem.synthesize_cross_cluster.return_value = [{"title": "synth", "saved": True}]

        result = _run_presynthesis(cfg, mem, top_n=1, dry_run=True)
        # top_n=1 → only "popular query" (count=5) processed
        assert len(result) == 1
        assert result[0]["query"] == "popular query"
        assert result[0]["synthesized"] == 1
