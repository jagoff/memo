from __future__ import annotations

import tempfile
from pathlib import Path

from memo.code_traceability import sync_ast_graph_links
from memo.graph import GraphStore


def test_code_ast_relations_in_graph_store() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "graph.db"
        store = GraphStore(db_path)

        store.upsert_code_ast_link(
            memory_id="mem_123",
            file_path="src/memo/recall_logic.py",
            symbol_name="render_recall_context",
            qualified_name="memo.recall_logic.render_recall_context",
            relation_type="modified",
            confidence=0.95,
        )

        links = store.get_ast_links_for_memory("mem_123")
        assert len(links) == 1
        assert links[0]["symbol_name"] == "render_recall_context"
        assert links[0]["file_path"] == "src/memo/recall_logic.py"

        memories = store.get_memories_for_ast_symbol("render_recall_context")
        assert memories == ["mem_123"]

        store.close()


def test_sync_ast_graph_links() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "graph.db"
        store = GraphStore(db_path)

        extra = {
            "code_refs": [
                {
                    "uri": "codegraph://repo1/sym1",
                    "file_path": "src/memo/graph.py",
                    "label": "GraphStore",
                    "qualified_name": "memo.graph.GraphStore",
                    "relation": "explicit",
                }
            ]
        }

        count = sync_ast_graph_links(store, "mem_456", extra)
        assert count == 1

        links = store.get_ast_links_for_memory("mem_456")
        assert len(links) == 1
        assert links[0]["symbol_name"] == "GraphStore"

        store.close()
