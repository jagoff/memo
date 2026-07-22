from __future__ import annotations

from memo.code_traceability import codegraph_uri


def test_memory_graph_trace_is_bidirectional(mem_with_stub, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_GRAPH_CODE_TRACE_ENABLED", "1")
    uri = codegraph_uri("memo-repo", "function:abc123")
    record = mem_with_stub.save(
        content="Changed the graph projection builder.",
        title="Projection builder change",
        extra={
            "code_refs": [
                {
                    "uri": uri,
                    "kind": "function",
                    "label": "rebuild",
                    "qualified_name": "GraphProjectionStore.rebuild",
                    "file_path": "src/memo/graph_projection.py",
                    "start_line": 600,
                    "end_line": 700,
                }
            ]
        },
    )
    mem_with_stub.rebuild_graph()

    by_memory = mem_with_stub.graph_trace(memory_id=record.id)
    by_code = mem_with_stub.graph_trace(code=uri)

    assert by_memory["available"] is True
    assert by_memory["memory_id"] == record.id
    assert by_memory["code_refs"][0]["uri"] == uri
    assert by_code["code_refs"][0]["uri"] == uri
    assert by_code["memories"][0]["id"] == record.id


def test_graph_trace_requires_exactly_one_direction(mem_with_stub) -> None:
    assert mem_with_stub.graph_trace() == {
        "available": False,
        "reason": "exactly_one_of_memory_id_or_code_required",
        "code_refs": [],
        "memories": [],
    }
    assert mem_with_stub.graph_trace(memory_id="x", code="y")["reason"] == (
        "exactly_one_of_memory_id_or_code_required"
    )


def test_graph_trace_reports_missing_projection(mem_with_stub) -> None:
    result = mem_with_stub.graph_trace(memory_id="missing")
    assert result["available"] is False
    assert result["reason"] == "projection_missing"
