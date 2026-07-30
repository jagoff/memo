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


def test_code_change_impact_ranks_linked_memory(mem_with_stub, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_GRAPH_CODE_TRACE_ENABLED", "1")
    uri = codegraph_uri("memo-repo", "function:impact")
    record = mem_with_stub.save(
        content="Changed impact-sensitive code.",
        title="Impact-linked memory",
        extra={
            "code_refs": [
                {
                    "uri": uri,
                    "kind": "function",
                    "label": "impact",
                    "qualified_name": "impact",
                    "file_path": "src/impact.py",
                    "code_evidence": {"schema": "memo.code_evidence.v1"},
                }
            ]
        },
    )
    mem_with_stub.rebuild_graph()
    monkeypatch.setattr(
        "memo.session_sources.gather_git_state",
        lambda cwd: {"modified_files": ["src/impact.py"]},
    )
    monkeypatch.setattr(
        "memo.code_impact.code_change_impact",
        lambda cwd, changed_files, depth=1: {
            "available": True,
            "reason": None,
            "repo_root": str(cwd),
            "changed_files": list(changed_files),
            "depth": depth,
            "symbols": [
                {
                    "stable_symbol_id": "function:impact",
                    "file_path": "src/impact.py",
                    "distance": 0,
                }
            ],
            "impacted_paths": ["src/impact.py"],
            "code_evidence": {"schema": "memo.code_evidence.v1"},
            "limitations": [],
        },
    )

    result = mem_with_stub.code_change_impact("/tmp/repo")

    assert result["available"] is True
    assert result["memories"][0]["id"] == record.id
    assert result["memories"][0]["distance"] == 0
    assert result["memories"][0]["code_refs"][0]["code_evidence"]["schema"] == (
        "memo.code_evidence.v1"
    )


def test_code_context_pack_ranks_linked_memory(mem_with_stub, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_GRAPH_CODE_TRACE_ENABLED", "1")
    uri = codegraph_uri("memo-repo", "function:context")
    record = mem_with_stub.save(
        content="The context builder owns architecture projection.",
        title="Context-linked memory",
        extra={
            "code_refs": [
                {
                    "uri": uri,
                    "kind": "function",
                    "label": "context",
                    "qualified_name": "context",
                    "file_path": "src/context.py",
                }
            ]
        },
    )
    mem_with_stub.rebuild_graph()
    monkeypatch.setattr(
        "memo.code_context.build_code_context_pack",
        lambda cwd, **kwargs: {
            "schema": "memo.code_context_pack.v1",
            "available": True,
            "reason": None,
            "provider": {"name": "codegraph"},
            "mode": kwargs["mode"],
            "findings": [
                {
                    "kind": "hotspot",
                    "id": "hotspot:function:context",
                    "data": {
                        "stable_symbol_id": "function:context",
                        "file_path": "src/context.py",
                    },
                    "evidence_uris": [uri],
                }
            ],
            "code_evidence": {
                "schema": "memo.code_evidence.v1",
                "repo_id": "memo-repo",
            },
        },
    )

    result = mem_with_stub.code_context_pack("/tmp/repo", mode="verify")

    assert result["available"] is True
    assert result["memories"][0]["id"] == record.id
    assert result["memories"][0]["matched_code_refs"] == 1
    assert result["projection_version"]
