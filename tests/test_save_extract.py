"""Write-time fact extraction (`extract_and_save_text` + `memo save --extract`).

mem0 ADD-model: an explicit save can decompose a raw blob into atomic facts
instead of storing one opaque record. These tests stub the helper LLM so they
run on any platform — they exercise the decomposition, tag propagation, and the
verbatim fallback policy, not a real MLX forward pass.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import memo.capture as capture_mod
from memo.capture import extract_and_save_text, extract_insights
from memo.cli import cli
from memo.config import Config
from memo.memory import Memory


@pytest.fixture
def mem_with_stub(tmp_cfg: Config, monkeypatch) -> Memory:
    """Real Memory with a 64-dim hash-bucket embedder stub (mirrors the
    capture-test fixture) so saves index without MLX."""
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=64,
    )

    def _stub_embed(self, inputs):
        out = []
        for s in inputs:
            v = [0.0] * 64
            v[sum(ord(c) for c in s) % 64] = 1.0
            out.append(v)
        return out

    def _stub_embed_query(self, query: str):
        v = [0.0] * 64
        v[sum(ord(c) for c in query) % 64] = 1.0
        return v

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed_query", _stub_embed_query)
    mem = Memory(cfg)
    yield mem
    mem.close()


def test_extract_decomposes_blob_into_atomic_facts(mem_with_stub, monkeypatch):
    cands = [
        {"title": "fact one", "body": "a" * 80, "type": "decision", "tags": []},
        {"title": "fact two", "body": "b" * 80, "type": "bug", "tags": ["x"]},
    ]
    monkeypatch.setattr(capture_mod, "extract_insights", lambda *a, **kw: cands)
    monkeypatch.setattr(capture_mod, "_passes_quality", lambda *a, **kw: True)
    monkeypatch.setattr(capture_mod, "find_near_duplicate", lambda *a, **kw: None)

    out = extract_and_save_text(mem_with_stub, mem_with_stub.cfg, "one blob of raw text")

    assert out["status"] == "extracted"
    assert len(out["saved"]) == 2
    # Each candidate became its own memory with its own type.
    types = {mem_with_stub.get(i).type for i in out["saved"]}
    assert types == {"decision", "bug"}


def test_extract_populates_temporal_fact_edges(mem_with_stub, monkeypatch):
    cands = [
        {
            "title": "memo uses temporal fact edges",
            "body": "memo now stores extracted atomic insights as temporal fact edges.",
            "type": "decision",
            "tags": ["memo", "temporal"],
        }
    ]
    monkeypatch.setattr(capture_mod, "extract_insights", lambda *a, **kw: cands)
    monkeypatch.setattr(capture_mod, "_passes_quality", lambda *a, **kw: True)
    monkeypatch.setattr(capture_mod, "find_near_duplicate", lambda *a, **kw: None)

    out = extract_and_save_text(mem_with_stub, mem_with_stub.cfg, "raw design note")
    rows = mem_with_stub.fact_edges.query(
        subject="memory",
        predicate="asserts",
        as_of="2999-01-01T00:00:00+00:00",
    )

    assert out["status"] == "extracted"
    assert out["facts"] == 1
    assert len(rows) == 1
    assert rows[0]["source_record_id"] == out["saved"][0]
    assert rows[0]["object"] == "memo uses temporal fact edges"
    assert rows[0]["provenance"]["extractor"] == "memo.capture"
    assert rows[0]["provenance"]["mode"] == "atomic-insight"
    assert rows[0]["metadata"]["type"] == "decision"
    assert rows[0]["metadata"]["tags"] == ["memo", "temporal"]


def test_extract_preserves_structured_fact_edges_from_helper(monkeypatch):
    payload = [
        {
            "title": "memo extraction emits fact edges",
            "body": "The capture extractor can emit structured temporal facts.",
            "type": "fact",
            "tags": ["memo", "capture", "temporal"],
            "fact_edges": [
                {"subject": "memo capture", "predicate": "emits", "object": "fact edges"}
            ],
        }
    ]

    class _Chat:
        def chat(self, **kwargs):
            return {"message": {"content": json.dumps(payload)}}

    monkeypatch.setenv("MEMO_REDACT_SECRETS", "0")

    out = extract_insights(_Chat(), "helper", "user", "assistant")

    assert out == [
        {
            "title": "memo extraction emits fact edges",
            "type": "fact",
            "body": "The capture extractor can emit structured temporal facts.",
            "tags": ["memo", "capture", "temporal"],
            "fact_edges": [
                {"subject": "memo capture", "predicate": "emits", "object": "fact edges"}
            ],
        }
    ]


def test_extract_saves_structured_temporal_fact_edges(mem_with_stub, monkeypatch):
    cands = [
        {
            "title": "memo capture records graph facts",
            "body": "The extraction path records concrete subject-predicate-object facts.",
            "type": "fact",
            "tags": ["memo", "graph"],
            "fact_edges": [
                {"subject": "memo capture", "predicate": "records", "object": "graph facts"},
                {"subject": "graph facts", "predicate": "support", "object": "temporal queries"},
            ],
        }
    ]
    monkeypatch.setattr(capture_mod, "extract_insights", lambda *a, **kw: cands)
    monkeypatch.setattr(capture_mod, "_passes_quality", lambda *a, **kw: True)
    monkeypatch.setattr(capture_mod, "find_near_duplicate", lambda *a, **kw: None)

    out = extract_and_save_text(mem_with_stub, mem_with_stub.cfg, "raw design note")
    rows = mem_with_stub.fact_edges.query(
        as_of="2999-01-01T00:00:00+00:00",
        include_inactive=True,
    )

    assert out["status"] == "extracted"
    assert out["facts"] == 2
    assert {(row["subject"], row["predicate"], row["object"]) for row in rows} == {
        ("memo capture", "records", "graph facts"),
        ("graph facts", "support", "temporal queries"),
    }
    assert all(row["source_record_id"] == out["saved"][0] for row in rows)
    assert all(row["provenance"]["extractor"] == "memo.capture" for row in rows)
    assert all(row["metadata"]["type"] == "fact" for row in rows)


def test_extract_merges_caller_tags_into_every_fact(mem_with_stub, monkeypatch):
    cands = [
        {"title": "fact one", "body": "a" * 80, "type": "note", "tags": []},
        {"title": "fact two", "body": "b" * 80, "type": "note", "tags": ["own"]},
    ]
    monkeypatch.setattr(capture_mod, "extract_insights", lambda *a, **kw: cands)
    monkeypatch.setattr(capture_mod, "_passes_quality", lambda *a, **kw: True)
    monkeypatch.setattr(capture_mod, "find_near_duplicate", lambda *a, **kw: None)

    out = extract_and_save_text(
        mem_with_stub, mem_with_stub.cfg, "blob", merge_tags=["project:demo"]
    )

    assert len(out["saved"]) == 2
    for rid in out["saved"]:
        assert "project:demo" in mem_with_stub.get(rid).tags
    # The candidate's own tag survives alongside the merged one.
    assert "own" in mem_with_stub.get(out["saved"][1]).tags


def test_extract_verbatim_fallback_when_nothing_extractable(mem_with_stub, monkeypatch):
    # Extractor finds nothing to atomize → explicit save must NOT vanish.
    monkeypatch.setattr(capture_mod, "extract_insights", lambda *a, **kw: [])

    out = extract_and_save_text(
        mem_with_stub,
        mem_with_stub.cfg,
        "an unstructured note the caller still wants kept",
        merge_tags=["project:demo"],
        title="kept note",
        type_="fact",
    )

    assert out["status"] == "verbatim"
    assert out["facts"] == 1
    assert len(out["saved"]) == 1
    rec = mem_with_stub.get(out["saved"][0])
    assert rec.title == "kept note"
    assert rec.type == "fact"
    assert "project:demo" in rec.tags
    assert "unstructured note" in rec.body


def test_extract_respects_dedup_no_verbatim_resave(mem_with_stub, monkeypatch):
    # Candidate found but it's a near-identical paraphrase → dropped. The blob
    # is NOT re-saved verbatim (the info is already in the corpus).
    cand = {"title": "dup", "body": "c" * 80, "type": "note", "tags": []}
    monkeypatch.setattr(capture_mod, "extract_insights", lambda *a, **kw: [cand])
    monkeypatch.setattr(capture_mod, "_passes_quality", lambda *a, **kw: True)
    monkeypatch.setattr(
        capture_mod,
        "find_near_duplicate",
        lambda *a, **kw: {"id": "x" * 32, "score": 0.99, "title": "dup"},
    )

    out = extract_and_save_text(mem_with_stub, mem_with_stub.cfg, "blob")

    assert out["status"] == "extracted"
    assert out["saved"] == []
    assert out["skipped_dup"] == 1


# ── CLI wiring ───────────────────────────────────────────────────────────────


def _cli_env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "d"),
        "MEMO_STATE_DIR": str(tmp_path / "s"),
        "MEMO_EMBEDDER_DIMS": "8",
    }


def _stub_cli_embedder(monkeypatch):
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, query: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


def test_cli_save_extract_flag(tmp_path, monkeypatch):
    _stub_cli_embedder(monkeypatch)
    monkeypatch.setattr(
        capture_mod,
        "extract_insights",
        lambda *a, **kw: [{"title": "t", "body": "d" * 80, "type": "note", "tags": []}],
    )
    monkeypatch.setattr(capture_mod, "_passes_quality", lambda *a, **kw: True)
    monkeypatch.setattr(capture_mod, "find_near_duplicate", lambda *a, **kw: None)

    r = CliRunner().invoke(
        cli, ["save", "raw blob text", "--extract", "--json"], env=_cli_env(tmp_path)
    )
    assert r.exit_code == 0, r.output
    summary = json.loads(r.output)
    assert summary["status"] == "extracted"
    assert len(summary["saved"]) == 1


def test_cli_save_extract_default_flag(tmp_path, monkeypatch):
    # MEMO_SAVE_EXTRACT=1 routes through extraction without the --extract flag.
    _stub_cli_embedder(monkeypatch)
    monkeypatch.setattr(capture_mod, "extract_insights", lambda *a, **kw: [])  # → verbatim
    env = {**_cli_env(tmp_path), "MEMO_SAVE_EXTRACT": "1"}

    r = CliRunner().invoke(cli, ["save", "a plain note", "--json"], env=env)
    assert r.exit_code == 0, r.output
    summary = json.loads(r.output)
    assert summary["status"] == "verbatim"
    assert len(summary["saved"]) == 1


# ── auto_project kwarg ────────────────────────────────────────────────────────


def test_extract_auto_project_false_skips_project_tag(mem_with_stub, monkeypatch):
    monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "1")
    monkeypatch.setenv("MEMO_PROJECT_TAG", "detected-repo")
    cands = [{"title": "fact one", "body": "a" * 80, "type": "note", "tags": []}]
    monkeypatch.setattr(capture_mod, "extract_insights", lambda *a, **kw: cands)
    monkeypatch.setattr(capture_mod, "_passes_quality", lambda *a, **kw: True)
    monkeypatch.setattr(capture_mod, "find_near_duplicate", lambda *a, **kw: None)

    out = extract_and_save_text(
        mem_with_stub, mem_with_stub.cfg, "blob", auto_project=False
    )

    rec = mem_with_stub.get(out["saved"][0])
    assert not any(t.startswith("project:") for t in rec.tags)


def test_extract_verbatim_fallback_honors_auto_project_false(mem_with_stub, monkeypatch):
    monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "1")
    monkeypatch.setenv("MEMO_PROJECT_TAG", "detected-repo")
    monkeypatch.setattr(capture_mod, "extract_insights", lambda *a, **kw: [])

    out = extract_and_save_text(
        mem_with_stub, mem_with_stub.cfg, "unextractable note", auto_project=False
    )

    assert out["status"] == "verbatim"
    rec = mem_with_stub.get(out["saved"][0])
    assert not any(t.startswith("project:") for t in rec.tags)
