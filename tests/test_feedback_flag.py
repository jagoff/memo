"""Typed content-lifecycle feedback — `Memory.feedback_flag`.

outdated → reversible archive; wrong → archive (+ supersede link when a
replacement is given). Distinct from the ranking-only feedback_record path.
Stubbed 4-dim embedder — no MLX.
"""

from __future__ import annotations

import pytest

from memo import cli_mandate
from memo.config import Config
from memo.memory import Memory


@pytest.fixture
def mem_with_stub(tmp_cfg: Config, monkeypatch) -> Memory:
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
        reranker_enabled=False,
    )

    def _stub_embed(self, inputs):
        out = []
        for s in inputs:
            v = [0.0] * 4
            v[sum(ord(c) for c in s) % 4] = 1.0
            out.append(v)
        return out

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    mem = Memory(cfg)
    yield mem
    mem.close()


def test_outdated_archives_and_hides_from_search(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="alpha body", title="Alpha")
    assert rec.id in {h.id for h in mem_with_stub.search("alpha", limit=5)}

    res = mem_with_stub.feedback_flag(rec.id, kind="outdated")
    assert res["action"] == "archived"
    assert res["archived"] is True
    assert res["superseded_by"] is None
    # Archived → no longer surfaced by recall.
    assert rec.id not in {h.id for h in mem_with_stub.search("alpha", limit=5)}


def test_wrong_with_replacement_records_supersede(mem_with_stub: Memory):
    old = mem_with_stub.save(content="the api limit is 100 rpm", title="Old")
    new = mem_with_stub.save(content="the api limit is 500 rpm", title="New")
    res = mem_with_stub.feedback_flag(old.id, kind="wrong", superseded_by=new.id)
    assert res["action"] == "superseded"
    assert res["superseded_by"] == new.id
    assert res["archived"] is True


def test_short_prefix_resolves(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="alpha body", title="Alpha")
    res = mem_with_stub.feedback_flag(rec.id[:8], kind="outdated")
    assert res["source_id"] == rec.id


def test_bad_kind_rejected(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="alpha body", title="Alpha")
    with pytest.raises(ValueError, match="kind must be"):
        mem_with_stub.feedback_flag(rec.id, kind="irrelevant")


def test_unknown_source_id_rejected(mem_with_stub: Memory):
    with pytest.raises(ValueError, match="no memory matches"):
        mem_with_stub.feedback_flag("deadbeefdeadbeef", kind="outdated")


def test_unknown_replacement_rejected(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="alpha body", title="Alpha")
    with pytest.raises(ValueError, match="superseded_by"):
        mem_with_stub.feedback_flag(rec.id, kind="wrong", superseded_by="nope0000nope0000")


def test_mandate_text_mentions_a_correction_verb_that_exists_on_the_surface():
    # T4: the cross-client mandate tells non-hook agents to correct stale
    # memories rather than work around them. The verb it names must exist on
    # the profile those clients are installed at — `memo_feedback_flag` is
    # registered only on full/default, so the mandate names the lifecycle
    # tools instead (see tests/test_instruction_tool_names.py).
    assert "memo_invalidate" in cli_mandate.MANDATE_TEXT
    assert "memo_supersede" in cli_mandate.MANDATE_TEXT
