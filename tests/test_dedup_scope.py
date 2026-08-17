"""`derived_save_scope()` demotes the near-duplicate save nag.

The dedup check in `write_ops.save` warns "consider `memo update` instead" when a
new memory is a near-duplicate of an existing one. That nudge is only actionable
for an interactive human — dream/consolidation batch saves produce near-dups by
design (the same run's consolidate pass merges them), so those saves run inside
`derived_save_scope()` and the warning drops to DEBUG.

The stub embedder returns one constant vector, so every save after the first is a
guaranteed near-duplicate — the test controls the *log level*, not the similarity
math.
"""

from __future__ import annotations

import logging

import pytest

from memo.config import Config
from memo.memory import Memory
from memo.memory.record import derived_save_scope

_DEDUP_MSG = "near-duplicate detected"


@pytest.fixture
def mem_const_embed(tmp_cfg: Config, monkeypatch) -> Memory:
    """Real Memory whose embedder maps every input to the same 8-dim vector, so
    any second save is a near-duplicate (cosine 1.0) of the first."""
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=8,
    )
    const = [1.0] + [0.0] * 7
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", lambda self, xs: [const for _ in xs])
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed_query", lambda self, q: const)
    mem = Memory(cfg)
    yield mem
    mem.close()


def _dedup_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if _DEDUP_MSG in r.getMessage()]


def test_dedup_warns_for_interactive_save(mem_const_embed, caplog, monkeypatch):
    # This test is about the WARN path specifically — MEMO_SAVE_ABSORB=1
    # (now the default) would silently rewrite the existing record instead
    # of warning, which is a different mechanism covered by test_save_absorb.py.
    monkeypatch.setenv("MEMO_SAVE_ABSORB", "0")
    mem_const_embed.save(content="seed body one", title="seed", type_="note")
    with caplog.at_level(logging.DEBUG, logger="memo.memory.record"):
        mem_const_embed.save(content="seed body two", title="dup", type_="note")

    recs = _dedup_records(caplog)
    assert recs, "expected a near-duplicate log for the interactive save"
    assert all(r.levelno == logging.WARNING for r in recs)


def test_dedup_demoted_to_debug_inside_derived_scope(mem_const_embed, caplog):
    mem_const_embed.save(content="seed body one", title="seed", type_="note")
    with caplog.at_level(logging.DEBUG, logger="memo.memory.record"):
        with derived_save_scope():
            mem_const_embed.save(content="seed body two", title="dup", type_="note")

    recs = _dedup_records(caplog)
    assert recs, "dedup check should still run inside the scope"
    assert all(r.levelno == logging.DEBUG for r in recs), (
        "derived_save_scope() must demote the near-duplicate nag to DEBUG"
    )


def test_scope_resets_after_exit(mem_const_embed, caplog):
    from memo.memory.record import in_derived_save_scope

    assert not in_derived_save_scope()
    with derived_save_scope():
        assert in_derived_save_scope()
    assert not in_derived_save_scope()


def test_apply_merge_suppresses_dedup_nag(mem_const_embed, caplog):
    """`apply_merge` (also reachable standalone via `memo consolidate`, outside
    dream's scope) must not nag: a merged record is a near-dup of its members by
    construction, so its save runs inside `derived_save_scope()`."""
    from memo.consolidation import AdvancedConsolidator, MergeProposal

    a = mem_const_embed.save(content="alpha body", title="a", type_="note")
    b = mem_const_embed.save(content="beta body", title="b", type_="note")
    cons = AdvancedConsolidator(mem_const_embed)
    proposal = MergeProposal(
        cluster_id=1,
        memory_ids=[a.id, b.id],
        merged_title="merged",
        merged_body="merged body",
        merge_strategy="synthesis",
        rationale="t",
        archived_ids=[b.id],
    )
    caplog.clear()  # drop the seed saves' own near-dup logs (const embedder)
    with caplog.at_level(logging.DEBUG, logger="memo.memory.record"):
        cons.apply_merge(proposal, dry_run=False)

    recs = _dedup_records(caplog)
    assert recs, "the merged save should hit the dedup check"
    assert all(r.levelno == logging.DEBUG for r in recs), (
        "apply_merge must run the merged save inside derived_save_scope()"
    )
