"""Controlled fixture guarding the P3 recall units in the states they target.

The committed regression corpus (`eval/regression_labels.json`) is single-answer
and dispute-free, so it can't exercise features built for paraphrase-crowding or
contradiction. This seeds a small isolated index that DOES contain those states
and asserts, via the same `run_config` the gate uses, that:

  1. dedup-collapse (default ON as of v3.0.0) rescues a distinct fact that a
     cluster of near-duplicate paraphrases would otherwise crowd out of top-K.
  2. declare-disputes surfaces both sides of an OPEN contradiction pair that the
     contradict-penalty would otherwise demote (competing pairs are never
     penalised, so `open` is the operative status).

Real MLX embeddings are required (retrieval dynamics must be faithful), so this
is `requires_mlx` + `slow` and auto-skips on non-Apple-Silicon / CI.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from memo.config import Config
from memo.eval_recall import Cfg, LabelSet, Prompt, run_config
from memo.memory import Memory

_PARAPHRASES = [
    "The nightly deploy pipeline runs database migrations before restarting the API server.",
    "The nightly deploy pipeline runs database migrations prior to restarting the API server.",
    "The nightly deploy pipeline executes database migrations before restarting the API server.",
    "The nightly deploy pipeline runs database migrations before it restarts the API server.",
    "Our nightly deploy pipeline runs database migrations before restarting the API server.",
    "The nightly deploy pipeline runs the database migrations before restarting the API server.",
    "The nightly deploy pipeline runs database migrations then restarts the API server.",
]
_DISTINCT = (
    "The nightly deploy pipeline also purges the CDN cache and warms the "
    "Elasticsearch search index once the rollout finishes."
)
_FILLERS = [
    "The production database is backed up nightly to S3 with 30-day retention.",
    "The production database connection pool is capped at 200 connections.",
    "The production database runs on Amazon RDS in the us-east-1 region.",
    "Production database credentials are stored in AWS Secrets Manager.",
    "The production database has read replicas across two availability zones.",
    "Production database schema changes are reviewed before every release.",
]


@pytest.fixture
def real_mlx_memory(tmp_cfg: Config) -> Iterator[Memory]:
    """Release SQLite handles and Metal model/cache between slow cases."""
    mem = Memory(tmp_cfg)
    yield mem
    mem.close()


def _seed(mem: Memory) -> tuple[str, str, str]:
    """Seed the fixture; return (distinct_id, older_id, newer_id)."""
    for i, text in enumerate(_PARAPHRASES):
        mem.save(content=text, title=f"deploy migrations {i}", type_="fact")
    distinct_id = mem.save(content=_DISTINCT, title="deploy cache warm", type_="fact").id
    older_id = mem.save(
        content="The production database engine is PostgreSQL 15 running on Amazon RDS.",
        title="prod db pg",
        type_="fact",
    ).id
    newer_id = mem.save(
        content="The production database engine is MySQL 8 running on Amazon RDS.",
        title="prod db mysql",
        type_="fact",
    ).id
    for i, text in enumerate(_FILLERS):
        mem.save(content=text, title=f"prod db filler {i}", type_="fact")
    # OPEN pair (older saved first => older `updated`). Open is the operative
    # status: the penalty demotes {open, evolved}; disputes protects {competing,
    # open}; a competing pair is never penalised (so disputes is a no-op there).
    pid = mem.contradict_store.upsert_open(
        older_id,
        newer_id,
        relationship="contradicts",
        confidence=0.95,
        rationale="fixture: prod db engine PostgreSQL vs MySQL",
    )
    assert pid > 0
    return distinct_id, older_id, newer_id


def _recall_of(mem: Memory, prompt: Prompt) -> float:
    labels = LabelSet(prompts=[prompt])
    # vec/0.0 = live-default mode, permissive floor so the effect under test is
    # crowding/demotion, not the min-sim gate.
    cfg = Cfg("A vec/0.0/keep", "vec", 0.0, exclude_archived=False)
    return run_config(mem, cfg, 5, labels).recall_at_k


@pytest.mark.requires_mlx
@pytest.mark.slow
def test_dedup_collapse_rescues_distinct_fact_from_paraphrase_crowding(
    real_mlx_memory: Memory, monkeypatch
) -> None:
    mem = real_mlx_memory
    distinct_id, _older, _newer = _seed(mem)
    prompt = Prompt(
        "what does the nightly deploy pipeline do after the rollout finishes",
        relevant=True,
        expect_ids=[distinct_id],
    )

    monkeypatch.setenv("MEMO_RECALL_DEDUP_COLLAPSE", "0")
    recall_off = _recall_of(mem, prompt)

    monkeypatch.setenv("MEMO_RECALL_DEDUP_COLLAPSE", "1")
    recall_on = _recall_of(mem, prompt)

    # OFF: the 7 near-duplicate paraphrases crowd the distinct fact out of top-5.
    assert recall_off == 0.0
    # ON: collapsing the cluster surfaces the distinct fact.
    assert recall_on == 1.0


@pytest.mark.requires_mlx
@pytest.mark.slow
def test_declare_disputes_surfaces_both_sides_of_open_pair(
    real_mlx_memory: Memory, monkeypatch
) -> None:
    mem = real_mlx_memory
    _distinct, older_id, newer_id = _seed(mem)
    prompt = Prompt(
        "what engine does the production database run",
        relevant=True,
        expect_ids=[older_id, newer_id],
    )
    monkeypatch.setenv("MEMO_RECALL_DEDUP_COLLAPSE", "0")  # isolate from collapse

    # penalty ON, disputes OFF: the older (open) side is demoted below top-5.
    monkeypatch.setenv("MEMO_CONTRADICT_PENALTY_ENABLED", "1")
    monkeypatch.delenv("MEMO_DECLARE_DISPUTES", raising=False)
    recall_penonly = _recall_of(mem, prompt)

    # penalty ON, disputes ON: both sides of the dispute surface.
    monkeypatch.setenv("MEMO_DECLARE_DISPUTES", "1")
    recall_pendisp = _recall_of(mem, prompt)

    assert recall_pendisp > recall_penonly
    assert recall_pendisp == 1.0
