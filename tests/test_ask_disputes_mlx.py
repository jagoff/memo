"""End-to-end MLX integration for dispute-aware ask (MEMO_ASK_DISPUTES).

The unit suite (`tests/test_ask_disputes.py`) stubs retrieval, the chat client,
and the pair lookup. This fixture runs the REAL pipeline — MLX embeddings drive
retrieval, the contradiction pair lives in the real contradict_store, and the
real 7B LLM answers — asserting the two spec guarantees end-to-end
(docs/SPECS/2026-07-28-ask-dispute-aware-design.md):

  1. Retrieved sources that belong to an OPEN contradiction pair come back
     annotated (`disputed_by` per source + top-level `disputed` map) while
     clean hits stay unannotated.
  2. When the corpus offers ONLY disputed evidence for the question, the
     deterministic gate abstains (`abstained == "disputed"`) with the contested
     message naming both sides — regardless of what the LLM generated.

Seeded state mirrors `tests/test_trust_states_fixture.py`: the PostgreSQL-15
vs MySQL-8 prod-db-engine pair, upserted OPEN in the contradict store.
Real model loads make this `requires_mlx` + `slow` (auto-skips off Apple
Silicon via the conftest gate).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from memo.config import Config
from memo.memory import Memory

_OLDER = "The production database engine is PostgreSQL 15 running on Amazon RDS."
_NEWER = "The production database engine is MySQL 8 running on Amazon RDS."
_QUESTION = "what engine does the production database run"
_FILLERS = [
    "The production database is backed up nightly to S3 with 30-day retention.",
    "The production database connection pool is capped at 200 connections.",
    "Production database credentials are stored in AWS Secrets Manager.",
]


@pytest.fixture
def real_mlx_memory(tmp_cfg: Config) -> Iterator[Memory]:
    """Release SQLite handles and Metal model/cache between slow cases."""
    mem = Memory(tmp_cfg)
    yield mem
    mem.close()


def _seed_open_pair(mem: Memory) -> tuple[str, str]:
    """Seed the contradicting fact pair and its OPEN entry; return (older, newer)."""
    older_id = mem.save(content=_OLDER, title="prod db pg", type_="fact").id
    newer_id = mem.save(content=_NEWER, title="prod db mysql", type_="fact").id
    pair_id = mem.contradict_store.upsert_open(
        older_id,
        newer_id,
        relationship="contradicts",
        confidence=0.95,
        rationale="fixture: prod db engine PostgreSQL vs MySQL",
    )
    assert pair_id > 0
    return older_id, newer_id


@pytest.mark.requires_mlx
@pytest.mark.slow
def test_ask_abstains_when_corpus_offers_only_disputed_evidence(
    real_mlx_memory: Memory,
) -> None:
    """Corpus = ONLY the two sides of the open pair → whatever the real LLM
    cites is a subset of the disputed set, so the deterministic gate MUST
    replace the answer with the contested abstention naming both ids."""
    mem = real_mlx_memory
    older_id, newer_id = _seed_open_pair(mem)

    out = mem.ask(_QUESTION, include_repos=False)

    # Real retrieval surfaced both sides, each annotated with its disputer.
    by_id = {s["id"]: s for s in out["sources"]}
    assert by_id[older_id]["disputed_by"] == [newer_id]
    assert by_id[newer_id]["disputed_by"] == [older_id]
    assert out["disputed"] == {older_id: [newer_id], newer_id: [older_id]}

    # Deterministic contested abstention, independent of the LLM's wording.
    assert out["abstained"] == "disputed"
    assert "couldn't find" in out["answer"]  # journey_check abstain marker
    assert older_id[:8] in out["answer"] and newer_id[:8] in out["answer"]
    assert "memo contradict" in out["answer"]


@pytest.mark.requires_mlx
@pytest.mark.slow
def test_ask_annotates_only_the_disputed_sources_amid_clean_corpus(
    real_mlx_memory: Memory,
) -> None:
    """With clean fillers alongside the pair, annotation stays surgical:
    both pair sides carry `disputed_by`, fillers don't, and the top-level
    `disputed` map contains exactly the pair (both directions)."""
    mem = real_mlx_memory
    older_id, newer_id = _seed_open_pair(mem)
    filler_ids = {
        mem.save(content=text, title=f"prod db filler {i}", type_="fact").id
        for i, text in enumerate(_FILLERS)
    }

    out = mem.ask(_QUESTION, include_repos=False)

    by_id = {s["id"]: s for s in out["sources"]}
    # Both engine facts are the closest matches for the question — they must
    # be in the k=5 sources the LLM saw, annotated with each other.
    assert by_id[older_id]["disputed_by"] == [newer_id]
    assert by_id[newer_id]["disputed_by"] == [older_id]
    # Clean fillers never gain the key; the map holds exactly the pair.
    assert all("disputed_by" not in by_id[fid] for fid in filler_ids if fid in by_id)
    assert out["disputed"] == {older_id: [newer_id], newer_id: [older_id]}
    # The gate may or may not abstain here (the LLM can lean on a clean
    # filler) — but if it does, it must be the dispute abstention.
    assert out.get("abstained") in (None, "disputed")
