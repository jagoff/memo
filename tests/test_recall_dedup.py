from dataclasses import dataclass

from memo import recall_logic as rl


@dataclass
class _Hit:
    id: str
    title: str
    body: str
    score: float


def test_collapses_near_duplicate_keeps_higher_score():
    a = _Hit("aaaaaaaa", "Deploy cutover en mac-work", "el cutover memflow a mac-work fue ok", 0.80)
    b = _Hit("bbbbbbbb", "Deploy cutover mac-work", "el cutover memflow a mac work fue ok", 0.72)
    c = _Hit("cccccccc", "BM25 spanish tokenizer", "fts5 wraps each token in phrase quotes", 0.70)
    out = rl.collapse_near_dups([a, b, c], threshold=0.7)
    ids = [h.id for h in out]
    assert "aaaaaaaa" in ids and "cccccccc" in ids
    assert "bbbbbbbb" not in ids  # near-dup of a, lower score → dropped
    assert len(out) == 2


def test_no_collapse_when_distinct():
    a = _Hit("aaaaaaaa", "Deploy cutover", "cutover memflow", 0.80)
    b = _Hit("bbbbbbbb", "Reranker batching", "batchear regresiona el 4B", 0.70)
    out = rl.collapse_near_dups([a, b], threshold=0.7)
    assert len(out) == 2
