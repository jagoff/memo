from memo.chat.dedup import collapse_near_duplicates, dedup_key, normalize_title, score_of


def _chunk(n: int, total: int, score: float, snippet: str) -> dict:
    return {
        "source": "memory",
        "id": f"m{n}",
        "type": "note",
        "score": score,
        "title": f"Proyecto X (§{n}/{total})",
        "snippet": snippet,
        "path": "notes/proyecto-x.md",
    }


def test_normalize_title_strips_chunk_marker() -> None:
    assert normalize_title("Proyecto X (§2/3)") == "proyecto x"
    assert normalize_title("  Plain  ") == "plain"


def test_dedup_key_components() -> None:
    s = _chunk(1, 2, 0.9, "snippet")
    source, title, path = dedup_key(s)
    assert source == "memory"
    assert title == "proyecto x"
    assert path == "notes/proyecto-x.md"


def test_chunks_collapse_and_merge_ordered() -> None:
    out = collapse_near_duplicates([_chunk(2, 3, 0.9, "parte dos"), _chunk(1, 3, 0.5, "parte uno")])
    assert len(out) == 1
    assert out[0]["snippet"] == "parte uno\n\nparte dos"
    assert out[0]["collapsed_variants"] == 1
    assert out[0]["id"] == "m2"  # survivor = mejor score


def test_untitled_rows_never_collapse() -> None:
    rows = [
        {"source": "memory", "id": "a", "title": "", "score": 1.0, "snippet": "x"},
        {"source": "memory", "id": "b", "title": "", "score": 0.5, "snippet": "y"},
    ]
    assert len(collapse_near_duplicates(rows)) == 2


def test_score_of_precedence() -> None:
    assert score_of({"score": 1.0, "normalized_score": 0.3}) == 0.3
    assert score_of({"score": 1.0, "normalized_score": 0.3, "rerank_score": 0.9}) == 0.9
    assert score_of({}) == 0.0


def test_distinct_docs_stay_separate() -> None:
    a = _chunk(1, 2, 0.9, "a")
    b = dict(_chunk(1, 2, 0.8, "b"), title="Otro doc (§1/2)", path="notes/otro.md")
    assert len(collapse_near_duplicates([a, b])) == 2
