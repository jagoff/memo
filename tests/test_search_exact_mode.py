"""mode=exact: strict keyword search with preconfigured tag/title boost.

`exact` is the precise-lookup mode — it never loosens an AND match into an
OR fallback (so a query term that isn't present yields nothing rather than
fuzzy partial hits), and it boosts the tags/title fields so a term living in
curated metadata outranks the same term buried in a long body.
"""

from __future__ import annotations


def test_exact_mode_does_not_or_fallback(mock_memory):
    # "zebra" exists, "giraffe" does not. bm25 loosens to OR and still finds
    # the zebra doc; exact mode keeps the strict AND and returns nothing.
    mock_memory.save(content="a zebra appears here", title="Z", tags=["t"])

    bm25 = mock_memory.search("zebra giraffe", mode="bm25", load_bodies=False)
    exact = mock_memory.search("zebra giraffe", mode="exact", load_bodies=False)

    assert any(r.title == "Z" for r in bm25), "bm25 OR-fallback should find zebra"
    assert exact == [], "exact must not loosen AND into OR"


def test_exact_mode_boosts_tag_over_body(mock_memory):
    # Same term: in A it lives in the tags, in B only in the body. Exact mode's
    # elevated tag weight should rank the tag match first.
    mock_memory.save(content="unrelated filler text", title="A", tags=["zebra"])
    mock_memory.save(content="zebra in the body only", title="B", tags=["misc"])

    results = mock_memory.search("zebra", mode="exact", load_bodies=False)
    titles = [r.title for r in results]
    assert "A" in titles and "B" in titles
    assert titles[0] == "A", f"tag match should rank first, got {titles}"
