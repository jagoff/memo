from memo.chat.fusion import normalize_scores, rrf_fuse, source_dedup_key


def _src(id_: str, score: float, source: str = "memory", **kw) -> dict:
    return {
        "source": source,
        "id": id_,
        "title": id_,
        "type": "note",
        "score": score,
        "snippet": "x",
        **kw,
    }


def test_rrf_prefers_items_in_both_rankings() -> None:
    a = [_src("shared", 0.9), _src("only-a", 0.8)]
    b = [_src("shared", 0.7, source="vault"), _src("only-b", 0.6, source="vault")]
    fused = rrf_fuse([a, b])
    assert fused[0]["id"] == "shared"
    assert fused[0]["rrf_origins"] == [0, 1]
    assert {s["id"] for s in fused} == {"shared", "only-a", "only-b"}


def test_rrf_limit_and_empty() -> None:
    assert rrf_fuse([]) == []
    fused = rrf_fuse([[_src("a", 1.0), _src("b", 0.5)]], limit=1)
    assert len(fused) == 1


def test_dedup_key_precedence() -> None:
    assert source_dedup_key({"locator": "repo:x:a.md:1-2@abc"}).startswith("loc::")
    assert source_dedup_key({"id": "m1"}) == "id::m1"
    assert source_dedup_key({"title": " Foo "}) == "title::foo"


def test_normalize_minmax_per_group() -> None:
    out = normalize_scores([_src("a", 10.0), _src("b", 5.0), _src("c", 0.0)])
    by_id = {s["id"]: s for s in out}
    assert by_id["a"]["normalized_score"] == 1.0
    assert by_id["c"]["normalized_score"] == 0.0
    assert by_id["a"]["score_group"] == "memory"


def test_normalize_singleton_and_tight_cluster_are_neutral() -> None:
    single = normalize_scores([_src("a", 3.0)])
    assert single[0]["normalized_score"] == 0.5
    tight = normalize_scores([_src("a", 100.0), _src("b", 99.0)])  # span 1 < 100*0.15
    assert all(s["normalized_score"] == 0.5 for s in tight)
