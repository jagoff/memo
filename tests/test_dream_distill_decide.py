import datetime as dt

from memo import dream_distill as dd


def _now() -> dt.datetime:
    return dt.datetime(2026, 7, 13, tzinfo=dt.UTC)


def _items():
    return [
        {
            "id": "a" * 32,
            "title": "Ta",
            "type": "decision",
            "tags": [],
            "path": "p",
            "updated": "2026-06-01",
            "emb": [],
        },
        {
            "id": "b" * 32,
            "title": "Tb",
            "type": "fact",
            "tags": [],
            "path": "p",
            "updated": "2026-06-01",
            "emb": [],
        },
        {
            "id": "c" * 32,
            "title": "Tc",
            "type": "note",
            "tags": [],
            "path": "p",
            "updated": "2026-06-01",
            "emb": [],
        },
    ]


def test_assemble_clusters_carries_maturity_and_drops_singletons():
    items = _items()
    health = {i["id"]: {"confidence": 0.9, "roi_score": 1.0} for i in items}
    support = {i["id"]: 4 for i in items}
    created = {i["id"]: "2026-06-01" for i in items}  # ~42 days old
    # one cluster of the first two, one singleton of the third
    clusters = dd.assemble_clusters(
        items,
        [[0, 1], [2]],
        health=health,
        support=support,
        created_by_id=created,
        min_cluster=2,
        now=_now(),
    )
    assert len(clusters) == 1  # singleton dropped
    cl = clusters[0]
    assert cl["ids"] == ["a" * 32, "b" * 32]
    assert cl["stats"].size == 2
    assert cl["stats"].mean_support == 4.0
    assert cl["stats"].min_age_days > 30


def test_decide_saves_mature_and_skips_immature():
    stats_mature = dd.cluster_maturity(
        [{"id": "a", "created": "2026-06-01", "confidence": 0.9, "support_count": 4}], now=_now()
    )
    stats_young = dd.cluster_maturity(
        [{"id": "b", "created": "2026-07-12", "confidence": 0.9, "support_count": 4}], now=_now()
    )
    clusters = [
        {"ids": ["a", "b"], "titles": ["A", "B"], "stats": stats_mature},
        {"ids": ["c", "d"], "titles": ["C", "D"], "stats": stats_young},
    ]

    def synth(cl):
        return {"title": "Distilled", "body": "the principle"}

    def never_exists(_phash):
        return False

    def mature(stats):
        return stats.min_age_days >= 14

    out = dd.decide_distillations(
        clusters,
        synthesize_fn=synth,
        exists_fn=never_exists,
        is_mature_fn=mature,
        dry_run=False,
        max_clusters=5,
    )
    statuses = {d["status"] for d in out}
    assert "save" in statuses
    assert "immature" in statuses
    saved = next(d for d in out if d["status"] == "save")
    assert saved["title"] == "Distilled"
    assert saved["provenance"] == ["a", "b"]
    assert saved["confidence"] == "high"  # conf 0.9, support 4


def test_decide_dedups_by_provenance():
    stats = dd.cluster_maturity(
        [{"id": "a", "created": "2026-06-01", "confidence": 0.9, "support_count": 4}], now=_now()
    )
    clusters = [{"ids": ["a", "b"], "titles": ["A", "B"], "stats": stats}]
    out = dd.decide_distillations(
        clusters,
        synthesize_fn=lambda cl: {"title": "x", "body": "y"},
        exists_fn=lambda _p: True,
        is_mature_fn=lambda s: True,
        dry_run=False,
        max_clusters=5,
    )
    assert out[0]["status"] == "skip_exists"


def test_decide_dry_run_saves_nothing():
    stats = dd.cluster_maturity(
        [{"id": "a", "created": "2026-06-01", "confidence": 0.9, "support_count": 4}], now=_now()
    )
    clusters = [{"ids": ["a", "b"], "titles": ["A", "B"], "stats": stats}]
    out = dd.decide_distillations(
        clusters,
        synthesize_fn=lambda cl: {"title": "x", "body": "y"},
        exists_fn=lambda _p: False,
        is_mature_fn=lambda s: True,
        dry_run=True,
        max_clusters=5,
    )
    assert out[0]["status"] == "would_save"
