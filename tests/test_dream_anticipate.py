"""dream_anticipate — gap surfacing + hot-query aggregation, no fabrication."""

from __future__ import annotations

from memo import dream_anticipate as da


class _Cfg:
    state_dir = "/tmp/unused"  # detect_gaps / read_recall_log are monkeypatched


def test_anticipate_aggregates_gaps_and_hot_queries(monkeypatch):
    monkeypatch.setattr(
        da, "detect_gaps",
        lambda sd, **k: [
            {"prompt": "how does X work", "count": 3},
            {"prompt": "where is Y", "count": 2},
        ],
    )
    monkeypatch.setattr(
        da, "read_recall_log",
        lambda sd, **k: [{"prompt": "deploy steps"}, {"prompt": "deploy steps"}, {"prompt": "auth flow"}],
    )
    res = da.anticipate(_Cfg(), mem=None, top_gaps=5, top_queries=5)
    assert [g["prompt"] for g in res["gaps"]] == ["how does X work", "where is Y"]
    assert res["hot_queries"][0] == "deploy steps"  # most frequent
    assert res["prewarmed"] == 0  # mem=None → no warming


def test_anticipate_never_raises_on_failure(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("log unreadable")

    monkeypatch.setattr(da, "detect_gaps", _boom)
    res = da.anticipate(_Cfg(), mem=None)
    assert "error" in res
    assert res["gaps"] == []


def test_anticipate_prewarms_when_mem_given(monkeypatch):
    monkeypatch.setattr(da, "detect_gaps", lambda sd, **k: [{"prompt": "g1", "count": 2}])
    monkeypatch.setattr(da, "read_recall_log", lambda sd, **k: [{"prompt": "hot1"}])
    warmed = []

    class _Emb:
        def embed_query(self, q):
            warmed.append(q)
            return [0.0]

    class _Mem:
        embedder = _Emb()

    res = da.anticipate(_Cfg(), mem=_Mem(), top_gaps=5, top_queries=5)
    assert res["prewarmed"] == 2  # g1 + hot1
    assert set(warmed) == {"g1", "hot1"}


def test_briefing_line():
    assert "no recurring gaps" in da.briefing_line({"gaps": []})
    line = da.briefing_line({"gaps": [{"prompt": "how does X work", "count": 3}, {"prompt": "b", "count": 2}]})
    assert "1 more" in line and "unmet gap" in line
