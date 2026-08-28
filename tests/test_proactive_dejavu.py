from memo.proactive.detectors.dejavu import detect_dejavu
from memo.proactive.nudge import KIND_DEJAVU


class _FakeMem:
    def recurring_pattern_pairs(self, *, limit):
        return [("m1", "how do I configure the recall daemon")]


def test_dejavu_nudges_cite_matching_memory():
    ns = detect_dejavu(_FakeMem(), now="2026-07-21T00:00:00Z")
    assert len(ns) == 1
    n = ns[0]
    assert n.kind == KIND_DEJAVU
    assert n.evidence == ("m1",)
    assert "how do I configure the recall daemon" in n.title


def test_dejavu_empty_source_emits_no_nudge():
    class _EmptyMem:
        def recurring_pattern_pairs(self, *, limit):
            return []

    assert detect_dejavu(_EmptyMem(), now="2026-07-21T00:00:00Z") == []


def test_dejavu_guarded_returns_empty_on_error():
    class Boom:
        def recurring_pattern_pairs(self, *, limit):
            raise RuntimeError("boom")

    assert detect_dejavu(Boom(), now="2026-07-21T00:00:00Z") == []


def test_real_facade_recurring_pattern_pairs_cites_matching_memory(mock_memory):
    from memo.dashboard_logs import append_recall_log

    prompt = "always restart the recall daemon after a config change"
    rec = mock_memory.save(content=prompt, type_="note")
    append_recall_log(mock_memory.cfg.state_dir, prompt=prompt, hits=[])
    append_recall_log(mock_memory.cfg.state_dir, prompt=prompt, hits=[])

    pairs = mock_memory.recurring_pattern_pairs()

    assert (rec.id, prompt) in pairs


def test_real_facade_recurring_pattern_pairs_drops_unmatched_prompt(mock_memory):
    from memo.dashboard_logs import append_recall_log

    prompt = "a question memo has no memory for at all"
    append_recall_log(mock_memory.cfg.state_dir, prompt=prompt, hits=[])
    append_recall_log(mock_memory.cfg.state_dir, prompt=prompt, hits=[])

    pairs = mock_memory.recurring_pattern_pairs()

    assert pairs == []


def test_real_facade_recurring_pattern_pairs_bounds_search_calls(mock_memory):
    """When matches are sparse, the candidate loop must stop after
    `max(limit * 4, 20)` distinct queries instead of walking the whole
    recall-log tail and calling `search()` once per unmatched candidate."""
    from memo.dashboard_logs import append_recall_log

    # 30 distinct recurring prompts, none with a matching memory — far more
    # than the max_candidates cap for the default limit=5 (max(5*4, 20) = 20).
    for i in range(30):
        prompt = f"unmatched recurring question number {i}"
        append_recall_log(mock_memory.cfg.state_dir, prompt=prompt, hits=[])
        append_recall_log(mock_memory.cfg.state_dir, prompt=prompt, hits=[])

    call_count = 0
    real_search = mock_memory.search

    def _counting_search(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_search(*args, **kwargs)

    mock_memory.search = _counting_search  # type: ignore[method-assign]

    pairs = mock_memory.recurring_pattern_pairs(limit=5, min_count=2)

    assert pairs == []
    assert call_count <= 20


def test_real_facade_recurring_pattern_pairs_drops_prompt_sharing_no_term(mock_memory, monkeypatch):
    """On a populated corpus the `if hits` guard is dead code.

    `search()` always ranks *something* once the corpus is large enough for a
    hit to clear the similarity floor, so "a pattern with no matching memory is
    dropped" never fires in production — every repeated prompt, gibberish
    included, gets cited against whatever came back. The older empty-corpus
    test passes only because an empty corpus makes the guard trivially true.

    Force the production condition and require the pattern to be dropped on the
    strength of the citation instead.
    """
    from memo.dashboard_logs import append_recall_log

    mock_memory.save(
        content="always restart the recall daemon after a config change",
        type_="note",
    )
    prompt = "asdfgh qwerty zzz"
    append_recall_log(mock_memory.cfg.state_dir, prompt=prompt, hits=[])
    append_recall_log(mock_memory.cfg.state_dir, prompt=prompt, hits=[])

    # The fixture corpus is too small for gibberish to clear the floor on its
    # own, so pin `search` to the hit a real corpus would have returned.
    hits = mock_memory.search("restart recall daemon", limit=1, disable_reranker=True)
    assert hits, "fixture sanity: a matching query must return the seeded memory"
    monkeypatch.setattr(mock_memory, "search", lambda *a, **k: hits)

    assert mock_memory.recurring_pattern_pairs() == []
