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
