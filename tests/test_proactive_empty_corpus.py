from pathlib import Path

from memo.proactive.engine import compute_routed
from memo.proactive.store import ProactiveStore


def test_empty_corpus_zero_noise(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMO_PROACTIVE_ENABLED", "1")
    s = ProactiveStore(tmp_path / "p.db")  # no candidates
    routed = compute_routed(s, now="2026-07-21T10:00:00Z", day="2026-07-21")
    assert routed.urgent is None and routed.digest == []
