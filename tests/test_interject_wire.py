from types import SimpleNamespace

from memo import interject as ij


class _Pair:
    def __init__(self, a, b):
        self.memory_id_a, self.memory_id_b = a, b


class _Store:
    def __init__(self, disputed_pairs):
        self._pairs = disputed_pairs

    def pairs_for_ids(self, ids, *, status="open"):
        return [p for p in self._pairs if p.memory_id_a in ids or p.memory_id_b in ids]


class _Mem:
    def __init__(self, pairs):
        self.contradict_store = _Store(pairs)


def _hit(id_, title, type_, score):
    return SimpleNamespace(id=id_, title=title, type=type_, score=score)


def test_evaluate_shadow_logs_but_does_not_render_when_off(monkeypatch, tmp_cfg):
    monkeypatch.delenv("MEMO_INTERJECT_ENABLED", raising=False)  # OFF
    monkeypatch.setattr(ij, "recalibrated_band_of_score", lambda sd, s: "high", raising=False)
    # stub the calibration band to "high" via the confidence_calibration seam
    import memo.confidence_calibration as cc
    monkeypatch.setattr(cc, "recalibrated_band", lambda sd, band: "high")

    mem = _Mem([_Pair("a" * 32, "z" * 32)])
    hits = [_hit("a" * 32, "Use vec mode", "decision", 0.9)]
    banner = ij.evaluate_and_render(
        tmp_cfg, mem, prompt="switch to hybrid instead", hits=hits, sim_threshold=0.6
    )
    assert banner is None  # OFF -> never renders
    rows = ij.read_shadow(tmp_cfg.state_dir)
    assert len(rows) == 1 and rows[0]["rendered"] is False  # but shadow-counted


def test_evaluate_renders_when_on_and_in_budget(monkeypatch, tmp_cfg):
    monkeypatch.setenv("MEMO_INTERJECT_ENABLED", "1")
    monkeypatch.setenv("MEMO_INTERJECT_MAX_PER_SESSION", "1")
    monkeypatch.setenv("MEMO_SESSION_ID", "sess-x")
    import memo.confidence_calibration as cc
    monkeypatch.setattr(cc, "recalibrated_band", lambda sd, band: "high")

    mem = _Mem([_Pair("a" * 32, "z" * 32)])
    hits = [_hit("a" * 32, "Use vec mode", "decision", 0.9)]
    banner = ij.evaluate_and_render(
        tmp_cfg, mem, prompt="revert to vec instead", hits=hits, sim_threshold=0.6
    )
    assert banner is not None and banner.startswith(ij.INTERJECT_HEADER)
    # second fire in the same session is over budget -> shadow-only
    banner2 = ij.evaluate_and_render(
        tmp_cfg, mem, prompt="switch again instead", hits=hits, sim_threshold=0.6
    )
    assert banner2 is None
    rows = ij.read_shadow(tmp_cfg.state_dir)
    assert [r["rendered"] for r in rows] == [False, True]  # newest_first: 2nd suppressed, 1st shown


def test_evaluate_never_raises_on_broken_store(monkeypatch, tmp_cfg):
    monkeypatch.setenv("MEMO_INTERJECT_ENABLED", "1")

    class _Broken:
        @property
        def contradict_store(self):
            raise RuntimeError("store down")

    hits = [_hit("a" * 32, "Use vec mode", "decision", 0.9)]
    # must not raise; no dispute -> no candidates -> None
    assert ij.evaluate_and_render(
        tmp_cfg, _Broken(), prompt="switch instead", hits=hits, sim_threshold=0.6
    ) is None


def test_evaluate_never_touches_store_without_reversal_signal(monkeypatch, tmp_cfg):
    """Regression: with no reversal signal in the prompt, guard_candidates is
    empty, so evaluate_and_render must short-circuit BEFORE reading
    contradict_store — no new hot-path store read on an ordinary turn."""
    monkeypatch.setenv("MEMO_INTERJECT_ENABLED", "1")

    class _StoreAccessed(RuntimeError):
        pass

    class _TripwireStore:
        def pairs_for_ids(self, ids, *, status="open"):
            raise _StoreAccessed("contradict_store touched with no guard candidate")

    class _Mem:
        contradict_store = _TripwireStore()

    hits = [_hit("a" * 32, "Use vec mode", "decision", 0.9)]
    # plain question, no reversal wording -> guard_candidates() == [] -> must
    # never reach the store.
    banner = ij.evaluate_and_render(
        tmp_cfg, _Mem(), prompt="what did we decide about vec mode?", hits=hits, sim_threshold=0.6
    )
    assert banner is None
    rows = ij.read_shadow(tmp_cfg.state_dir)
    assert rows == []  # nothing shadow-logged either -- no candidate at all
