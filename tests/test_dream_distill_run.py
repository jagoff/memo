import memo.dream_distill as dd


class _Store:
    def __init__(self, ids):
        self._ids = ids

    def get_health_batch(self, ids):
        return {i: {"confidence": 0.9, "roi_score": 1.0} for i in ids}

    def get_support_batch(self, ids):
        return {i: 4 for i in ids}

    def get_batch(self, ids):
        return [{"id": i, "created": "2026-06-01"} for i in ids]  # ~42 days old


class _FakeMem:
    """Enough of Memory for run_distill: clustering + save + search stubs."""

    def __init__(self):
        self._ids = ["a" * 32, "b" * 32, "c" * 32]
        self.store = _Store(self._ids)
        self.saved: list[dict] = []

    def _pull_embeddings(self, *, exclude_types=None):
        return [
            {"id": i, "title": f"T{i[:1]}", "type": "decision", "tags": [], "path": "p",
             "updated": "2026-06-01", "emb": [1.0, 0.0]}
            for i in self._ids
        ]

    def _greedy_cluster(self, items, threshold):
        return [[0, 1, 2]]  # one cluster of all three

    def search(self, q, *, limit=5, disable_reranker=True):
        return []  # nothing exists yet

    def _ensure_chat(self):
        return object()

    def save(self, **kwargs):
        self.saved.append(kwargs)

        class _Rec:
            id = "distilled-id"

        return _Rec()


def test_run_distill_disabled_by_default(monkeypatch, tmp_cfg):
    monkeypatch.delenv("MEMO_DREAM_DISTILL_ENABLED", raising=False)
    res = dd.run_distill(tmp_cfg, _FakeMem())
    assert res["status"] == "disabled"


def test_run_distill_saves_mature_cluster(monkeypatch, tmp_cfg):
    monkeypatch.setenv("MEMO_DREAM_DISTILL_ENABLED", "1")
    monkeypatch.setattr(dd, "_llm_distill", lambda mem, cl: {"title": "Principle", "body": "the distilled insight"})
    mem = _FakeMem()
    res = dd.run_distill(tmp_cfg, mem, min_cluster=2, min_support=2, min_age_days=14)
    assert res["status"] == "done"
    assert any(d["status"] == "saved" for d in res["distilled"])
    # saved as type=synthesis synthesis_kind=distillation with source provenance
    assert len(mem.saved) == 1
    saved = mem.saved[0]
    assert saved["type_"] == "synthesis"
    assert saved["extra"]["synthesis_kind"] == "distillation"
    assert set(saved["extra"]["synthesis_sources"]) == set(mem._ids)
    assert "[distill " in saved["content"]  # dedup marker present
    assert saved["extra"]["synthesis_confidence"] == "high"


def test_run_distill_success_status_is_saved_not_save(monkeypatch, tmp_cfg):
    """Regression for the dream-status/progress-line undercount bug: a real
    successful save must be countable by the "saved" counters used in
    cli_dream.py (`dream status` + the nightly progress line). This goes
    through the REAL decide_distillations (no hard-coded fake decision) so it
    proves the producer — not just a test double — emits "saved"."""
    monkeypatch.setenv("MEMO_DREAM_DISTILL_ENABLED", "1")
    monkeypatch.setattr(dd, "_llm_distill", lambda mem, cl: {"title": "Principle", "body": "the distilled insight"})
    mem = _FakeMem()
    res = dd.run_distill(tmp_cfg, mem, min_cluster=2, min_support=2, min_age_days=14)
    assert res["status"] == "done"
    assert mem.saved  # a real save happened
    saved_decisions = [d for d in res["distilled"] if d.get("status") == "saved"]
    assert len(saved_decisions) == 1
    # the "save" status (pre-fix vocabulary) must never leak into the receipt
    assert all(d.get("status") != "save" for d in res["distilled"])
    # the counters in cli_dream.py (`dream status` + nightly progress line) key
    # on "saved" — this is what makes a real successful distillation countable.
    countable = sum(1 for d in res["distilled"] if d.get("status") == "saved")
    assert countable == 1


def test_run_distill_reversible_never_mutates_sources(monkeypatch, tmp_cfg):
    """The pass may ONLY save. It must not call any source-mutating store op."""
    monkeypatch.setenv("MEMO_DREAM_DISTILL_ENABLED", "1")
    monkeypatch.setattr(dd, "_llm_distill", lambda mem, cl: {"title": "P", "body": "b"})
    mem = _FakeMem()
    # If run_distill tried to supersede/archive/delete a source, these attrs would
    # be accessed. They don't exist on _FakeMem, so any such call raises — and
    # run_distill would surface an error instead of "done". Assert clean save-only.
    res = dd.run_distill(tmp_cfg, mem, min_cluster=2, min_support=2, min_age_days=14)
    assert res["status"] == "done"
    assert mem.saved  # a save happened
    # no method other than the stubbed ones was needed -> reversible by construction


def test_run_distill_dry_run_saves_nothing(monkeypatch, tmp_cfg):
    monkeypatch.setenv("MEMO_DREAM_DISTILL_ENABLED", "1")
    monkeypatch.setattr(dd, "_llm_distill", lambda mem, cl: {"title": "P", "body": "b"})
    mem = _FakeMem()
    res = dd.run_distill(tmp_cfg, mem, min_cluster=2, dry_run=True)
    assert res["status"] == "done"
    assert mem.saved == []
    assert any(d["status"] == "would_save" for d in res["distilled"])


def test_run_distill_skips_immature_cluster(monkeypatch, tmp_cfg):
    monkeypatch.setenv("MEMO_DREAM_DISTILL_ENABLED", "1")
    monkeypatch.setattr(dd, "_llm_distill", lambda mem, cl: {"title": "P", "body": "b"})
    mem = _FakeMem()
    # require 999 days of age -> nothing mature
    res = dd.run_distill(tmp_cfg, mem, min_cluster=2, min_age_days=999)
    assert res["status"] == "done"
    assert mem.saved == []
    assert all(d["status"] in ("immature", "skip_exists") for d in res["distilled"])


def test_run_distill_llm_failure_swallowed_no_crash(monkeypatch, tmp_cfg):
    """_llm_distill returning None (LLM error/timeout/bad JSON) must not crash
    run_distill — it surfaces as a synth_failed decision, no save, no raise."""
    monkeypatch.setenv("MEMO_DREAM_DISTILL_ENABLED", "1")
    monkeypatch.setattr(dd, "_llm_distill", lambda mem, cl: None)
    mem = _FakeMem()
    res = dd.run_distill(tmp_cfg, mem, min_cluster=2, min_support=2, min_age_days=14)
    assert res["status"] == "done"
    assert mem.saved == []
    assert any(d["status"] == "synth_failed" for d in res["distilled"])


def test_run_distill_llm_raises_swallowed_no_crash(monkeypatch, tmp_cfg):
    """chat_with_timeout propagates non-timeout errors (per its own contract);
    _llm_distill raising must still be swallowed by run_distill's outer guard."""
    monkeypatch.setenv("MEMO_DREAM_DISTILL_ENABLED", "1")

    def _boom(mem, cl):
        raise RuntimeError("llm exploded")

    monkeypatch.setattr(dd, "_llm_distill", _boom)
    mem = _FakeMem()
    res = dd.run_distill(tmp_cfg, mem, min_cluster=2, min_support=2, min_age_days=14)
    assert res["status"] == "error"
    assert "llm exploded" in res["error"]
    assert mem.saved == []


def test_run_distill_pull_embeddings_exception_swallowed(monkeypatch, tmp_cfg):
    """Any unexpected exception from the real store/clustering path must be
    swallowed into the receipt (status=error), never raised out."""
    monkeypatch.setenv("MEMO_DREAM_DISTILL_ENABLED", "1")

    class _BoomMem(_FakeMem):
        def _pull_embeddings(self, *, exclude_types=None):
            raise RuntimeError("boom")

    res = dd.run_distill(tmp_cfg, _BoomMem())
    assert res["status"] == "error"
    assert "boom" in res["error"]
