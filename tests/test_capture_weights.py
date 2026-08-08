"""Citation-type distribution feedback (Q3 Mes 2, wave 2).

Covers `memo.capture_weights` (nightly stats + capture-time loader), the dream
`capture_weights` pass (`_run_capture_weights` + `memo dream run` wiring under
the MEMO_DREAM_EVAL_ENABLED gate), and the capture-time consult
(`reweight_ambiguous_type` + its `_extract_and_save` hook gated by
MEMO_CAPTURE_TYPE_FEEDBACK). Conventions follow test_capture_hygiene.py:
stubbed 64-dim embedder, isolated Config via tmp_cfg, monkeypatched module
globals so only the pass under test is in play.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import memo.capture as capture_mod
from memo.capture import reweight_ambiguous_type
from memo.capture_weights import (
    compute_type_citation_stats,
    load_type_weights,
    weights_path,
)
from memo.cli_dream import dream_cmd
from memo.cli_dream_passes import _run_capture_weights
from memo.config import Config
from memo.memory import Memory

# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mem_with_stub(tmp_cfg: Config, monkeypatch) -> Memory:
    """Real Memory with a 64-dim hash-bucket embedder stub (same shape as
    test_capture_hygiene.py's fixture)."""
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=64,
    )

    def _stub_embed(self, inputs):
        out = []
        for s in inputs:
            h = sum(ord(c) for c in s) % 64
            v = [0.0] * 64
            v[h] = 1.0
            out.append(v)
        return out

    def _stub_embed_query(self, query: str):
        h = sum(ord(c) for c in query) % 64
        v = [0.0] * 64
        v[h] = 1.0
        return v

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed_query", _stub_embed_query)
    mem = Memory(cfg)
    yield mem
    mem.close()


def _write_grounding(state_dir, rows: list[dict]) -> None:
    p = state_dir / "grounding.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _rows(recall_id: str, n: int, used_score: float) -> list[dict]:
    return [
        {
            "ts": f"2026-07-01T00:00:{i:02d}+00:00",
            "session_id": "s1",
            "turn": i,
            "recall_id": recall_id,
            "used_score": used_score,
            "method": "both",
        }
        for i in range(n)
    ]


def _write_weights(cfg: Config, weights: dict[str, float]) -> None:
    p = weights_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"computed_ts": "2026-07-02T00:00:00+00:00", "weights": weights}),
        encoding="utf-8",
    )


# ── compute_type_citation_stats: synthetic grounding.log + real store ────────


def test_compute_stats_joins_log_to_store_types(mem_with_stub):
    mem = mem_with_stub
    cfg = mem.cfg
    dec = mem.save(
        content="We use flock for the machine sync lock, one owner per machine.",
        title="sync lock ownership",
        type_="decision",
    )
    note = mem.save(
        content="General background about the recall daemon socket location.",
        title="daemon socket note",
        type_="note",
    )
    # decision: 6 recalled, all cited; note: 6 recalled, never cited.
    _write_grounding(cfg.state_dir, _rows(dec.id[:8], 6, 0.9) + _rows(note.id[:8], 6, 0.1))

    payload = compute_type_citation_stats(cfg, mem)
    assert payload["stats"]["decision"] == {"recalled": 6, "cited": 6, "rate": 1.0}
    assert payload["stats"]["note"] == {"recalled": 6, "cited": 0, "rate": 0.0}
    # center = 6/12 = 0.5 → decision 2.0, note 0.0 → clamped to 0.5.
    assert payload["weights"]["decision"] == 2.0
    assert payload["weights"]["note"] == 0.5

    raw = json.loads(weights_path(cfg).read_text(encoding="utf-8"))
    assert raw["computed_ts"]
    assert raw["weights"] == payload["weights"]


def test_compute_stats_min_observations_gets_neutral_weight(mem_with_stub):
    mem = mem_with_stub
    cfg = mem.cfg
    dec = mem.save(content="decision body here", title="d", type_="decision")
    bug = mem.save(content="bug body here", title="b", type_="bug")
    # decision qualifies (6 obs, rate 0.5); bug has only 2 obs — below the
    # 5-observation floor: no signal, no bias.
    _write_grounding(
        cfg.state_dir,
        _rows(dec.id[:8], 3, 0.9) + _rows(dec.id[:8], 3, 0.1) + _rows(bug.id[:8], 2, 0.9),
    )
    payload = compute_type_citation_stats(cfg, mem)
    assert payload["stats"]["bug"]["recalled"] == 2
    assert payload["weights"]["bug"] == 1.0
    # decision's rate equals the center (only qualified type) → exactly 1.0.
    assert payload["weights"]["decision"] == 1.0


def test_compute_stats_weights_are_clamped(mem_with_stub):
    mem = mem_with_stub
    cfg = mem.cfg
    dec = mem.save(content="decision body here", title="d", type_="decision")
    note = mem.save(content="note body here", title="n", type_="note")
    # center = 6/24 = 0.25 → decision raw 4.0 → clamped 2.0; note 0.0 → 0.5.
    _write_grounding(cfg.state_dir, _rows(dec.id[:8], 6, 0.9) + _rows(note.id[:8], 18, 0.0))
    payload = compute_type_citation_stats(cfg, mem)
    assert payload["weights"]["decision"] == 2.0
    assert payload["weights"]["note"] == 0.5


def test_compute_stats_no_citations_anywhere_all_neutral(mem_with_stub):
    mem = mem_with_stub
    cfg = mem.cfg
    dec = mem.save(content="decision body here", title="d", type_="decision")
    _write_grounding(cfg.state_dir, _rows(dec.id[:8], 6, 0.1))
    payload = compute_type_citation_stats(cfg, mem)
    assert payload["weights"] == {"decision": 1.0}  # zero center ⇒ no bias


def test_compute_stats_skips_unresolvable_and_short_ids(mem_with_stub):
    mem = mem_with_stub
    cfg = mem.cfg
    _write_grounding(
        cfg.state_dir,
        [
            *_rows("deadbeef", 6, 0.9),  # no such memory in the store
            {"recall_id": "abc", "used_score": 0.9},  # <8 chars
        ],
    )
    payload = compute_type_citation_stats(cfg, mem)
    assert payload["stats"] == {}
    assert payload["weights"] == {}


# ── load_type_weights ─────────────────────────────────────────────────────────


def test_load_missing_file_returns_empty(tmp_cfg):
    assert load_type_weights(tmp_cfg) == {}


def test_load_corrupt_file_returns_empty(tmp_cfg):
    p = weights_path(tmp_cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert load_type_weights(tmp_cfg) == {}


def test_load_wrong_shape_returns_empty(tmp_cfg):
    p = weights_path(tmp_cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"weights": [1, 2]}), encoding="utf-8")
    assert load_type_weights(tmp_cfg) == {}


def test_load_clamps_and_drops_non_numeric(tmp_cfg):
    _write_weights(
        tmp_cfg, {"decision": 9.0, "note": 0.01, "bug": "high", "fact": True, "preference": 1.3}
    )
    assert load_type_weights(tmp_cfg) == {"decision": 2.0, "note": 0.5, "preference": 1.3}


def test_load_roundtrips_computed_file(mem_with_stub):
    mem = mem_with_stub
    dec = mem.save(content="decision body here", title="d", type_="decision")
    _write_grounding(mem.cfg.state_dir, _rows(dec.id[:8], 6, 0.9))
    computed = compute_type_citation_stats(mem.cfg, mem)
    assert load_type_weights(mem.cfg) == computed["weights"]


# ── reweight_ambiguous_type (pure) ────────────────────────────────────────────

# One decision marker ("decided to"), no other markers.
_DEC_TEXT = "Sync locking\n\nWe decided to use flock for the machine sync lock."
# One decision marker + one bug marker — a genuine marker tie.
_TIE_TEXT = "Cache issue\n\nWe decided to purge it; the root cause was a stale cache."
_PLAIN_TEXT = "Daemon socket\n\nThe recall daemon listens on a unix socket in state_dir."


def test_reweight_retypes_uncorroborated_note_to_weighted_marker_type():
    assert reweight_ambiguous_type("note", _DEC_TEXT, {"decision": 1.5, "note": 0.8}) == "decision"


def test_reweight_empty_weights_is_noop():
    assert reweight_ambiguous_type("note", _DEC_TEXT, {}) == "note"


def test_reweight_neutral_weights_keep_claimed_type():
    # Winner must STRICTLY exceed the claimed type's weight — 1.0 vs 1.0 stays.
    assert reweight_ambiguous_type("note", _DEC_TEXT, {"decision": 1.0, "note": 1.0}) == "note"


def test_reweight_corroborated_claim_never_touched():
    # Claimed decision, own marker present — not ambiguous even if bug outweighs.
    assert (
        reweight_ambiguous_type("decision", _TIE_TEXT, {"bug": 2.0, "decision": 0.5}) == "decision"
    )


def test_reweight_no_markers_at_all_is_noop():
    assert reweight_ambiguous_type("note", _PLAIN_TEXT, {"decision": 2.0, "note": 0.5}) == "note"


def test_reweight_marker_tie_broken_by_weight():
    w = {"bug": 1.8, "decision": 1.2, "note": 1.0}
    assert reweight_ambiguous_type("note", _TIE_TEXT, w) == "bug"


def test_reweight_misclaimed_marked_type_switches():
    # Claimed fact with zero fact markers; decision markers present and heavier.
    assert reweight_ambiguous_type("fact", _DEC_TEXT, {"decision": 1.5, "fact": 1.0}) == "decision"


def test_reweight_lower_weighted_marker_type_does_not_switch():
    # Markers point at decision but grounding says decisions are cited LESS.
    assert reweight_ambiguous_type("note", _DEC_TEXT, {"decision": 0.6, "note": 1.0}) == "note"


# ── capture-time consult through _extract_and_save ────────────────────────────


def _cand(title: str, body: str, type_: str = "note") -> dict:
    return {"title": title, "body": body, "type": type_, "tags": []}


def _wire(monkeypatch, candidates: list[dict]) -> None:
    """Route _extract_and_save straight at `candidates`, bypassing LLM/quality/
    store-dedup so only the type-feedback consult is in play."""
    monkeypatch.setattr(capture_mod, "extract_insights", lambda *a, **kw: list(candidates))
    monkeypatch.setattr(capture_mod, "_passes_quality", lambda *a, **kw: True)
    monkeypatch.setattr(capture_mod, "find_near_duplicate", lambda *a, **kw: None)


def _spy_saves(monkeypatch, mem) -> list[dict]:
    saves: list[dict] = []
    orig_save = mem.save
    monkeypatch.setattr(mem, "save", lambda **kw: (saves.append(kw), orig_save(**kw))[1])
    return saves


_AMBIG_BODY = "We decided to use flock for the machine-level sync lock going forward."


def test_capture_flag_on_retypes_ambiguous_candidate(mem_with_stub, monkeypatch):
    monkeypatch.setenv("MEMO_CAPTURE_TYPE_FEEDBACK", "1")
    _write_weights(mem_with_stub.cfg, {"decision": 1.5, "note": 0.8})
    _wire(monkeypatch, [_cand("Sync locking approach", _AMBIG_BODY, type_="note")])
    saves = _spy_saves(monkeypatch, mem_with_stub)
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert len(out["saved"]) == 1
    assert saves[0]["type_"] == "decision"


def test_capture_flag_off_is_byte_identical(mem_with_stub, monkeypatch):
    # Weights file exists and favors decision — but the flag (default OFF)
    # means the classification is untouched.
    _write_weights(mem_with_stub.cfg, {"decision": 1.5, "note": 0.8})
    _wire(monkeypatch, [_cand("Sync locking approach", _AMBIG_BODY, type_="note")])
    saves = _spy_saves(monkeypatch, mem_with_stub)
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert len(out["saved"]) == 1
    assert saves[0]["type_"] == "note"


def test_capture_flag_on_without_weights_file_is_noop(mem_with_stub, monkeypatch):
    monkeypatch.setenv("MEMO_CAPTURE_TYPE_FEEDBACK", "1")
    _wire(monkeypatch, [_cand("Sync locking approach", _AMBIG_BODY, type_="note")])
    saves = _spy_saves(monkeypatch, mem_with_stub)
    capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert saves[0]["type_"] == "note"


def test_capture_flag_on_corroborated_claim_untouched(mem_with_stub, monkeypatch):
    monkeypatch.setenv("MEMO_CAPTURE_TYPE_FEEDBACK", "1")
    _write_weights(mem_with_stub.cfg, {"bug": 2.0, "decision": 0.5})
    _wire(monkeypatch, [_cand("Sync locking approach", _AMBIG_BODY, type_="decision")])
    saves = _spy_saves(monkeypatch, mem_with_stub)
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert saves[0]["type_"] == "decision"
    assert out["retyped"] == 0  # corroborated claim: consult ran but changed nothing


def test_capture_retyped_counter_counts_actual_type_changes(mem_with_stub, monkeypatch):
    monkeypatch.setenv("MEMO_CAPTURE_TYPE_FEEDBACK", "1")
    _write_weights(mem_with_stub.cfg, {"decision": 1.5, "note": 0.8})
    _wire(monkeypatch, [_cand("Sync locking approach", _AMBIG_BODY, type_="note")])
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert out["retyped"] == 1


def test_capture_retyped_zero_when_flag_off(mem_with_stub, monkeypatch):
    # Weights file exists and favors decision — but the flag (default OFF)
    # means the consult never runs and the counter stays 0.
    _write_weights(mem_with_stub.cfg, {"decision": 1.5, "note": 0.8})
    _wire(monkeypatch, [_cand("Sync locking approach", _AMBIG_BODY, type_="note")])
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert out["retyped"] == 0


# ── atomic write: tmp name unique per writer ──────────────────────────────────


def test_compute_stats_tmp_name_is_pid_suffixed(mem_with_stub, monkeypatch):
    """Two concurrent dream runs must not interleave writes into one tmp file —
    the staging name carries the writer's pid (same pattern as presence.py)."""
    import os

    mem = mem_with_stub
    dec = mem.save(content="decision body here", title="d", type_="decision")
    _write_grounding(mem.cfg.state_dir, _rows(dec.id[:8], 6, 0.9))

    seen: list[str] = []
    real_replace = os.replace

    def _spy(src, dst):
        seen.append(str(src))
        return real_replace(src, dst)

    monkeypatch.setattr("memo.capture_weights.os.replace", _spy)
    compute_type_citation_stats(mem.cfg, mem)
    assert seen, "atomic replace never ran"
    assert seen[0].endswith(f".{os.getpid()}.tmp")
    assert weights_path(mem.cfg).is_file()


# ── dream pass: _run_capture_weights + `memo dream run` wiring ────────────────


def test_run_capture_weights_receipt_fragment(mem_with_stub):
    mem = mem_with_stub
    dec = mem.save(content="decision body here", title="d", type_="decision")
    note = mem.save(content="note body here", title="n", type_="note")
    _write_grounding(mem.cfg.state_dir, _rows(dec.id[:8], 6, 0.9) + _rows(note.id[:8], 6, 0.1))
    frag = _run_capture_weights(mem.cfg, mem)
    assert frag == {"types": 2, "top": "decision:2"}
    assert weights_path(mem.cfg).is_file()


def test_run_capture_weights_empty_log(mem_with_stub):
    frag = _run_capture_weights(mem_with_stub.cfg, mem_with_stub)
    assert frag == {"types": 0, "top": None}


class _StubLifecycle:
    def enforce_forget_ttl(self, dry_run=False):
        return []


class _StubMem:
    lifecycle = _StubLifecycle()

    def search(self, query, limit, mode="vec", budget_ms=None):
        return []


_SKIPS = [
    "--skip-entities",
    "--skip-decay",
    "--skip-maintain",
    "--skip-orientation",
    "--skip-signal-gather",
    "--skip-prune-floor",
    "--skip-evict",
    "--skip-compress",
    "--skip-prewarm",
    "--skip-presynthesis",
]


def _dream_env(monkeypatch, tmp_path):
    state = tmp_path / "state"
    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setenv("MEMO_OUTCOME_RANKING_ENABLED", "0")
    monkeypatch.setattr("memo.cli_dream._get_memory", lambda cfg: _StubMem())
    return state


def _last_receipt(state):
    return json.loads((state / "dream" / "last.json").read_text(encoding="utf-8"))


def test_dream_run_writes_capture_weights_receipt(tmp_path, monkeypatch):
    state = _dream_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "memo.capture_weights.compute_type_citation_stats",
        lambda cfg, mem: {"weights": {"decision": 1.4, "note": 0.7}, "stats": {}},
    )
    res = CliRunner().invoke(dream_cmd, ["run", *_SKIPS])
    assert res.exit_code == 0, res.output
    receipt = _last_receipt(state)
    assert receipt["capture_weights"] == {"types": 2, "top": "decision:1.4"}
    assert not any(e.startswith("capture_weights:") for e in receipt["errors"])


def test_dream_run_capture_weights_error_lands_in_receipt(tmp_path, monkeypatch):
    state = _dream_env(monkeypatch, tmp_path)

    def _boom(cfg, mem):
        raise RuntimeError("grounding log busted")

    monkeypatch.setattr("memo.capture_weights.compute_type_citation_stats", _boom)
    res = CliRunner().invoke(dream_cmd, ["run", *_SKIPS])
    assert res.exit_code == 0, res.output
    receipt = _last_receipt(state)
    assert "capture_weights" not in receipt
    assert "capture_weights: RuntimeError: grounding log busted" in " | ".join(receipt["errors"])


def test_dream_run_dry_run_skips_capture_weights(tmp_path, monkeypatch):
    state = _dream_env(monkeypatch, tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(
        "memo.capture_weights.compute_type_citation_stats",
        lambda cfg, mem: (calls.append(1), {"weights": {}, "stats": {}})[1],
    )
    res = CliRunner().invoke(dream_cmd, ["run", "--dry-run", *_SKIPS])
    assert res.exit_code == 0, res.output
    assert calls == []
    assert not (state / "capture" / "type_weights.json").exists()


def test_dream_run_eval_flag_off_skips_capture_weights(tmp_path, monkeypatch):
    state = _dream_env(monkeypatch, tmp_path)
    monkeypatch.setenv("MEMO_DREAM_EVAL_ENABLED", "0")
    calls: list[int] = []
    monkeypatch.setattr(
        "memo.capture_weights.compute_type_citation_stats",
        lambda cfg, mem: (calls.append(1), {"weights": {}, "stats": {}})[1],
    )
    res = CliRunner().invoke(dream_cmd, ["run", *_SKIPS])
    assert res.exit_code == 0, res.output
    assert calls == []
    receipt = _last_receipt(state)
    assert "capture_weights" not in receipt
