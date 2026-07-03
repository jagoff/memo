"""Single-source RankKnobs resolution — `knobs_from_flags`.

`_recall_logic` and the eval harness must resolve every ranking knob through
the same helper so they cannot diverge. The equality tests capture the knobs
`_recall_logic` actually passes to `rank_hits` and assert they are exactly
`knobs_from_flags(cwd=...)` under the same environment.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from memo.recall_logic import RankKnobs, apply_injection_filters, knobs_from_flags


def _capture_recall_logic_knobs(monkeypatch, tmp_path, *, cwd: str | None = None) -> RankKnobs:
    """Run `_recall_logic` with a stubbed rank_hits and return the knobs it built."""
    import memo.recall_logic as rl

    captured: dict[str, Any] = {}

    def fake_rank_hits(hits: list[Any], knobs: RankKnobs, **kw: Any) -> list[Any]:
        captured["knobs"] = knobs
        return []

    monkeypatch.setattr(rl, "rank_hits", fake_rank_hits)
    mem = SimpleNamespace(
        search=lambda *a, **k: [],
        embedder=SimpleNamespace(is_warm=True),
    )
    cfg = SimpleNamespace(state_dir=tmp_path)
    out, _cb = rl._recall_logic("what did we decide about the store", cwd=cwd, mem=mem, cfg=cfg)
    assert out == "{}"
    return captured["knobs"]


# ── knobs_from_flags == _recall_logic resolution (parametrized env combos) ──


@pytest.mark.parametrize(
    "env",
    [
        {},  # pure defaults (plus whatever the ambient env carries — both sides see it)
        {"MEMO_RECALL_TOP_K": "5", "MEMO_RECALL_MIN_SIM": "0.7"},
        {"MEMO_RECALL_MODE": "bm25", "MEMO_RECALL_MIN_BODY_CHARS": "0"},
        {
            "MEMO_RECALL_MMR_LAMBDA": "0.4",
            "MEMO_RECALL_SYNTHESIS_BOOST": "0.05",
            "MEMO_RECALL_PROJECT_BOOST": "0.5",
            "MEMO_RECALL_GLOBAL_BOOST": "0.2",
        },
        {"MEMO_RECALL_CONTEXTUAL": "1"},
        {"MEMO_PROJECT_TAG": "myproj"},  # + cwd → project_tag resolves in both
        {"MEMO_PROJECT_TAG": "myproj", "MEMO_RECALL_PROJECT_BOOST": "0"},  # boost=0 gates tag off
    ],
    ids=[
        "defaults",
        "topk-minsim",
        "mode-minbody",
        "boosts-mmr-synth",
        "contextual",
        "project",
        "project-gated",
    ],
)
def test_knobs_from_flags_matches_recall_logic(monkeypatch, tmp_path, env) -> None:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cwd = str(tmp_path) if "MEMO_PROJECT_TAG" in env else None

    expected = knobs_from_flags(cwd=cwd)
    actual = _capture_recall_logic_knobs(monkeypatch, tmp_path, cwd=cwd)

    assert actual == expected
    # Spot-check the env actually took effect (not two identical bugs).
    if "MEMO_RECALL_TOP_K" in env:
        assert expected.top_k == 5
    if "MEMO_RECALL_MIN_SIM" in env:
        assert expected.min_sim == 0.7
    if "MEMO_RECALL_MMR_LAMBDA" in env:
        assert expected.mmr_lambda == 0.4
        assert expected.synthesis_boost == 0.05
    if "MEMO_RECALL_CONTEXTUAL" in env:
        assert expected.contextual is True
    if env.get("MEMO_PROJECT_TAG") and env.get("MEMO_RECALL_PROJECT_BOOST") != "0":
        assert expected.project_tag == "project:myproj"
    if env.get("MEMO_RECALL_PROJECT_BOOST") == "0":
        assert expected.project_tag is None  # boost=0 gates project resolution off


def test_clean_env_resolves_registry_defaults(monkeypatch) -> None:
    """With no env set, resolution follows the FLAG registry defaults — which
    deliberately differ from the bare RankKnobs dataclass defaults on
    `contextual` (registry default ON, dataclass default OFF)."""
    for name in (
        "MEMO_RECALL_TOP_K",
        "MEMO_RECALL_MIN_SIM",
        "MEMO_RECALL_MIN_BODY_CHARS",
        "MEMO_RECALL_MODE",
        "MEMO_RECALL_PROJECT_BOOST",
        "MEMO_RECALL_GLOBAL_BOOST",
        "MEMO_RECALL_CONTEXTUAL",
        "MEMO_RECALL_MMR_LAMBDA",
        "MEMO_RECALL_SYNTHESIS_BOOST",
    ):
        monkeypatch.delenv(name, raising=False)
    knobs = knobs_from_flags()
    assert knobs == RankKnobs(contextual=True)
    assert RankKnobs() != knobs  # dataclass defaults are NOT the flag defaults


# ── precedence: explicit kwargs > flags; overrides > everything ─────────────


def test_explicit_kwargs_win_over_flags(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_RECALL_TOP_K", "9")
    monkeypatch.setenv("MEMO_RECALL_MODE", "hybrid")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.8")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "77")
    knobs = knobs_from_flags(top_k=2, mode="vec", min_sim=0.1, min_body_chars=0)
    assert knobs.top_k == 2
    assert knobs.mode == "vec"
    assert knobs.min_sim == 0.1
    assert knobs.min_body_chars == 0


def test_overrides_win_over_kwargs_and_flags(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_RECALL_MMR_LAMBDA", "0.4")
    knobs = knobs_from_flags(
        min_sim=0.3,
        overrides={"min_sim": 0.9, "mmr_lambda": 0.7, "synthesis_boost": 0.05},
    )
    assert knobs.min_sim == 0.9  # override beats the explicit kwarg
    assert knobs.mmr_lambda == 0.7  # override beats the env flag
    assert knobs.synthesis_boost == 0.05


def test_explicit_project_tag_wins_over_cwd_resolution(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MEMO_PROJECT_TAG", "envproj")
    knobs = knobs_from_flags(project_tag="project:pinned", cwd=str(tmp_path))
    assert knobs.project_tag == "project:pinned"


def test_no_cwd_no_project_tag(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_PROJECT_TAG", "envproj")
    assert knobs_from_flags().project_tag is None  # no cwd → tag never resolved


def test_unknown_override_field_raises() -> None:
    with pytest.raises(TypeError):
        knobs_from_flags(overrides={"not_a_knob": 1.0})


# ── tuned-overlay resolution (env > overlay > default) ──────────────────────


def test_knobs_pick_up_tuned_overlay(monkeypatch, tmp_path) -> None:
    from memo.tuned_overlay import write_overlay

    write_overlay(tmp_path, {"MEMO_RECALL_MIN_SIM": 0.65}, {"by": "test"})
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("MEMO_RECALL_MIN_SIM", raising=False)
    assert knobs_from_flags().min_sim == 0.65
    # An explicit env var still beats the overlay.
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.55")
    assert knobs_from_flags().min_sim == 0.55


# ── apply_injection_filters (hook's post-rank skip-below / gap trim) ─────────


def _hits(*scores: float) -> list[SimpleNamespace]:
    return [SimpleNamespace(id=f"h{i}", score=s) for i, s in enumerate(scores)]


def test_injection_filters_passthrough_when_off(monkeypatch) -> None:
    # Both flags have non-zero REGISTRY defaults (skip 0.45, gap 0.10) — pin
    # them to 0 to prove the pass-through path.
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0")
    monkeypatch.setenv("MEMO_RECALL_GAP_THRESHOLD", "0")
    hits = _hits(0.9, 0.5, 0.1)
    assert apply_injection_filters(hits) == hits


def test_injection_filters_skip_below_drops_all(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0.95")
    monkeypatch.setenv("MEMO_RECALL_GAP_THRESHOLD", "0")
    assert apply_injection_filters(_hits(0.9, 0.5)) == []


def test_injection_filters_skip_below_keeps_when_top_clears(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0.8")
    monkeypatch.setenv("MEMO_RECALL_GAP_THRESHOLD", "0")
    hits = _hits(0.9, 0.5)
    assert apply_injection_filters(hits) == hits


def test_injection_filters_gap_trims_to_top_hit(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0")
    monkeypatch.setenv("MEMO_RECALL_GAP_THRESHOLD", "0.2")
    hits = _hits(0.9, 0.5, 0.4)
    assert apply_injection_filters(hits) == hits[:1]
