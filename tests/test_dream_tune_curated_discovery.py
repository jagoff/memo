"""Regression: the tuner's curated no-regression gate must find its labels on
an installed runtime, not only in a dev checkout.

Found running memo as an end user: the nightly receipt reported
"200 harvested + 0 curated" every night. `_curated_raw` looked in
`state_dir/eval/` (nothing puts a copy there) and then at
`Path(__file__).parent.parent.parent / "eval"`, which only resolves to a repo
root from `src/memo/`. From `site-packages/memo/` the same arithmetic lands in
`lib/python3.14/eval/`, so no curated document was ever found — and
`curated_gate_min_sim` fails open (`{"ok": True, "reason": "no_curated_labels"}`).
The guard CLAUDE.md describes as rejecting a change that helps mined labels but
hurts the curated set was therefore inert on every real install.
"""

from __future__ import annotations

import json

from memo import dream_tune

CURATED = {
    "schema": "memo.eval_recall.labels.v1",
    "prompts": [{"prompt": "curated probe", "relevant": True, "expect_ids": ["deadbeef"]}],
}


def test_state_dir_labels_win(tmp_path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "regression_labels.json").write_text(json.dumps(CURATED), encoding="utf-8")

    assert dream_tune._curated_prompts(tmp_path) == CURATED["prompts"]


def test_packaged_labels_are_used_when_state_dir_has_none(tmp_path, monkeypatch) -> None:
    packaged = tmp_path / "wheel" / "regression_labels.json"
    packaged.parent.mkdir(parents=True)
    packaged.write_text(json.dumps(CURATED), encoding="utf-8")
    monkeypatch.setattr(dream_tune, "_packaged_curated_labels", lambda: packaged)
    # Kill the dev-checkout fallback so only the packaged copy can answer.
    monkeypatch.setattr(dream_tune, "__file__", str(tmp_path / "nowhere" / "dream_tune.py"))

    assert dream_tune._curated_prompts(tmp_path / "empty-state") == CURATED["prompts"]


def test_repo_checkout_still_resolves_its_committed_labels(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dream_tune, "_packaged_curated_labels", lambda: None)

    prompts = dream_tune._curated_prompts(tmp_path / "empty-state")

    assert prompts, "the repo-committed eval/regression_labels.json must still be found in dev"


def test_curated_labels_ship_in_the_wheel() -> None:
    """The force-include must stay in pyproject: dropping it silently returns
    the gate to failing open."""
    import tomllib
    from pathlib import Path

    pyproject = Path(dream_tune.__file__).resolve().parents[2] / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    force_include = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert force_include.get("eval/regression_labels.json") == (
        "memo/agent_assets/eval/regression_labels.json"
    )
