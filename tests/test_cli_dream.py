"""`memo dream run` — bi-temporal validity-extract wiring in the orchestrator.

The pass itself is unit-tested in ``test_dream_validity_extract.py``; here we
prove the ``dream_run`` seam: it fires only when
``MEMO_DREAM_VALIDITY_EXTRACT_ENABLED`` is set (and not a dry run), copies the
pass fragment into the receipt, and wraps any (defensive) raise into
``receipt["errors"]`` instead of aborting the pipeline.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from click.testing import CliRunner

from memo.cli_dream import dream_cmd

_SKIPS = [
    "--skip-orientation",
    "--skip-signal-gather",
    "--skip-entities",
    "--skip-decay",
    "--skip-prune-floor",
    "--skip-evict",
    "--skip-compress",
    "--skip-prewarm",
    "--skip-presynthesis",
]


def _run(mock_memory) -> dict:
    with patch("memo.cli_dream._get_memory", return_value=mock_memory):
        res = CliRunner().invoke(dream_cmd, ["run", "--json", *_SKIPS])
    assert res.exit_code == 0, res.output
    out = res.output
    return json.loads(out[out.index("{") :])


def test_dream_run_invokes_validity_pass_when_flag_on(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_VALIDITY_EXTRACT_ENABLED", "1")

    called: list[bool] = []

    def fake_pass(mem, receipt, *, limit=50, dry_run=False):
        called.append(True)
        receipt["validity_extract"] = {"scanned": 2, "updated": [{"id": "abc"}]}

    monkeypatch.setattr("memo.cli_dream._run_validity_extract", fake_pass)

    receipt = _run(mock_memory)
    assert called == [True]
    assert receipt["validity_extract"]["updated"] == [{"id": "abc"}]
    assert not any("validity_extract" in e for e in receipt["errors"])


def test_dream_run_skips_validity_pass_when_flag_off(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_VALIDITY_EXTRACT_ENABLED", "0")

    called: list[bool] = []
    monkeypatch.setattr(
        "memo.cli_dream._run_validity_extract",
        lambda *a, **k: called.append(True),
    )

    receipt = _run(mock_memory)
    assert called == []
    assert "validity_extract" not in receipt


def test_dream_run_wraps_validity_pass_error_into_receipt(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_VALIDITY_EXTRACT_ENABLED", "1")

    def boom(*a, **k):
        raise RuntimeError("validity kaput")

    monkeypatch.setattr("memo.cli_dream._run_validity_extract", boom)

    receipt = _run(mock_memory)
    assert any("validity_extract" in e and "validity kaput" in e for e in receipt["errors"])


def test_dream_run_invokes_vector_hygiene_pass_when_flag_on(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_VECTOR_HYGIENE_ENABLED", "1")

    called: list[bool] = []

    def fake_pass(cfg, mem, *, dry_run=False):
        called.append(True)
        return {"status": "done", "cache_packed": 1, "cache_pruned": 2}

    monkeypatch.setattr("memo.dream_vector.run_vector_hygiene", fake_pass)

    receipt = _run(mock_memory)
    assert called == [True]
    assert receipt["vector_hygiene"]["cache_pruned"] == 2
    assert not any("vector_hygiene" in e for e in receipt["errors"])


def test_dream_run_skips_vector_hygiene_pass_when_flag_off(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_VECTOR_HYGIENE_ENABLED", "0")

    called: list[bool] = []
    monkeypatch.setattr(
        "memo.dream_vector.run_vector_hygiene",
        lambda *a, **k: called.append(True),
    )

    receipt = _run(mock_memory)
    assert called == []
    assert "vector_hygiene" not in receipt


def test_dream_run_wraps_vector_hygiene_pass_error_into_receipt(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_VECTOR_HYGIENE_ENABLED", "1")
    monkeypatch.setattr(
        "memo.dream_vector.run_vector_hygiene",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("hygiene kaput")),
    )

    receipt = _run(mock_memory)
    assert any("vector_hygiene" in e and "hygiene kaput" in e for e in receipt["errors"])


def test_dream_run_reports_vector_hygiene_pass_internal_error_status(mock_memory, monkeypatch):
    """The pass can also fail "softly" (returns status=error) rather than
    raising; dream_run must fold that into receipt["errors"] too."""
    monkeypatch.setenv("MEMO_DREAM_VECTOR_HYGIENE_ENABLED", "1")
    monkeypatch.setattr(
        "memo.dream_vector.run_vector_hygiene",
        lambda *a, **k: {"status": "error", "error": "soft failure"},
    )

    receipt = _run(mock_memory)
    assert any("vector_hygiene" in e and "soft failure" in e for e in receipt["errors"])


def test_dream_run_invokes_vector_views_pass_when_flag_on(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_VECTOR_VIEWS_ENABLED", "1")

    called: list[bool] = []

    def fake_pass(cfg, mem, *, night_cap=1000, dry_run=False):
        called.append(True)
        return {"status": "done", "indexed": 5, "backlog": 5, "errors": 0}

    monkeypatch.setattr("memo.dream_vector_views.run_title_view_pass", fake_pass)

    receipt = _run(mock_memory)
    assert called == [True]
    assert receipt["vector_views"]["indexed"] == 5
    assert not any("vector_views" in e for e in receipt["errors"])


def test_dream_run_skips_vector_views_pass_when_flag_off(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_VECTOR_VIEWS_ENABLED", "0")

    called: list[bool] = []
    monkeypatch.setattr(
        "memo.dream_vector_views.run_title_view_pass",
        lambda *a, **k: called.append(True),
    )

    receipt = _run(mock_memory)
    assert called == []
    assert "vector_views" not in receipt


def test_dream_run_wraps_vector_views_pass_error_into_receipt(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_VECTOR_VIEWS_ENABLED", "1")
    monkeypatch.setattr(
        "memo.dream_vector_views.run_title_view_pass",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("views kaput")),
    )

    receipt = _run(mock_memory)
    assert any("vector_views" in e and "views kaput" in e for e in receipt["errors"])


def test_dream_run_persists_receipt_when_memory_load_crashes(monkeypatch, tmp_path):
    """F1: a hard crash before the per-pass guards (here Memory construction)
    must STILL persist a receipt carrying the error — otherwise
    `dream status`/doctor keep showing the last good night while the pipeline
    is silently dead."""
    from memo.config import Config
    from memo.dream_utils import _state_path

    state = tmp_path / "state"
    data = tmp_path / "data"
    state.mkdir()
    data.mkdir()
    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(data))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")

    def boom(_cfg):
        raise RuntimeError("memory kaput")

    monkeypatch.setattr("memo.cli_dream._get_memory", boom)

    # NOT a dry-run: the receipt is only persisted on a real run.
    res = CliRunner().invoke(dream_cmd, ["run", "--json", *_SKIPS])
    assert res.exit_code == 0, res.output
    receipt = json.loads(res.output[res.output.index("{") :])
    assert any("pipeline" in e and "memory kaput" in e for e in receipt["errors"])

    # Persisted to disk so status/doctor observe the failed night.
    last = _state_path(Config.from_env()) / "last.json"
    assert last.exists(), "a crashed pipeline must still write last.json"
    persisted = json.loads(last.read_text(encoding="utf-8"))
    assert any("memory kaput" in e for e in persisted["errors"])
