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
    monkeypatch.delenv("MEMO_DREAM_VALIDITY_EXTRACT_ENABLED", raising=False)

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
