"""Dream integration — nightly proactive-candidate refresh (Task 11).

`_run_proactive_refresh` wraps `refresh_candidates` for the dream pipeline:
guarded (failures land in `receipt["errors"]`, never raised out) and records
`receipt["proactive"] = {"candidates": N}` on success.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memo.proactive.engine import refresh_candidates
from memo.proactive.store import ProactiveStore

pytestmark = pytest.mark.resource_hygiene


class _FakeMem:
    def superseded_pairs(self):
        return [("old1", "new1", "use X")]

    def open_loops(self, limit):
        return [("m9", "finish study")]


def test_refresh_writes_candidates(tmp_path: Path):
    with ProactiveStore(tmp_path / "p.db") as store:
        n = refresh_candidates(_FakeMem(), store, now="2026-07-21T00:00:00Z")
        assert n == 2
        assert len(store.active_candidates("2026-07-21T01:00:00Z")) == 2


def test_dream_pass_guarded(tmp_path: Path):
    from memo.cli_dream_passes import _run_proactive_refresh

    receipt: dict = {"errors": []}
    _run_proactive_refresh(_FakeMem(), tmp_path / "p.db", receipt, now="2026-07-21T00:00:00Z")
    assert receipt["proactive"]["candidates"] == 2
    assert receipt["errors"] == []


def test_dream_pass_records_error_never_raises(tmp_path: Path):
    from memo.cli_dream_passes import _run_proactive_refresh

    # A db_path that is itself an existing directory makes sqlite3.connect()
    # raise inside ProactiveStore.__init__ — exercises the outer guard.
    bad_db_path = tmp_path / "p.db"
    bad_db_path.mkdir()

    receipt: dict = {"errors": []}
    _run_proactive_refresh(_FakeMem(), bad_db_path, receipt, now="2026-07-21T00:00:00Z")

    assert receipt["errors"], "expected the sqlite3 failure to land in receipt['errors']"
    assert "proactive" not in receipt
