"""code_drift_lines — nightly code-drift outcome surfaced in the briefing."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memo.briefing import code_drift_lines


def _cfg(tmp_path: Path) -> Any:
    return SimpleNamespace(state_dir=tmp_path)


def _write_receipt(state_dir: Path, code_drift: dict[str, Any] | None) -> None:
    d = state_dir / "dream"
    d.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {"ts": 123.0}
    if code_drift is not None:
        receipt["code_drift"] = code_drift
    (d / "last.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_outdated_renders_line(tmp_path: Path) -> None:
    _write_receipt(
        tmp_path,
        {
            "status": "ok",
            "scanned": 5,
            "outdated": [
                {"id": "a", "refs_dead": 1, "refs_total": 1},
                {"id": "b", "refs_dead": 2, "refs_total": 2},
            ],
            "partial": [],
        },
    )
    lines = code_drift_lines(_cfg(tmp_path))
    joined = "\n".join(lines)
    assert "code-drift" in joined
    assert "2 memorias archivadas" in joined
    assert "memo dream status" in joined


def test_partial_and_repaired_counts(tmp_path: Path) -> None:
    _write_receipt(
        tmp_path,
        {
            "status": "ok",
            "scanned": 4,
            "outdated": [],
            "partial": [{"id": "a", "refs_dead": 1, "refs_total": 3}],
            "repaired": [
                {"id": "b", "from": "x", "to": "y"},
                {"id": "c", "from": "u", "to": "v"},
                {"id": "d", "from": "p", "to": "q"},
            ],
        },
    )
    joined = "\n".join(code_drift_lines(_cfg(tmp_path)))
    assert "0 memorias archivadas" in joined
    assert "1 parciales" in joined
    assert "3 reparadas" in joined


def test_clean_ok_run_is_empty(tmp_path: Path) -> None:
    _write_receipt(
        tmp_path,
        {"status": "ok", "scanned": 7, "outdated": [], "partial": []},
    )
    assert code_drift_lines(_cfg(tmp_path)) == []


def test_status_disabled_is_empty(tmp_path: Path) -> None:
    _write_receipt(tmp_path, {"status": "disabled"})
    assert code_drift_lines(_cfg(tmp_path)) == []


def test_status_aborted_is_empty(tmp_path: Path) -> None:
    _write_receipt(tmp_path, {"status": "aborted", "reason": "codegraph_db_missing"})
    assert code_drift_lines(_cfg(tmp_path)) == []


def test_missing_receipt_is_empty(tmp_path: Path) -> None:
    assert code_drift_lines(_cfg(tmp_path)) == []


def test_receipt_without_code_drift_key_is_empty(tmp_path: Path) -> None:
    _write_receipt(tmp_path, None)
    assert code_drift_lines(_cfg(tmp_path)) == []


def test_corrupt_receipt_is_empty(tmp_path: Path) -> None:
    d = tmp_path / "dream"
    d.mkdir(parents=True)
    (d / "last.json").write_text("{corrupt", encoding="utf-8")
    assert code_drift_lines(_cfg(tmp_path)) == []


def test_flag_off_never_opens_the_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_receipt(
        tmp_path,
        {"status": "ok", "scanned": 1, "outdated": [{"id": "a"}], "partial": []},
    )
    monkeypatch.setenv("MEMO_BRIEFING_CODE_DRIFT", "0")

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("flag off must not touch the receipt")

    monkeypatch.setattr(Path, "open", _boom)
    assert code_drift_lines(_cfg(tmp_path)) == []
