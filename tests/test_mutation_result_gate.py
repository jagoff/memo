from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_mutation_results.py"


def _mutation_gate() -> ModuleType:
    assert SCRIPT.is_file(), "scripts/check_mutation_results.py must exist"
    spec = importlib.util.spec_from_file_location("memo_mutation_result_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata_payload(results: dict[str, int | None]) -> dict[str, object]:
    return {
        "exit_code_by_key": results,
        "type_check_error_by_key": {},
        "durations_by_key": {},
        "estimated_durations_by_key": {},
    }


def _write_payload(root: Path, payload: object, *, filename: str = "example.py.meta") -> Path:
    path = root / "src" / "memo" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_meta(
    root: Path,
    results: dict[str, int | None],
    *,
    filename: str = "example.py.meta",
) -> Path:
    return _write_payload(root, _metadata_payload(results), filename=filename)


def _write_baseline(
    root: Path,
    blocked: dict[str, str],
    *,
    overrides: dict[str, object] | None = None,
) -> Path:
    gate = _mutation_gate()
    payload = gate.blocking_summary(blocked)
    payload.update(overrides or {})
    path = root / "mutation-baseline.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_blocking_mutants_reports_survivors_and_incomplete_results(tmp_path: Path) -> None:
    _write_meta(
        tmp_path,
        {
            "memo.example.x_killed__mutmut_1": 1,
            "memo.example.x_survived__mutmut_2": 0,
            "memo.example.x_unchecked__mutmut_3": None,
            "memo.example.x_no_tests__mutmut_4": 33,
            "memo.example.x_no_tests_pytest__mutmut_5": 5,
            "memo.example.x_suspicious__mutmut_6": 35,
            "memo.example.x_unknown__mutmut_7": 4,
            "memo.example.x_interrupted__mutmut_8": 2,
        },
    )

    assert _mutation_gate().blocking_mutants(tmp_path) == {
        "memo.example.x_interrupted__mutmut_8": "interrupted",
        "memo.example.x_no_tests__mutmut_4": "no-tests",
        "memo.example.x_no_tests_pytest__mutmut_5": "no-tests",
        "memo.example.x_survived__mutmut_2": "survived",
        "memo.example.x_suspicious__mutmut_6": "suspicious",
        "memo.example.x_unchecked__mutmut_3": "not-checked",
        "memo.example.x_unknown__mutmut_7": "suspicious",
    }


def test_gate_exits_nonzero_for_blocking_mutant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_meta(tmp_path, {"memo.example.x_value__mutmut_1": 0})

    assert _mutation_gate().main([str(tmp_path)]) == 1
    assert "survived: memo.example.x_value__mutmut_1" in capsys.readouterr().out


def test_gate_passes_when_every_mutant_was_killed(tmp_path: Path) -> None:
    assertion = "memo.example.x_assertion__mutmut_1"
    collection = "memo.example.x_collection__mutmut_2"
    payload = _metadata_payload({assertion: 1, collection: 3})
    payload["durations_by_key"] = {assertion: 0.25, collection: 1}
    payload["estimated_durations_by_key"] = {assertion: 0.1, collection: 0}
    payload["type_check_error_by_key"] = {assertion: None}
    _write_payload(tmp_path, payload)

    assert _mutation_gate().main([str(tmp_path)]) == 0


def test_gate_accepts_optional_type_check_mapping_omitted_by_mutmut_3_loader(
    tmp_path: Path,
) -> None:
    payload = _metadata_payload({"memo.example.x__mutmut_1": 1})
    del payload["type_check_error_by_key"]
    _write_payload(tmp_path, payload)

    assert _mutation_gate().main([str(tmp_path)]) == 0


def test_gate_passes_only_when_exact_reviewed_baseline_matches(tmp_path: Path) -> None:
    mutant = "memo.example.x_value__mutmut_1"
    blocked = {mutant: "survived"}
    _write_meta(tmp_path, {mutant: 0})
    baseline = _write_baseline(tmp_path, blocked)

    assert _mutation_gate().main([str(tmp_path), "--baseline", str(baseline)]) == 0


@pytest.mark.parametrize(
    "results",
    [
        {
            "memo.example.x_value__mutmut_1": 0,
            "memo.example.x_new__mutmut_2": 0,
        },
        {"memo.example.x_value__mutmut_1": 1},
        {"memo.example.x_value__mutmut_1": 33},
    ],
)
def test_gate_rejects_any_blocking_set_baseline_drift(
    tmp_path: Path,
    results: dict[str, int],
) -> None:
    mutant = "memo.example.x_value__mutmut_1"
    baseline = _write_baseline(tmp_path, {mutant: "survived"})
    _write_meta(tmp_path, results)

    assert _mutation_gate().main([str(tmp_path), "--baseline", str(baseline)]) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 2},
        {"blocking_count": -1},
        {"blocking_sha256": "not-a-digest"},
        {"blocking_by_reason": {"survived": "one"}},
        {"blocking_by_reason": {"survived": 2}},
        {"unexpected": True},
    ],
)
def test_gate_rejects_malformed_baseline(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    baseline = _write_baseline(
        tmp_path,
        {"memo.example.x_value__mutmut_1": "survived"},
        overrides=overrides,
    )

    with pytest.raises(RuntimeError, match="invalid mutation baseline"):
        _mutation_gate().load_baseline(baseline)


def test_gate_rejects_missing_metadata(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no mutmut metadata"):
        _mutation_gate().blocking_mutants(tmp_path)


def test_gate_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.py.meta"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"malformed mutmut metadata .*broken\.py\.meta"):
        _mutation_gate().blocking_mutants(tmp_path)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"exit_code_by_key": []},
        {"exit_code_by_key": {"memo.example.x__mutmut_1": "0"}},
    ],
)
def test_gate_rejects_malformed_metadata_schema(tmp_path: Path, payload: object) -> None:
    _write_payload(tmp_path, payload, filename="broken.py.meta")

    with pytest.raises(RuntimeError, match=r"malformed mutmut metadata .*broken\.py\.meta"):
        _mutation_gate().blocking_mutants(tmp_path)


@pytest.mark.parametrize(
    "missing_key",
    ["exit_code_by_key", "durations_by_key", "estimated_durations_by_key"],
)
def test_gate_requires_mutmut_3_metadata_keys(tmp_path: Path, missing_key: str) -> None:
    payload = _metadata_payload({"memo.example.x__mutmut_1": 1})
    del payload[missing_key]
    _write_payload(tmp_path, payload)

    with pytest.raises(RuntimeError, match=rf"missing required key.*{missing_key}"):
        _mutation_gate().blocking_mutants(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("durations_by_key", []),
        ("estimated_durations_by_key", None),
        ("type_check_error_by_key", "not-a-map"),
    ],
)
def test_gate_requires_mutmut_3_metadata_mappings(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _metadata_payload({"memo.example.x__mutmut_1": 1})
    payload[field] = value
    _write_payload(tmp_path, payload)

    with pytest.raises(RuntimeError, match=rf"{field} must be an object"):
        _mutation_gate().blocking_mutants(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("durations_by_key", "slow"),
        ("estimated_durations_by_key", None),
        ("type_check_error_by_key", 7),
    ],
)
def test_gate_validates_mutmut_3_metadata_mapping_values(
    tmp_path: Path, field: str, value: object
) -> None:
    mutant = "memo.example.x__mutmut_1"
    payload = _metadata_payload({mutant: 1})
    payload[field] = {mutant: value}
    _write_payload(tmp_path, payload)

    with pytest.raises(RuntimeError, match=rf"invalid {field} value"):
        _mutation_gate().blocking_mutants(tmp_path)


def test_gate_rejects_unexpected_mutmut_metadata_keys(tmp_path: Path) -> None:
    payload = _metadata_payload({"memo.example.x__mutmut_1": 1})
    payload["future_unvalidated_results"] = {}
    _write_payload(tmp_path, payload)

    with pytest.raises(RuntimeError, match=r"unexpected key.*future_unvalidated_results"):
        _mutation_gate().blocking_mutants(tmp_path)


def test_gate_rejects_duplicate_mutant_names_across_metadata(tmp_path: Path) -> None:
    mutant = "memo.example.x__mutmut_1"
    _write_meta(tmp_path, {mutant: 1}, filename="first.py.meta")
    second = _write_meta(tmp_path, {mutant: 0}, filename="second.py.meta")

    with pytest.raises(
        RuntimeError,
        match=rf"malformed mutmut metadata {second}.*duplicate mutant name",
    ):
        _mutation_gate().blocking_mutants(tmp_path)


def test_gate_rejects_metadata_without_mutants(tmp_path: Path) -> None:
    _write_meta(tmp_path, {})

    with pytest.raises(RuntimeError, match="no mutant results"):
        _mutation_gate().blocking_mutants(tmp_path)
