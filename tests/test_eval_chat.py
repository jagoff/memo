from memo.eval_chat import apply_checks


def _done(answer: str, ids: list[str]) -> dict:
    return {"type": "done", "answer": answer, "sources": [{"id": i} for i in ids]}


def test_require_and_forbid_substrings() -> None:
    query = {
        "id": "q1",
        "checks": {"require_substrings": ["Avature"], "forbid_substrings": ["lambda"]},
    }
    ok = apply_checks(query, _done("Avature es una empresa", ["s1"]), 100)
    assert ok["passed"] is True
    bad = apply_checks(query, _done("usa lambda", ["s1"]), 100)
    assert bad["passed"] is False


def test_forbid_refusal_and_min_sources() -> None:
    from memo.chat.synthesis import REFUSAL

    query = {"id": "q2", "checks": {"forbid_refusal": True, "min_sources": 2}}
    assert apply_checks(query, _done(REFUSAL, ["a", "b"]), 10)["passed"] is False
    assert apply_checks(query, _done("respuesta", ["a"]), 10)["passed"] is False
    assert apply_checks(query, _done("respuesta", ["a", "b"]), 10)["passed"] is True


def test_expected_source_hit() -> None:
    query = {"id": "q3", "expected_source_ids": ["dev-PublicCloud"], "checks": {}}
    assert apply_checks(query, _done("x", ["dev-PublicCloudInfrastructure"]), 10)["passed"] is True
    assert apply_checks(query, _done("x", ["otro"]), 10)["passed"] is False
