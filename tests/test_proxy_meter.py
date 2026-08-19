import json

from memo.proxy.meter import (
    Record,
    append,
    is_holdout,
    summarize,
    usage_from_response,
)
from memo.proxy.plan import Context


def test_holdout_assignment_is_stable_for_a_key():
    assert is_holdout("abc", 0.5) == is_holdout("abc", 0.5)


def test_holdout_fraction_zero_holds_nothing_out():
    assert not any(is_holdout(str(i), 0.0) for i in range(200))


def test_holdout_fraction_one_holds_everything_out():
    assert all(is_holdout(str(i), 1.0) for i in range(200))


def test_holdout_fraction_is_roughly_honoured():
    n = sum(is_holdout(str(i), 0.1) for i in range(2000))
    assert 120 < n < 280  # 10% of 2000, generous band


def test_usage_reads_all_four_provider_fields():
    body = {
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_creation_input_tokens": 30,
            "cache_read_input_tokens": 40,
        }
    }
    assert usage_from_response(body) == {
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_creation_tokens": 30,
        "cache_read_tokens": 40,
    }


def test_usage_of_a_bodyless_response_is_all_zeroes():
    assert usage_from_response({}) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
    }


def test_append_writes_one_json_line_per_record(tmp_path):
    append(
        tmp_path,
        Record(
            request_key="k1",
            holdout=False,
            transforms=["toolschemas"],
            est_saved_tokens=100,
            input_tokens=1,
            output_tokens=2,
            cache_creation_tokens=3,
            cache_read_tokens=4,
            retrieved=0,
        ),
    )
    lines = (tmp_path / "proxy" / "requests.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["request_key"] == "k1"


def test_summarize_compares_treated_against_holdout(tmp_path):
    for i in range(10):
        append(
            tmp_path,
            Record(
                request_key=f"t{i}",
                holdout=False,
                transforms=["x"],
                est_saved_tokens=50,
                input_tokens=500,
                output_tokens=10,
                cache_creation_tokens=0,
                cache_read_tokens=0,
                retrieved=0,
            ),
        )
    for i in range(10):
        append(
            tmp_path,
            Record(
                request_key=f"h{i}",
                holdout=True,
                transforms=[],
                est_saved_tokens=0,
                input_tokens=1000,
                output_tokens=10,
                cache_creation_tokens=0,
                cache_read_tokens=0,
                retrieved=0,
            ),
        )
    s = summarize(tmp_path)
    assert s["n_treated"] == 10
    assert s["n_holdout"] == 10
    assert s["mean_input_treated"] == 500
    assert s["mean_input_holdout"] == 1000
    assert s["measured_saving_frac"] == 0.5


def test_summarize_reports_no_data_rather_than_a_zero(tmp_path):
    assert summarize(tmp_path)["n_treated"] == 0
    assert summarize(tmp_path)["measured_saving_frac"] is None


def test_summarize_survives_a_corrupt_line(tmp_path):
    (tmp_path / "proxy").mkdir()
    (tmp_path / "proxy" / "requests.jsonl").write_text("{not json\n")
    assert summarize(tmp_path)["skipped"] == 1


def test_summarize_skips_a_valid_json_line_that_is_not_an_object(tmp_path):
    (tmp_path / "proxy").mkdir()
    (tmp_path / "proxy" / "requests.jsonl").write_text("42\n")
    assert summarize(tmp_path)["skipped"] == 1


def test_summarize_survives_a_torn_write_with_invalid_utf8(tmp_path):
    (tmp_path / "proxy").mkdir()
    (tmp_path / "proxy" / "requests.jsonl").write_bytes(b'{"holdout": false}\n\xc3\x28')
    result = summarize(tmp_path)
    assert result["n_treated"] == 1


def test_summarize_skips_a_row_whose_counter_is_not_a_number(tmp_path):
    (tmp_path / "proxy").mkdir()
    (tmp_path / "proxy" / "requests.jsonl").write_text(
        '{"holdout": false, "input_tokens": "abc"}\n'
    )
    assert summarize(tmp_path)["measured_saving_frac"] is None


def test_by_transform_share_reflects_real_per_transform_savings_not_a_flat_split(tmp_path):
    """Round-2 regression: `plan.apply_all` appends every ENABLED transform's
    name to `transforms` regardless of whether it saved anything, and the old
    meter.py credited the row's whole scalar `est_saved_tokens` to every name
    in that list — so a row where 5 transforms ran but only 1 actually saved
    anything reported a flat 1/5 share for each. Verified on a real payload
    shape: jsoncrush saved 100%, the other four saved 0 — all five used to
    print 20%. `saved_by` (from `TransformPlan`) is the real per-transform
    breakdown; shares must reflect it, not `len(transforms)`."""
    for i in range(10):
        append(
            tmp_path,
            Record(
                request_key=f"a{i}",
                holdout=False,
                transforms=["toolschemas", "structmap", "delta", "jsoncrush", "toolresults"],
                est_saved_tokens=100,
                saved_by={"jsoncrush": 100},
                input_tokens=500,
                output_tokens=10,
            ),
        )
    by = summarize(tmp_path)["by_transform"]
    assert by["jsoncrush"]["share"] == 1.0
    assert by["jsoncrush"]["est_saved_tokens"] == 1000
    for name in ("toolschemas", "structmap", "delta", "toolresults"):
        assert by[name]["est_saved_tokens"] == 0
        assert by[name]["share"] in (0.0, None)
        # still shows it ran on all 10 requests, just earned nothing
        assert by[name]["n"] == 10


def test_by_transform_splits_savings_honestly_across_two_earners(tmp_path):
    for i in range(10):
        append(
            tmp_path,
            Record(
                request_key=f"a{i}",
                holdout=False,
                transforms=["toolschemas"],
                est_saved_tokens=100,
                saved_by={"toolschemas": 100},
                input_tokens=500,
                output_tokens=10,
            ),
        )
    for i in range(10):
        append(
            tmp_path,
            Record(
                request_key=f"b{i}",
                holdout=False,
                transforms=["jsoncrush"],
                est_saved_tokens=20,
                saved_by={"jsoncrush": 20},
                input_tokens=500,
                output_tokens=10,
            ),
        )
    by = summarize(tmp_path)["by_transform"]
    assert by["toolschemas"]["n"] == 10
    assert by["toolschemas"]["est_saved_tokens"] == 1000
    assert by["toolschemas"]["share"] == round(1000 / 1200, 4)
    assert by["jsoncrush"]["n"] == 10
    assert by["jsoncrush"]["est_saved_tokens"] == 200
    assert by["jsoncrush"]["share"] == round(200 / 1200, 4)


def test_by_transform_share_is_none_without_any_savings(tmp_path):
    append(
        tmp_path,
        Record(
            request_key="a",
            holdout=False,
            transforms=["delta"],
            est_saved_tokens=0,
            input_tokens=500,
            output_tokens=10,
        ),
    )
    by = summarize(tmp_path)["by_transform"]
    assert by["delta"]["share"] is None


def test_by_transform_has_no_dead_retrieval_columns(tmp_path):
    """Round-2: `retrieved`/`retrieval_rate` per transform had no writer
    anywhere (`retrieved` is never set on a real Record) — a rate that is
    always 0.0 and an alarm that can never fire is the same banned "prints a
    literal 0" shape this task exists to remove. Dropped, not faked."""
    append(
        tmp_path,
        Record(
            request_key="a",
            holdout=False,
            transforms=["delta"],
            est_saved_tokens=10,
            saved_by={"delta": 10},
            input_tokens=500,
            output_tokens=10,
        ),
    )
    by = summarize(tmp_path)["by_transform"]
    assert "retrieved" not in by["delta"]
    assert "retrieval_rate" not in by["delta"]


def test_by_transform_against_a_real_registry_plan_not_a_hand_written_row(tmp_path, monkeypatch):
    """The hand-written single-transform Records above prove the aggregation
    math, but production never produces that shape: `rewrite_body` runs the
    real 5-transform registry, and `plan.applied` lists every one that's
    enabled whether or not it rewrote anything. Run the real pipeline over a
    payload only one transform can act on, and confirm `by_transform` credits
    only the real earner — not a flat 1/5 split across all five."""
    import json as _json

    from memo.proxy.server import rewrite_body

    monkeypatch.setenv("MEMO_CRUSHER_ENABLED", "1")
    big_array = _json.dumps([{"id": i, "text": "row " * 20} for i in range(200)])
    raw = _json.dumps(
        {"messages": [{"role": "user", "content": [{"type": "tool_result", "content": big_array}]}]}
    ).encode()
    ctx = Context(state_dir=tmp_path, session_key="s1", project=None)
    _out, plan = rewrite_body(raw, ctx)

    # Production shape: several transforms are "enabled" and applied, but not
    # all of them touched this payload.
    assert len(plan.applied) >= 2
    assert plan.saved_by, "at least one real transform must have earned credit"
    assert set(plan.saved_by) < set(plan.applied), (
        "at least one applied transform must have earned nothing"
    )
    assert sum(plan.saved_by.values()) == plan.est_saved_tokens

    append(
        tmp_path,
        Record(
            request_key="prod1",
            holdout=False,
            transforms=plan.applied,
            est_saved_tokens=plan.est_saved_tokens,
            saved_by=plan.saved_by,
            input_tokens=500,
            output_tokens=10,
        ),
    )
    by = summarize(tmp_path)["by_transform"]
    shares = {name: v["share"] for name, v in by.items()}
    # The old bug: every applied transform reports the identical flat share.
    assert len(set(shares.values())) > 1, f"flat share across all transforms: {shares}"
    for name in plan.applied:
        if name not in plan.saved_by:
            assert by[name]["est_saved_tokens"] == 0
            assert by[name]["share"] in (0.0, None)
