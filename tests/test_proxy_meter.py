import json

from memo.proxy.meter import (
    Record,
    append,
    is_holdout,
    summarize,
    usage_from_response,
)


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


def test_by_transform_reports_counts_retrieval_rate_and_savings_share(tmp_path):
    for i in range(10):
        append(
            tmp_path,
            Record(
                request_key=f"a{i}",
                holdout=False,
                transforms=["toolschemas"],
                est_saved_tokens=100,
                input_tokens=500,
                output_tokens=10,
                retrieved=0,
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
                input_tokens=500,
                output_tokens=10,
                retrieved=1 if i < 6 else 0,
            ),
        )
    by = summarize(tmp_path)["by_transform"]
    assert by["toolschemas"]["n"] == 10
    assert by["toolschemas"]["retrieved"] == 0
    assert by["toolschemas"]["retrieval_rate"] == 0.0
    # total est_saved_tokens = 10*100 + 10*20 = 1200; toolschemas share = 1000/1200
    assert by["toolschemas"]["share"] == round(1000 / 1200, 4)
    assert by["jsoncrush"]["n"] == 10
    assert by["jsoncrush"]["retrieved"] == 6
    assert by["jsoncrush"]["retrieval_rate"] == 0.6
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
