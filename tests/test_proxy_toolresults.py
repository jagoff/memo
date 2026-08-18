from memo.proxy.plan import Context
from memo.proxy.transforms.toolresults import (
    DEFAULT_FILTERS_DIR,
    Filter,
    ToolResults,
    apply_pipeline,
    generic_fallback,
    load_filters,
)
from memo.proxy.zones import Zones


def test_keep_lines_retains_only_matching_lines():
    out = apply_pipeline("a: ok\nb: FAIL\nc: ok\n", [{"action": "keep_lines", "pattern": "FAIL"}])
    assert out == "b: FAIL"


def test_remove_lines_drops_matching_lines():
    out = apply_pipeline("keep\nnoise\n", [{"action": "remove_lines", "pattern": "noise"}])
    assert out == "keep"


def test_aggregate_counts_instead_of_listing():
    text = "\n".join(f"PASS test_{i}" for i in range(500))
    out = apply_pipeline(text, [{"action": "aggregate", "pattern": "^PASS", "label": "passed"}])
    assert out == "500 passed"


def test_head_and_tail_compose():
    text = "\n".join(str(i) for i in range(100))
    out = apply_pipeline(text, [{"action": "head", "n": 2}])
    assert out == "0\n1"


def test_truncate_lines_caps_line_width():
    out = apply_pipeline("x" * 200, [{"action": "truncate_lines", "max": 10}])
    assert len(out) <= 13  # 10 chars plus an ellipsis marker


def test_an_unknown_action_is_a_no_op_not_a_crash():
    assert apply_pipeline("text", [{"action": "does_not_exist"}]) == "text"


def test_generic_fallback_keeps_head_and_tail_and_says_what_it_dropped():
    text = "\n".join(str(i) for i in range(1000))
    out = generic_fallback(text, max_chars=200)
    assert out.startswith("0\n1")
    assert "999" in out
    assert "elided" in out
    assert len(out) < len(text)


def test_short_output_passes_through_the_fallback_untouched():
    assert generic_fallback("short", max_chars=200) == "short"


def test_filters_load_from_yaml(tmp_path):
    (tmp_path / "f.yaml").write_text(
        "name: demo\n"
        "match:\n"
        "  command: git\n"
        "  subcommand: status\n"
        "pipeline:\n"
        "  - action: head\n"
        "    n: 5\n"
    )
    filters = load_filters(tmp_path)
    assert filters[0].name == "demo"
    assert filters[0].match_command == "git"


def test_a_malformed_filter_file_is_skipped_not_fatal(tmp_path):
    (tmp_path / "bad.yaml").write_text("{{{not yaml")
    assert load_filters(tmp_path) == []


# --- Beyond the brief's baseline: more pipeline-action and fail-open coverage ---


def test_tail_keeps_only_the_last_n_lines():
    text = "\n".join(str(i) for i in range(10))
    out = apply_pipeline(text, [{"action": "tail", "n": 3}])
    assert out == "7\n8\n9"


def test_dedup_drops_repeated_lines_preserving_first_occurrence_order():
    out = apply_pipeline("a\nb\na\nc\nb\n", [{"action": "dedup"}])
    assert out == "a\nb\nc"


def test_json_extract_pulls_a_dotted_path():
    out = apply_pipeline(
        '{"result": {"count": 3}}', [{"action": "json_extract", "path": "result.count"}]
    )
    assert out == "3"


def test_json_extract_on_non_json_is_a_no_op():
    assert apply_pipeline("not json", [{"action": "json_extract", "path": "a.b"}]) == "not json"


def test_format_template_substitutes_the_text_placeholder():
    out = apply_pipeline("42", [{"action": "format_template", "template": "count={text}"}])
    assert out == "count=42"


def test_a_broken_regex_pattern_is_a_no_op_not_a_crash():
    # Unbalanced group -- re.compile would raise re.error.
    assert apply_pipeline("text", [{"action": "keep_lines", "pattern": "("}]) == "text"


def test_one_bad_action_does_not_abort_the_rest_of_the_pipeline():
    out = apply_pipeline(
        "keep\nnoise\n",
        [
            {"action": "keep_lines", "pattern": "("},  # broken regex, no-op
            {"action": "remove_lines", "pattern": "noise"},
        ],
    )
    assert out == "keep"


def test_bundled_starter_filters_load_without_error():
    filters = load_filters(DEFAULT_FILTERS_DIR)
    names = {f.name for f in filters}
    assert {"git-status", "pytest", "npm-install"} <= names
    for f in filters:
        assert isinstance(f, Filter)
        assert f.pipeline


def test_load_filters_on_a_missing_directory_returns_empty(tmp_path):
    assert load_filters(tmp_path / "does-not-exist") == []


# --- ToolResults transform: fail-open, recovery-first cutting ---


def _ctx(tmp_path):
    return Context(state_dir=tmp_path, session_key="s1", project="memo")


def _zones_with_tool_result(command: str, output: str, tool_use_id: str = "t1") -> Zones:
    return Zones(
        live_messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": "Bash",
                        "input": {"command": command},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": output}],
            },
        ]
    )


def test_apply_uses_a_matching_filter_and_reports_a_saving(tmp_path):
    lines = ["=" * 20 + " test session " + "=" * 20]
    lines += [f"tests/test_x.py::test_{i} PASSED" for i in range(200)]
    lines += ["FAILED tests/test_x.py::test_broken - AssertionError"]
    output = "\n".join(lines)
    zones = _zones_with_tool_result("pytest -x", output)
    saved = ToolResults().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert "FAILED" in new_text
    assert saved > 0
    assert len(new_text) < len(output)


def test_apply_falls_back_to_generic_when_no_filter_matches(tmp_path):
    output = "\n".join(str(i) for i in range(2000))
    zones = _zones_with_tool_result("some-random-cli --verbose", output)
    ToolResults().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert "elided" in new_text
    assert len(new_text) < len(output)


def test_apply_leaves_short_output_untouched(tmp_path):
    zones = _zones_with_tool_result("some-random-cli", "short output")
    saved = ToolResults().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert new_text == "short output"
    assert saved == 0


def test_apply_never_cuts_without_a_recovery_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "memo.proxy.transforms.toolresults.ccr.stash", lambda state_dir, content: ""
    )
    output = "\n".join(str(i) for i in range(2000))
    zones = _zones_with_tool_result("some-random-cli", output)
    saved = ToolResults().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert new_text == output
    assert saved == 0


def test_apply_disabled_flag_skips_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_PROXY_TOOL_RESULTS", "0")
    assert ToolResults().enabled() is False


def test_apply_never_raises_on_malformed_live_messages(tmp_path):
    zones = Zones(
        live_messages=[None, {"role": "user", "content": "not a list"}, {"content": [None, 42]}]
    )
    saved = ToolResults().apply(zones, _ctx(tmp_path))
    assert saved == 0


def test_apply_marker_lets_the_original_be_recovered(tmp_path):
    output = "\n".join(str(i) for i in range(2000))
    zones = _zones_with_tool_result("some-random-cli", output)
    ToolResults().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert "memo_retrieve" in new_text

    from memo.proxy import ccr

    key = new_text.split('key="')[1].split('"')[0]
    assert ccr.recover(tmp_path, key) == output


# --- Fix round 1 regressions -------------------------------------------------


def test_apply_never_stores_a_net_larger_block_than_the_original(tmp_path):
    """Critical 1 repro: the shipped git-status filter's truncate_lines step
    barely shortens two long-path porcelain lines, and the recovery marker's
    own overhead (~130 bytes) outweighs that small saving. The stored block
    must never end up larger than the original -- previously it did, while
    apply() reported saved=0 as if nothing had happened, hiding a net-LOSS
    cut from both the caller and the token ledger."""

    def _line(prefix: str) -> str:
        return f"{prefix} src/" + ("nested/" * 28) + "file.py"

    original = f"{_line('M ')}\n{_line('??')}\n"
    zones = _zones_with_tool_result("git status", original)

    saved = ToolResults().apply(zones, _ctx(tmp_path))

    stored = zones.live_messages[1]["content"][0]["content"]
    assert len(stored) <= len(original), (
        f"stored block ({len(stored)} chars) is larger than the original "
        f"({len(original)} chars) -- a net-loss cut reported as neutral"
    )
    assert stored == original  # left untouched: the cut wasn't worth spending
    assert saved == 0


def test_pytest_filter_survives_a_pathological_separator_line():
    """Critical 2 repro: the shipped pytest.yaml keep_lines pattern
    `(FAILED|ERROR|error:|assert|=+ .* =+)` backtracks O(n^2) on a long line
    of '=' with no closing structure to match -- no malice required, just a
    large diff, a printed data structure, or a stray separator in captured
    output. The proxy runs synchronously in the critical path of every model
    call, so a slow match doesn't fail one request, it hangs the whole
    session. Must stay bounded regardless of how long the adversarial line
    actually is."""
    import time

    filters = load_filters(DEFAULT_FILTERS_DIR)
    pytest_filter = next(f for f in filters if f.name == "pytest")

    adversarial_line = "=" * 100_000  # no spaces anywhere: never completes a match
    text = f"{adversarial_line}\nFAILED tests/test_x.py::test_broken\n"

    start = time.perf_counter()
    apply_pipeline(text, pytest_filter.pipeline)
    elapsed = time.perf_counter() - start

    # Generous enough not to flake on a busy machine; tight enough to catch
    # a return of quadratic behavior (measured ~3.3s at this size, unbounded).
    assert elapsed < 2.0, f"pipeline took {elapsed:.2f}s on a pathological line -- ReDoS"
