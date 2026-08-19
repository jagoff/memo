from memo.proxy.plan import Context
from memo.proxy.transforms.delta import Delta, diff_against, seen_files
from memo.proxy.zones import Zones


def test_an_unchanged_reread_collapses_to_a_notice():
    text = "line1\nline2\nline3\n"
    out = diff_against(text, text)
    assert "unchanged" in out.lower()
    assert len(out) < len(text)


def test_a_changed_reread_shows_only_the_diff():
    before = "line1\nline2\nline3\n"
    after = "line1\nCHANGED\nline3\n"
    out = diff_against(before, after)
    assert "CHANGED" in out
    assert "line1" not in out or len(out) < len(after)


def test_a_first_read_with_no_previous_copy_is_untouched():
    assert diff_against("", "fresh content") == "fresh content"


# --- Beyond the brief's baseline ---------------------------------------------


def test_diff_against_never_raises_on_non_string_current():
    """`current` reaching `.splitlines()` as a non-string (a malformed
    upstream block) must fall back to returning it as-is, not propagate."""
    assert diff_against("previous text", None) is None  # type: ignore[arg-type]


# --- seen_files: scoped to frozen_messages only -------------------------------


def _read_pair(tool_use_id: str, file_path: str, content: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": "Read",
                    "input": {"file_path": file_path},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}],
        },
    ]


def test_seen_files_maps_path_to_content_from_frozen_messages():
    zones = Zones(frozen_messages=_read_pair("r1", "a.py", "original content"))
    assert seen_files(zones) == {"a.py": "original content"}


def test_seen_files_ignores_live_messages():
    """The turn currently in flight is not yet part of any response the
    model has seen -- see the module docstring for why counting it as
    "seen" would be wrong, not just an oversight."""
    zones = Zones(live_messages=_read_pair("r1", "a.py", "original content"))
    assert seen_files(zones) == {}


def test_seen_files_ignores_non_read_tools():
    zones = Zones(
        frozen_messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "ls"}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "b1", "content": "file.txt"}],
            },
        ]
    )
    assert seen_files(zones) == {}


def test_seen_files_never_raises_on_malformed_messages():
    zones = Zones(frozen_messages=[None, {"content": "not a list"}, {"content": [None, 42]}])
    assert seen_files(zones) == {}


def test_seen_files_keeps_the_most_recent_occurrence_of_a_path():
    zones = Zones(
        frozen_messages=[
            *_read_pair("r1", "a.py", "v1"),
            *_read_pair("r2", "a.py", "v2"),
        ]
    )
    assert seen_files(zones) == {"a.py": "v2"}


# --- Delta transform: Read-scoped, re-read-only, fail-open -------------------


def _ctx(tmp_path):
    return Context(state_dir=tmp_path, session_key="s1", project="memo")


def _zones_with_reread(previous: str, current: str, path: str = "a.py") -> Zones:
    return Zones(
        frozen_messages=_read_pair("r1", path, previous),
        live_messages=_read_pair("r2", path, current),
    )


def test_apply_diffs_a_reread_against_the_previously_seen_content(tmp_path):
    before = "\n".join(f"line{i}" for i in range(200))
    after = before.replace("line100", "line100-CHANGED")
    zones = _zones_with_reread(before, after)
    saved = Delta().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert saved > 0
    assert "line100-CHANGED" in new_text
    assert len(new_text) < len(after)


def test_apply_marker_lets_the_original_be_recovered(tmp_path):
    before = "\n".join(f"line{i}" for i in range(200))
    after = before.replace("line100", "line100-CHANGED")
    zones = _zones_with_reread(before, after)
    Delta().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert "memo_retrieve" in new_text

    from memo.proxy import ccr

    key = new_text.split('key="')[1].split('"')[0]
    assert ccr.recover(tmp_path, key) == after


def test_apply_collapses_an_identical_reread_to_the_unchanged_notice(tmp_path):
    text = "\n".join(f"line{i}" for i in range(200))
    zones = _zones_with_reread(text, text)
    saved = Delta().apply(zones, _ctx(tmp_path))
    new_text = zones.live_messages[1]["content"][0]["content"]
    assert saved > 0
    assert "unchanged" in new_text.lower()
    assert len(new_text) < len(text)


def test_apply_leaves_a_first_read_untouched(tmp_path):
    """No prior occurrence in frozen_messages -- StructMap's case, not ours."""
    content = "\n".join(f"line{i}" for i in range(200))
    zones = Zones(live_messages=_read_pair("r1", "a.py", content))
    saved = Delta().apply(zones, _ctx(tmp_path))
    assert saved == 0
    assert zones.live_messages[1]["content"][0]["content"] == content


def test_apply_leaves_a_non_read_tool_result_untouched(tmp_path):
    zones = Zones(
        frozen_messages=_read_pair("r1", "a.py", "original"),
        live_messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "ls"}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "b1", "content": "some output"}],
            },
        ],
    )
    saved = Delta().apply(zones, _ctx(tmp_path))
    assert saved == 0
    assert zones.live_messages[1]["content"][0]["content"] == "some output"


def test_apply_never_cuts_without_a_recovery_path(tmp_path, monkeypatch):
    monkeypatch.setattr("memo.proxy.transforms.delta.ccr.stash", lambda state_dir, content: "")
    before = "\n".join(f"line{i}" for i in range(200))
    after = before.replace("line100", "line100-CHANGED")
    zones = _zones_with_reread(before, after)
    saved = Delta().apply(zones, _ctx(tmp_path))
    assert saved == 0
    assert zones.live_messages[1]["content"][0]["content"] == after


def test_apply_disabled_flag_skips_everything(monkeypatch):
    monkeypatch.setenv("MEMO_PROXY_STRUCTMAP", "0")
    assert Delta().enabled() is False


def test_apply_never_raises_on_malformed_live_messages(tmp_path):
    zones = Zones(
        live_messages=[None, {"role": "user", "content": "not a list"}, {"content": [None, 42]}]
    )
    saved = Delta().apply(zones, _ctx(tmp_path))
    assert saved == 0


def test_apply_never_raises_when_frozen_messages_itself_is_malformed(tmp_path):
    """`seen_files` is delta's only "store" -- it is read straight out of
    `zones.frozen_messages` (see the module docstring), so a malformed or
    corrupt-shaped `frozen_messages` is the equivalent of an unreadable
    store for this transform. Must fall open, not raise."""
    zones = Zones(
        frozen_messages=[None, {"role": "user", "content": "not a list"}, {"content": [None, 42]}],
        live_messages=_read_pair("r1", "a.py", "current content"),
    )
    assert seen_files(zones) == {}
    saved = Delta().apply(zones, _ctx(tmp_path))
    assert saved == 0


def test_apply_never_raises_on_a_non_string_tool_result_content(tmp_path):
    """A `tool_result` block whose `content` is neither a string nor a list
    (a malformed upstream payload) must fall open, not raise, for both the
    frozen-side scan and the live-side rewrite."""
    zones = Zones(
        frozen_messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "r1",
                        "name": "Read",
                        "input": {"file_path": "a.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "r1", "content": 12345}],
            },
        ],
        live_messages=_read_pair("r2", "a.py", "current content"),
    )
    saved = Delta().apply(zones, _ctx(tmp_path))
    assert saved == 0


def test_apply_never_stores_a_net_larger_block_than_the_original(tmp_path):
    """A change small enough that the diff plus the recovery marker's own
    overhead (~130 bytes) is net-larger than the original re-read must be
    left completely untouched, never written back as a "cut" that actually
    grew the block."""
    before = "line1\nline2\nline3\n"
    after = "line1\nCHANGED\nline3\n"
    zones = _zones_with_reread(before, after)
    saved = Delta().apply(zones, _ctx(tmp_path))
    stored = zones.live_messages[1]["content"][0]["content"]
    assert len(stored) <= len(after)
    assert stored == after
    assert saved == 0
