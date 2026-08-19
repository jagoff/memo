import pytest

from memo.proxy.plan import Context
from memo.proxy.transforms.pixel import Pixel, est_image_tokens, is_profitable, render
from memo.proxy.zones import Zones


def test_image_token_estimate_follows_the_documented_formula():
    # Anthropic's documented approximation: tokens ~= (w * h) / 750
    assert est_image_tokens(1000, 1000) == pytest.approx(1333, abs=2)


def test_short_text_is_never_profitable_to_render():
    assert not is_profitable("just a short line")


def test_a_large_dense_block_is_profitable():
    assert is_profitable("x" * 200_000)


def test_render_returns_png_bytes_or_none_without_pillow():
    out = render("hello\nworld")
    assert out is None or out[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_of_empty_text_is_none():
    assert render("") is None


# --- Beyond the brief's baseline: is_profitable edge cases, render output,
# and Pixel.apply() end-to-end (flag gating, recovery-first, fail-open,
# and the documented outcome that the final byte gate blocks real content) ---


def test_is_profitable_is_false_for_empty_or_non_string_input():
    assert not is_profitable("")
    assert not is_profitable(None)  # type: ignore[arg-type]


def test_is_profitable_never_raises_on_pathological_input():
    # A single absurdly long line with no newlines anywhere.
    assert is_profitable("z" * 5_000_000) in (True, False)


def test_render_produces_a_real_png_when_pillow_is_available():
    pytest.importorskip("PIL")
    out = render("dense content\n" * 500)
    assert out is not None
    assert out[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_is_none_on_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no PIL for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert render("some text here") is None


def _ctx(tmp_path):
    return Context(state_dir=tmp_path, session_key="s1", project="memo")


def _zones_with_tool_result(output: str, tool_use_id: str = "t1") -> Zones:
    return Zones(
        live_messages=[
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": output}],
            },
        ]
    )


def test_apply_leaves_a_short_block_untouched(tmp_path):
    zones = _zones_with_tool_result("short output")
    saved = Pixel().apply(zones, _ctx(tmp_path))
    content = zones.live_messages[0]["content"][0]["content"]
    assert content == "short output"
    assert saved == 0


def test_apply_never_fires_on_real_dense_content_because_the_final_byte_gate_vetoes_it(tmp_path):
    """Documents the actual, measured outcome (see task-13-report.md): a real
    rendered PNG of glyph text -- unlike a solid-color image -- carries per-
    character entropy DEFLATE cannot erase, and base64 inflates it further by
    ~33%. Across inputs from a few hundred chars to 200,000 chars of maximally
    compressible content ("x" repeated), the base64 image + marker payload
    was LARGER in bytes than the plain text it would replace in every case
    measured. `is_profitable` (the token-estimate gate) says yes; the
    final-payload byte gate (brief item 6) is what actually decides, and it
    says no -- so nothing is written back and the block passes through
    unchanged. A future rendering change that starts actually beating this
    gate is welcome; this test exists so that behavior change is visible and
    intentional, not a silent regression to a bloated wire payload."""
    pytest.importorskip("PIL")
    output = "x" * 200_000
    zones = _zones_with_tool_result(output)
    assert is_profitable(output)  # the token-estimate gate alone WOULD fire

    saved = Pixel().apply(zones, _ctx(tmp_path))

    content = zones.live_messages[0]["content"][0]["content"]
    assert content == output  # untouched: gate 2 vetoed the write-back
    assert saved == 0


def test_apply_never_cuts_without_a_recovery_path(tmp_path, monkeypatch):
    """`ccr.stash` runs BEFORE the final byte gate (brief item 5: never cut
    without a recovery path first) -- so a stash failure must leave the
    block untouched regardless of what the byte gate would have decided."""
    pytest.importorskip("PIL")
    monkeypatch.setattr("memo.proxy.transforms.pixel.ccr.stash", lambda state_dir, content: "")
    monkeypatch.setattr("memo.proxy.transforms.pixel.is_profitable", lambda text: True)
    output = "some tool output text"
    zones = _zones_with_tool_result(output)
    saved = Pixel().apply(zones, _ctx(tmp_path))
    content = zones.live_messages[0]["content"][0]["content"]
    assert content == output
    assert saved == 0


def test_apply_disabled_flag_reports_not_enabled(monkeypatch):
    monkeypatch.setenv("MEMO_PROXY_PIXEL", "0")
    assert Pixel().enabled() is False


def test_apply_enabled_by_default():
    assert Pixel().enabled() is True


def test_apply_never_raises_on_malformed_live_messages(tmp_path):
    zones = Zones(
        live_messages=[None, {"role": "user", "content": "not a list"}, {"content": [None, 42]}]
    )
    saved = Pixel().apply(zones, _ctx(tmp_path))
    assert saved == 0


def test_apply_ignores_non_tool_result_blocks(tmp_path):
    zones = Zones(
        live_messages=[
            {"role": "assistant", "content": [{"type": "text", "text": "x" * 200_000}]},
        ]
    )
    saved = Pixel().apply(zones, _ctx(tmp_path))
    assert saved == 0
    assert zones.live_messages[0]["content"][0]["type"] == "text"
