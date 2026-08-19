import base64

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
# and Pixel.apply() end-to-end (flag gating, recovery-first, fail-open, and
# -- fix round 1 -- the corrected firing discipline: token-estimate gate plus
# an absolute dimension ceiling, no final byte-size veto). See pixel.py's
# module docstring for why the byte-size gate this file originally locked in
# was wrong: bytes and billed image tokens are different units and do not
# move together for a rendered PNG the way they do for the text transforms. ---


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


def test_apply_fires_on_dense_content_above_the_threshold_and_the_original_is_recoverable(
    tmp_path,
):
    """Fix round 1: pixel mode DOES fire once the byte-size veto is replaced
    with the correct token-based gate. "x" * 10_000 (a single dense line,
    well above the ~2,566-char single-line breakeven measured for the
    current geometry constants -- see task-13-report.md) clears
    `is_profitable`, gets rendered, and is written back as an `image` block
    with a recovery marker -- and the ORIGINAL text (not the rendered
    image) is exactly what `ccr.recover` returns for the marker's key."""
    pytest.importorskip("PIL")
    output = "x" * 10_000
    assert is_profitable(output)
    zones = _zones_with_tool_result(output)

    saved = Pixel().apply(zones, _ctx(tmp_path))

    content = zones.live_messages[0]["content"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    image_bytes = base64.b64decode(content[0]["source"]["data"])
    assert image_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert content[1]["type"] == "text"
    assert "memo_crush_retrieve" in content[1]["text"]
    assert saved > 0

    from memo.proxy import ccr

    key = content[1]["text"].split('hash_marker="')[1].split('"')[0]
    assert ccr.recover(tmp_path, key) == output


def test_apply_does_not_fire_on_long_but_low_density_content(tmp_path):
    """Not just "short text" (already covered above) -- a block that is long
    in TOTAL characters but spread thin (many short lines) never wins on the
    token-estimate gate either: image tokens scale with the page's pixel
    AREA, and a narrow-but-tall page of sparse short lines costs more area
    per character than the same character count packed densely."""
    output = "\n".join(str(i) for i in range(500))
    assert not is_profitable(output)
    zones = _zones_with_tool_result(output)

    saved = Pixel().apply(zones, _ctx(tmp_path))

    content = zones.live_messages[0]["content"][0]["content"]
    assert content == output
    assert saved == 0


def test_is_profitable_refuses_a_block_whose_page_would_exceed_the_dimension_ceiling():
    """The absolute safety bound (fix round 1, item 3): Anthropic's Messages
    API hard-rejects any image over 8000x8000px, independent of its encoded
    byte size. A single unbroken line of 260,000 repeated characters wraps
    to a page ~8,213px tall at the current geometry constants -- over the
    ceiling -- even though the token-estimate math ALONE (ignoring the
    ceiling) would call it profitable at roughly half the profitability
    threshold. The ceiling must still refuse it."""
    from memo.mcp_budget import est_tokens
    from memo.proxy.transforms.pixel import (
        _MAX_IMAGE_DIM_PX,
        _PROFITABLE_FRACTION,
        _page_geometry,
        est_image_tokens,
    )

    output = "x" * 260_000
    width, height = _page_geometry(output)
    assert height > _MAX_IMAGE_DIM_PX, "test setup: this block must actually exceed the ceiling"
    # Setup check: the token-estimate math alone, ignoring the ceiling,
    # WOULD call this profitable -- proving the ceiling is the thing doing
    # the refusing, not a coincidentally-unprofitable token estimate.
    assert est_image_tokens(width, height) < _PROFITABLE_FRACTION * est_tokens(output)

    assert not is_profitable(output)


def test_render_refuses_a_block_whose_page_would_exceed_the_dimension_ceiling():
    """Defense in depth: `render` enforces the same ceiling independently of
    `is_profitable`, in case it is ever called directly."""
    pytest.importorskip("PIL")
    assert render("x" * 260_000) is None


def test_apply_never_cuts_without_a_recovery_path(tmp_path, monkeypatch):
    """`ccr.stash` runs BEFORE the block is mutated (brief item 5: never cut
    without a recovery path first) -- so a stash failure must leave the
    block completely untouched even when the governing gate says fire."""
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
