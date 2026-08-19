"""Pixel mode: render a dense text `tool_result` block to a PNG and ship it as
an `image` content block instead of text, because Anthropic's vision-token
accounting can be cheaper than text tokens for very dense content.

The most speculative transform in this plan (see the task-13 brief). The
profitability gate is the point, not the renderer -- see `is_profitable`
below and the numbers in this module's own test coverage / the task-13
report. A transform that never fires on real content is a correct, reportable
outcome, not a bug to paper over (rule 7 of the brief).

Two INDEPENDENT gates, in order, and both must pass before anything is
written back:

1. `is_profitable(text)` -- a pure token-ESTIMATE comparison, no Pillow, no
   rendering. It computes the same page geometry `render()` would actually
   lay out (`_page_geometry`, shared by both functions so the estimate can
   never diverge from what actually gets drawn) and asks whether the
   Anthropic-documented image-token approximation
   (`tokens ~= width * height / 750`) comes in meaningfully under the text's
   own token estimate. This is the gate described in the brief's item 4.
2. The FINAL-payload byte guard in `_rewrite_block` -- once an image is
   actually rendered and base64-encoded, the REAL comparison is the
   marker-and-image-included payload against the original text, in bytes
   (brief item 6). Base64 inflates bytes by ~33% on top of a PNG that (unlike
   a solid-color image) carries real per-glyph entropy DEFLATE cannot erase,
   so in practice this second gate is the one that decides: measured across
   inputs from 17 chars to 200,000 chars of maximally-compressible content,
   the base64 image + marker payload was LARGER in bytes than the plain text
   it would replace in every case tried (see task-13-report.md). Gate 1 can
   say "fewer tokens" while gate 2 correctly says "more bytes to transmit for
   it" -- both are honest, about different units, and gate 2 is the one that
   actually decides whether this transform ever writes back.
"""

from __future__ import annotations

import base64
import json
import logging

from memo.flags import flag_bool
from memo.mcp_budget import est_tokens
from memo.proxy import ccr
from memo.proxy.plan import ZONE_LIVE, Context
from memo.proxy.zones import Zones

_log = logging.getLogger(__name__)

# --- shared page-geometry model ---------------------------------------------
# Used by BOTH is_profitable (estimate, Pillow-free) and render (actual
# layout) -- see module docstring. Metrics are measured against Pillow's own
# bundled `ImageFont.load_default()` bitmap font (9px glyph width, ~11px line
# pitch at 2px top bearing -- verified via `ImageDraw.textbbox`), which is
# what `render()` uses: no external font file, no platform dependency, works
# identically in CI as on a dev machine.
_CHARS_PER_LINE = 120
_CHAR_W_PX = 9
_LINE_H_PX = 11
_PAD_PX = 20

# Only fire when the image is estimated to cost less than this fraction of
# the text-token cost -- a marginal win isn't worth shifting a block from
# plain text to an image (still recoverable via ccr, but strictly more
# machinery) for a rounding error.
_PROFITABLE_FRACTION = 0.8


def _page_geometry(text: str) -> tuple[int, int]:
    """(width_px, height_px) `render()` would produce for `text`, without
    rendering it. Wraps each source line to `_CHARS_PER_LINE`; the widest
    wrapped chunk (capped at `_CHARS_PER_LINE`) sets the page width, so a
    single very long line degrades to a narrow-but-tall page rather than an
    unbounded-width one."""
    wrapped = 0
    max_len = 0
    for line in text.split("\n"):
        if not line:
            wrapped += 1
            continue
        for start in range(0, len(line), _CHARS_PER_LINE):
            max_len = max(max_len, len(line[start : start + _CHARS_PER_LINE]))
            wrapped += 1
    wrapped = max(wrapped, 1)
    width = max_len * _CHAR_W_PX + 2 * _PAD_PX
    height = wrapped * _LINE_H_PX + 2 * _PAD_PX
    return width, height


def est_image_tokens(width: int, height: int) -> int:
    """Anthropic's documented approximation: tokens ~= (w * h) / 750."""
    return round(width * height / 750)


def is_profitable(text: str) -> bool:
    """True only when the image `render()` WOULD produce is estimated to
    cost under `_PROFITABLE_FRACTION` of the text-token cost. Pure
    arithmetic on the shared geometry model -- no Pillow import, no
    rendering, so this is cheap to call on every candidate block before
    deciding whether rendering is even worth attempting."""
    try:
        if not isinstance(text, str) or not text:
            return False
        text_cost = est_tokens(text)
        if text_cost <= 0:
            return False
        width, height = _page_geometry(text)
        image_cost = est_image_tokens(width, height)
        return image_cost < _PROFITABLE_FRACTION * text_cost
    except Exception:
        return False


def render(text: str) -> bytes | None:
    """PNG bytes for `text`, or None. Missing Pillow (no `[http]` extra),
    empty input, and any rendering failure are all the same "did not
    produce an image" outcome to the caller -- this function never raises."""
    if not isinstance(text, str) or not text:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    try:
        import io

        width, height = _page_geometry(text)
        image = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        y = _PAD_PX
        for line in text.split("\n"):
            if not line:
                y += _LINE_H_PX
                continue
            for start in range(0, len(line), _CHARS_PER_LINE):
                draw.text(
                    (_PAD_PX, y), line[start : start + _CHARS_PER_LINE], fill="black", font=font
                )
                y += _LINE_H_PX
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception:
        _log.debug("proxy: pixel render failed", exc_info=True)
        return None


def _block_text(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            c["text"]
            for c in content
            if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str)
        ]
        return "\n".join(parts)
    return ""


class Pixel:
    name = "pixel"
    zone = ZONE_LIVE

    def enabled(self) -> bool:
        try:
            return bool(flag_bool("MEMO_PROXY_PIXEL"))
        except Exception:
            return False

    def apply(self, zones: Zones, ctx: Context) -> int:
        try:
            if not zones.live_messages:
                return 0
            saved = 0
            for message in zones.live_messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    saved += self._rewrite_block(block, ctx)
            return saved
        except Exception:
            return 0

    def _rewrite_block(self, block: object, ctx: Context) -> int:
        try:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                return 0
            text = _block_text(block)
            # Gate 1 (brief item 4): cheap, Pillow-free -- run BEFORE any
            # rendering is even attempted.
            if not text or not is_profitable(text):
                return 0

            png_bytes = render(text)
            if not png_bytes:
                return 0

            # Recovery path (brief item 5) BEFORE the cut is written back --
            # an empty key means "could not stash," and the block is left
            # completely untouched.
            key = ccr.stash(ctx.state_dir, text)
            if not key:
                return 0

            marker = ccr.marker(key, kept_chars=0, dropped_chars=len(text), stashed=text)
            new_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(png_bytes).decode("ascii"),
                    },
                },
                {"type": "text", "text": marker},
            ]

            # Gate 2 (brief item 6): the REAL comparison. Base64 inflates
            # bytes by ~33% on top of a PNG that (unlike a solid-color image)
            # carries real per-glyph entropy -- only the FINAL, marker- and
            # image-included payload decides whether this write-back is
            # genuinely smaller than the text it replaces.
            new_bytes = len(json.dumps(new_content, ensure_ascii=False).encode("utf-8"))
            if new_bytes >= len(text.encode("utf-8")):
                return 0

            block["content"] = new_content
            width, height = _page_geometry(text)
            new_cost = est_image_tokens(width, height) + est_tokens(marker)
            return max(0, est_tokens(text) - new_cost)
        except Exception:
            return 0
