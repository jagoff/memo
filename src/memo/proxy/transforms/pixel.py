"""Pixel mode: render a dense text `tool_result` block to a PNG and ship it as
an `image` content block instead of text, because Anthropic's vision-token
accounting can be cheaper than text tokens for very dense content.

The most speculative transform in this plan (see the task-13 brief). The
profitability gate is the point, not the renderer -- see `is_profitable`
below and the numbers in this module's own test coverage / the task-13
report.

**Fix round 1 correction.** An earlier version of this module gated the
write-back on comparing the FINAL byte-serialized payload (base64 image +
marker) against the original text's byte length -- the same discipline the
five TEXT transforms in this package correctly use. That discipline is wrong
here: Anthropic does not bill an image by its base64 payload size, it bills
by PIXEL DIMENSIONS (`tokens ~= width * height / 750`, this module's
documented approximation -- see `est_image_tokens`). Base64 always inflates
bytes by ~33% on top of a PNG that (unlike a solid-color image) carries real
per-glyph entropy DEFLATE cannot fully erase, so a byte-size gate vetoed
every case measured, closing the exact door this transform exists to walk
through: a payload that is LARGER in bytes can still be CHEAPER in the units
the provider actually charges. Firing is governed by the TOKEN comparison
alone (`is_profitable`); see task-13-report.md for the corrected numbers and
what changed.

Two things must both hold before anything is written back:

1. `is_profitable(text)` -- a pure token-ESTIMATE comparison, no Pillow, no
   rendering. It computes the same page geometry `render()` would actually
   lay out (`_page_geometry`, shared by both functions so the estimate can
   never diverge from what actually gets drawn), rejects anything whose
   geometry would exceed `_MAX_IMAGE_DIM_PX` (see below), and otherwise asks
   whether the estimated image-token cost comes in under
   `_PROFITABLE_FRACTION` of the text's own token estimate -- "meaningfully
   below," per the brief, not just numerically less.
2. An absolute dimension ceiling, `_MAX_IMAGE_DIM_PX = 8000`: Anthropic's
   Messages API documents a hard per-image maximum of 8000x8000 pixels --
   requests with a larger image are rejected outright
   (https://platform.claude.com/docs/en/build-with-claude/vision, "Image
   limits and costs", fetched 2026-08-19), independent of file size. This is
   the ceiling to enforce, not a self-imposed byte cap: this module's own
   page geometry has UNBOUNDED height for a sufficiently long single line or
   a sufficiently line-dense block (width saturates at `_CHARS_PER_LINE`,
   height does not), and the token-ratio math in (1) does not naturally bound
   it -- a maximally dense, highly repetitive block can stay profitable by
   the token estimate at any scale. A byte-size cap was considered and
   rejected as the primary bound: a single very long line of a repeated
   character compresses so well under PNG/DEFLATE that its encoded size
   stays small even once its HEIGHT has blown past 8000px (measured: a
   single-line block reaches 8000px height at roughly 87,000 characters,
   where its PNG is still well under 200KB) -- so a byte ceiling generous
   enough not to reject ordinary dense content would not have caught this
   case, while the dimension ceiling catches it exactly, because it is the
   real failure mode. Checked in both `is_profitable` (so a pathological
   block is rejected before any rendering is attempted) and `render` itself
   (defense in depth, in case `render` is ever called directly).

**What this module cannot verify.** `est_image_tokens` is Anthropic's
documented *approximation*; provider-side accounting for very large or
unusually shaped images, and any resizing the provider applies before
billing, are not something a local formula or a unit test can confirm.
Separately, and more importantly: whether a model reads rendered text as
reliably as it reads plain text -- at this font, this density, this
line-wrap -- is a comprehension question, not a token-accounting question,
and nothing in this module or its test suite measures it. A green
profitability gate proves the transform is estimated to be CHEAPER; it does
not prove the model understands the image as well as it would have
understood the text. That risk is real and unmeasured; see task-13-report.md.
"""

from __future__ import annotations

import base64
import logging

from memo.flags import flag_bool
from memo.mcp_budget import est_tokens
from memo.proxy import ccr
from memo.proxy.plan import ZONE_LIVE, Context
from memo.proxy.zones import Zones

_log = logging.getLogger(__name__)

# --- shared page-geometry model ---------------------------------------------
# Used by BOTH is_profitable (estimate, Pillow-free) and render (actual
# layout) -- see module docstring. Font metrics (9px glyph width, ~11px line
# pitch) are measured against Pillow's own bundled `ImageFont.load_default()`
# bitmap font via `ImageDraw.textbbox`, which is what `render()` uses: no
# external font file, no platform dependency, works identically in CI as on
# a dev machine. The wrap width (350 chars/line) is a layout choice, not a
# font metric: wide enough that a 200,000-character dense single-line block
# renders at ~6,300px tall -- comfortably under the 8000px ceiling below --
# while still reading as a single reasonably-proportioned page rather than a
# needle-thin column.
_CHARS_PER_LINE = 350
_CHAR_W_PX = 9
_LINE_H_PX = 11
_PAD_PX = 20

# Only fire when the image is estimated to cost less than this fraction of
# the text-token cost -- "meaningfully below" per the brief, not just
# numerically less: a marginal win isn't worth shifting a block from plain
# text to an image (still recoverable via ccr, but strictly more machinery,
# and see the module docstring's comprehension caveat) for a rounding error.
_PROFITABLE_FRACTION = 0.8

# Anthropic's documented hard per-image maximum (Messages API): a larger
# image is rejected outright, independent of its encoded byte size. See the
# module docstring for why this -- not a byte-size cap -- is the ceiling
# that actually bounds the pathological case.
_MAX_IMAGE_DIM_PX = 8000


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
    """True only when (a) the page `render()` would produce stays within
    Anthropic's hard 8000x8000px per-image limit, AND (b) the image is
    estimated to cost under `_PROFITABLE_FRACTION` of the text-token cost.
    Pure arithmetic on the shared geometry model -- no Pillow import, no
    rendering -- so this is cheap to call on every candidate block before
    deciding whether rendering is even worth attempting."""
    try:
        if not isinstance(text, str) or not text:
            return False
        text_cost = est_tokens(text)
        if text_cost <= 0:
            return False
        width, height = _page_geometry(text)
        if width > _MAX_IMAGE_DIM_PX or height > _MAX_IMAGE_DIM_PX:
            return False
        image_cost = est_image_tokens(width, height)
        return image_cost < _PROFITABLE_FRACTION * text_cost
    except Exception:
        return False


def render(text: str) -> bytes | None:
    """PNG bytes for `text`, or None. Missing Pillow (no `[http]` extra),
    empty input, a page that would exceed `_MAX_IMAGE_DIM_PX` (defense in
    depth -- `is_profitable` already checks this, but `render` must not
    depend on being called through it), and any rendering failure are all
    the same "did not produce an image" outcome to the caller -- this
    function never raises."""
    if not isinstance(text, str) or not text:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    try:
        import io

        width, height = _page_geometry(text)
        if width > _MAX_IMAGE_DIM_PX or height > _MAX_IMAGE_DIM_PX:
            return None
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
            # The governing gate: token estimate + absolute dimension
            # ceiling, both inside is_profitable. Cheap, Pillow-free -- runs
            # BEFORE any rendering is even attempted.
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

            # Final re-check in the SAME unit is_profitable used (tokens, not
            # bytes): the marker itself costs a few tokens on top of the bare
            # image estimate, so re-derive the real total and refuse the
            # write-back on the rare case that pushes it to parity or worse.
            width, height = _page_geometry(text)
            new_cost = est_image_tokens(width, height) + est_tokens(marker)
            text_cost = est_tokens(text)
            if new_cost >= text_cost:
                return 0

            block["content"] = [
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
            return text_cost - new_cost
        except Exception:
            return 0
