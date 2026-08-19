"""Constructs the transform list. Separate from plan.py so transforms may import
the protocol without a cycle back through the planner."""

from __future__ import annotations

from memo.proxy.plan import Transform
from memo.proxy.transforms.delta import Delta
from memo.proxy.transforms.jsoncrush import JsonCrush
from memo.proxy.transforms.pixel import Pixel
from memo.proxy.transforms.structmap import StructMap
from memo.proxy.transforms.toolresults import ToolResults
from memo.proxy.transforms.toolschemas import ToolSchemas


def build_registry() -> list[Transform]:
    """Every enabled transform, in application order.

    ToolSchemas runs first, but that's incidental, not load-bearing: it is
    the one transform in ZONE_PREFIX (system/tools), a completely different
    part of the payload from everything below it, so its position relative
    to the rest cannot change what they see.

    StructMap and Delta run BEFORE JsonCrush and ToolResults, and that part
    IS load-bearing, for the same starvation reason task 11 already
    established for JsonCrush-before-ToolResults: whichever transform
    recognizes a block's real structure has to get first look at it, or the
    generic one downstream consumes it first and leaves nothing meaningful
    behind. Concretely: a `Read` tool's `input` carries `file_path`, never
    `command`, so ToolResults' own command-matching (`_tool_use_commands`)
    never produces a filter for it -- every `Read` result falls through to
    `generic_fallback`, a blind head+tail cut with no awareness that the
    text is source code or that the same path may have been read before.
    Letting ToolResults (or JsonCrush, for a `Read` of a `.json` file that
    happens to parse as a top-level array) touch a `Read` block first would
    hand StructMap/Delta an already-sliced-and-marked fragment instead of
    the real file: a signature map built from half a file, or a diff against
    text that was never the actual previous read, are both worse than the
    "wrong signature map" this pair's own module docstrings already refuse
    to produce on a parse failure -- so they must not be handed a mangled
    input in the first place.

    StructMap before Delta (rather than the reverse) is not load-bearing:
    the two are mutually exclusive by construction (`seen_files()` in
    delta.py splits blocks into "first read" and "re-read", and each
    transform only acts on its own case), so which one is asked first cannot
    change which blocks either one touches. The order here follows the pair's
    natural reading order -- first read, then re-read -- for the module list
    to read the same way a person reasons about the two cases.

    JsonCrush still runs before ToolResults for task 11's original reason:
    it only fires on a block that still parses as a clean JSON array, and
    ToolResults' generic_fallback would otherwise break that JSON's validity
    before JsonCrush ever saw it (CLAUDE.md: L1 SmartCrusher +44.4%).
    ToolResults runs last among the text-preserving transforms, as the
    text-level backstop over whatever every transform ahead of it left
    behind or declined to touch.

    Pixel runs LAST of all, after ToolResults, for the same starvation
    principle that orders every other pair above, taken to its logical
    end: Pixel is the most generic transform here -- it does not parse
    JSON, does not diff against a prior read, does not match a command to a
    filter pipeline. It replaces a `tool_result` block's TEXT content with an
    `image` block, which is exactly the kind of mangled, non-text input the
    module docstring above already refuses to hand StructMap/Delta or
    JsonCrush. Any transform placed after Pixel that still expects `content`
    to contain readable text -- ToolResults' own `_block_text`, StructMap's
    file-signature parser, JsonCrush's `json.loads` -- would either silently
    no-op on an image block or, worse, misread the leftover marker text as
    the whole tool result. Running Pixel last means every text-based
    transform gets first look, exactly as the docstring above already
    requires for JsonCrush-before-ToolResults; Pixel then only ever
    considers what nothing upstream could shrink further -- the correct
    role for the plan's most speculative, least-proven transform (see
    `pixel.py`'s module docstring and task-13-report.md): it gets the
    leftover worst case, never a block another transform hasn't had its
    chance at yet.
    """
    return [ToolSchemas(), StructMap(), Delta(), JsonCrush(), ToolResults(), Pixel()]
