"""Constructs the transform list. Separate from plan.py so transforms may import
the protocol without a cycle back through the planner."""

from __future__ import annotations

from memo.proxy.plan import Transform
from memo.proxy.transforms.delta import Delta
from memo.proxy.transforms.jsoncrush import JsonCrush
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
    ToolResults runs last as the text-level backstop over whatever every
    transform ahead of it left behind or declined to touch.
    """
    return [ToolSchemas(), StructMap(), Delta(), JsonCrush(), ToolResults()]
