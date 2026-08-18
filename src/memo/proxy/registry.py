"""Constructs the transform list. Separate from plan.py so transforms may import
the protocol without a cycle back through the planner."""

from __future__ import annotations

from memo.proxy.plan import Transform
from memo.proxy.transforms.jsoncrush import JsonCrush
from memo.proxy.transforms.toolresults import ToolResults
from memo.proxy.transforms.toolschemas import ToolSchemas


def build_registry() -> list[Transform]:
    """Every enabled transform, in application order. Populated by later tasks.

    JsonCrush runs BEFORE ToolResults deliberately: JsonCrush only fires on a
    block that still parses as a clean JSON array (`[...]`), and ToolResults'
    generic_fallback (the default for any command without a matching YAML
    filter) inserts a `[... N chars elided ...]` marker mid-string plus a
    trailing recovery marker -- either one breaks JSON validity. Running
    ToolResults first would starve JsonCrush of the well-formed JSON arrays
    it is measured against (CLAUDE.md: L1 SmartCrusher +44.4%), for exactly
    the large-JSON-tool-output case task 11 targets. ToolResults still runs
    second as a text-level backstop over whatever JsonCrush leaves behind
    (crushed JSON that is still large, or content JsonCrush declined).
    """
    return [ToolSchemas(), JsonCrush(), ToolResults()]
