"""Flags for the context-compression proxy (`memo proxy serve`).

The proxy sits between Claude Code and the provider. Every transform is ON by
default and self-limiting; `MEMO_PROXY_ENABLED=0` (or `memo proxy off`) turns
the whole thing into pure passthrough.
"""

from __future__ import annotations

from memo.flags_base import FlagSpec, _spec

SPECS: tuple[FlagSpec, ...] = (
    _spec(
        "MEMO_PROXY_ENABLED",
        "bool",
        True,
        "proxy",
        "Master switch for payload rewriting. When off the proxy still forwards "
        "and still measures, but changes nothing — the honest passthrough mode.",
    ),
    _spec(
        "MEMO_PROXY_PORT",
        "int",
        8768,
        "proxy",
        "Loopback port the proxy listens on. 8765 and 8767 are commonly taken "
        "by other local services; install fails loudly on a bound port.",
        min_val=1024,
        max_val=65535,
    ),
    _spec(
        "MEMO_PROXY_HOLDOUT_FRAC",
        "float",
        0.05,
        "proxy",
        "Fraction of requests forwarded uncompressed as the control arm. This is "
        "what every savings claim is measured against; 0 disables measurement "
        "of savings, not the transforms.",
        min_val=0.0,
        max_val=0.5,
    ),
    _spec(
        "MEMO_PROXY_TOOL_SCHEMAS",
        "bool",
        True,
        "proxy",
        "Prune MCP tool definitions to those used recently in this project.",
    ),
    _spec(
        "MEMO_PROXY_TOOL_WINDOW_SESSIONS",
        "int",
        20,
        "proxy",
        "Sessions of usage history that decide which tool schemas survive pruning.",
        min_val=1,
    ),
    _spec(
        "MEMO_PROXY_TOOL_SCHEMAS_SCOPE",
        "str",
        "all",
        "proxy",
        "Which tool schemas MEMO_PROXY_TOOL_SCHEMAS is allowed to prune: 'all' "
        "(every tool on the wire, not just memo's — the aggressive default; "
        "measured live traffic showed memo_* tools are 0% of a real payload's "
        "schema cost) or 'memo' (only memo_* tools, the original conservative "
        "scope, kept as a one-flag-away fallback).",
        choices=("all", "memo"),
    ),
    _spec(
        "MEMO_PROXY_TOOL_RESULTS",
        "bool",
        True,
        "proxy",
        "Apply declarative per-command filters to tool_result blocks.",
    ),
    _spec(
        "MEMO_PROXY_STRUCTMAP",
        "bool",
        True,
        "proxy",
        "Replace code-file reads with signatures, and re-reads with a diff.",
    ),
    _spec(
        "MEMO_PROXY_JSONCRUSH",
        "bool",
        False,
        "proxy",
        "Run the existing L1 JSON crusher over large JSON tool results. Off by "
        "default: across 37 measured real requests it never fired once, so it "
        "shipped as pure per-request cost. A workload with denser JSON tool "
        "results than memo's own may still want it -- hence the flag, not a "
        "deletion.",
    ),
    _spec(
        "MEMO_PROXY_PIXEL",
        "bool",
        False,
        "proxy",
        "Render dense text blocks to PNG when the per-block profitability gate "
        "says the image costs fewer tokens. No-op without the [http] extra. Off "
        "by default: measured at 89 tokens saved per request (0.1% of the "
        "proxy's total) against a comprehension cost that was never measured, "
        "and its own profitability gate is an ESTIMATE (width * height / 750), "
        "not billed truth. Trading an unmeasured risk for a measured 0.1% is "
        "the wrong side of that bet; turn it back on to re-open it.",
    ),
    _spec(
        "MEMO_PROXY_CONTENT_SCOPE",
        "str",
        "all",
        "proxy",
        "Which messages structmap/delta/jsoncrush/toolresults/pixel are "
        "allowed to scan and rewrite: 'all' (the whole conversation, frozen "
        "zone included -- the aggressive default; each of these five "
        "transforms maps a block to byte-identical output on every turn, so "
        "widening past the live zone never rewrites what the provider already "
        "cached) or 'tail' (only the live window -- the original conservative "
        "scope, kept as a one-flag-away rollback). Does not affect "
        "MEMO_PROXY_TOOL_SCHEMAS, which already scans the whole prefix.",
        choices=("all", "tail"),
    ),
    _spec(
        "MEMO_PROXY_CAPTURE",
        "int",
        0,
        "proxy",
        "Diagnostic: dump this many raw request/rewrite pairs to "
        "<state_dir>/proxy/capture/ and then stop. 0 (default) captures "
        "nothing. Payloads contain the full conversation -- enable only "
        "when investigating, on a machine you trust with your own prompts.",
    ),
)
