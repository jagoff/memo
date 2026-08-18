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
        True,
        "proxy",
        "Run the existing L1 JSON crusher over large JSON tool results.",
    ),
    _spec(
        "MEMO_PROXY_PIXEL",
        "bool",
        True,
        "proxy",
        "Render dense text blocks to PNG when the per-block profitability gate "
        "says the image costs fewer tokens. No-op without the [http] extra.",
    ),
    _spec(
        "MEMO_PROXY_RETRIEVE_ALARM_FRAC",
        "float",
        0.05,
        "proxy",
        "Retrieval rate above which a transform is reported as over-cutting — a "
        "recovered original costs its tokens twice.",
        min_val=0.0,
        max_val=1.0,
    ),
)