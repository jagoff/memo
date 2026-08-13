#!/usr/bin/env python3
"""Replay a real Claude Code transcript's memo read-tool calls with the
emission ledger off, then on, and measure whether MEMO_EMITTED_LEDGER pays
for itself.

Scope (per the Task 5 finding recorded in
docs/SPECS/2026-08-10-emission-ledger-design.md): only `memo_search`,
`memo_ask`, and `memo_evidence_pack` are replayed. `memo_context` and
`memo_unified_briefing` were dropped from the feature itself -- their bodies
live inside a packed prose string, not a partitionable hit list -- so there
is nothing for this harness to replay for them either.

Criterion 1 wants the ratio of tokens MEMO put into one context window, not
tokens overall: the denominator is recall-hook injections plus participating
tool results, both measured with MEMO_EMITTED_LEDGER=0. The hook's own
contribution is taken directly from the transcript's recorded hook output,
not re-simulated -- the recall hook only WRITES to the ledger (records what
it injected so a later MCP call can digest it); it never READS the ledger to
shrink its own output. So the hook's token cost is identical whether the
flag is 0 or 1, and the number the transcript already recorded is the real
one, not an approximation.

Criterion 2 (`memo_get_after_digest` / `digests_served`) is NOT computed as
a pass/fail here. A replay has no model in the loop deciding whether to
recover a digested id via `memo_get`, so a synthetic run cannot produce that
rate -- see docs/eval/emission-ledger-replay.md for what can measure it.

Usage:
    uv run --no-sync python scripts/eval_emission_ledger.py \
        ~/.claude/projects/-Users-fer/<session>.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memo.mcp_budget import est_tokens

_PARTICIPATING = {"memo_search", "memo_ask", "memo_evidence_pack"}
_REDUCTION_THRESHOLD = 0.25


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


def _tool_calls(transcript: Path) -> list[ToolCall]:
    """Every participating memo tool call in the transcript, in order."""
    calls: list[ToolCall] = []
    for line in transcript.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:  # noqa: S112  # malformed transcript line -- skip it, keep reading
            continue
        content = row.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = str(block.get("name", "")).removeprefix("mcp__memo__")
            if name in _PARTICIPATING:
                calls.append(ToolCall(name, dict(block.get("input") or {})))
    return calls


def _hook_tokens(transcript: Path) -> int:
    """Tokens memo's recall hook actually injected into this session's
    window, read straight off the transcript rather than re-simulated (see
    module docstring for why that is the correct, and simpler, number)."""
    total = 0
    for line in transcript.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:  # noqa: S112  # malformed transcript line -- skip it, keep reading
            continue
        att = row.get("attachment")
        if not isinstance(att, dict):
            continue
        if att.get("hookEvent") != "UserPromptSubmit":
            continue
        if att.get("type") != "hook_additional_context":
            continue
        for chunk in att.get("content") or []:
            total += est_tokens(str(chunk))
    return total


def _run_search(memory: Any, args: dict[str, Any]) -> int:
    """Mirror memo_search's tool body (server_core_search.py): search,
    truncate to body_chars, then apply_ledger. Returns the est_tokens cost
    of what the caller would actually receive."""
    from memo.server_common import apply_ledger

    limit = int(args.get("limit") or 10)
    body_chars_raw = args.get("body_chars")
    body_chars = 280 if body_chars_raw is None else int(body_chars_raw)

    records = memory.search(
        str(args.get("query") or ""),
        limit=limit,
        type_=args.get("type"),
        mode=str(args.get("mode") or "hybrid"),
        quality_rerank=True,
    )
    hits: list[dict[str, Any]] = []
    for r in records:
        d = r.to_dict()
        body = d.get("body") or ""
        if body_chars >= 0 and len(body) > body_chars:
            d["body"] = body[:body_chars].rstrip() + "…"
            d["body_truncated"] = True
        hits.append(d)
    kept, extra = apply_ledger(memory, "memo_search", hits)
    return est_tokens(json.dumps({"hits": kept, **extra}, default=str))


def _run_ask(memory: Any, args: dict[str, Any]) -> int:
    """Mirror memo_ask's tool body (server_core_search.py)."""
    from memo.server_common import apply_ledger

    res = memory.ask(
        str(args.get("question") or ""),
        k=int(args.get("k") or 5),
        type_=args.get("type"),
        snippet_chars=args.get("snippet_chars"),
        include_repos=bool(args.get("include_repos", True)),
    )
    out = res if isinstance(res, dict) else {"answer": str(res)}
    payload: dict[str, Any] = {"answer": out.get("answer", "")}
    sources = out.get("sources")
    if isinstance(sources, list):
        kept, extra = apply_ledger(
            memory,
            "memo_ask",
            sources,
            text_of=lambda h: str(h.get("snippet") or "") if h.get("source") == "memory" else "",
        )
        payload["sources"] = kept
        payload.update(extra)
    return est_tokens(json.dumps(payload, default=str))


def _run_evidence_pack(memory: Any, args: dict[str, Any]) -> int:
    """Mirror memo_evidence_pack's tool body (server_operational.py)."""
    from memo.server_common import apply_ledger

    k = max(1, min(int(args.get("k") or 8), 50))
    min_coverage_raw = args.get("min_coverage")
    min_coverage = 0.2 if min_coverage_raw is None else float(min_coverage_raw)

    out = memory.evidence_pack(
        str(args.get("question") or ""),
        k=k,
        max_chars=int(args.get("max_chars") or 12_000),
        min_coverage=min_coverage,
        type_=args.get("type"),
        as_of=args.get("as_of"),
    ).to_dict()
    payload: dict[str, Any] = {
        "confidence": out.get("confidence"),
        "coverage": out.get("coverage"),
    }
    items = out.get("items")
    if isinstance(items, list):
        kept, extra = apply_ledger(
            memory, "memo_evidence_pack", items, text_of=lambda h: str(h.get("snippet") or "")
        )
        payload["items"] = kept
        payload.update(extra)
    return est_tokens(json.dumps(payload, default=str))


_RUNNERS = {
    "memo_search": _run_search,
    "memo_ask": _run_ask,
    "memo_evidence_pack": _run_evidence_pack,
}


def _reset_session(state_dir: Path, session: str) -> None:
    """Clear BOTH the ledger entries and the counters sidecar for a
    throwaway eval session.

    `emitted_ledger.reset()` deliberately clears only the `.jsonl` entries,
    never the `.counters.json` sidecar -- for a real session that is
    correct (counters must survive a mid-session PreCompact reset, see
    `emitted_ledger.reset`'s own docstring). But this harness reuses the
    same "eval-off" / "eval-on" session ids across repeated runs, and
    `bump()` is a read-modify-write ADD onto whatever is already on disk --
    so without also clearing the counters file, a second run's
    `digests_served` / `net_saved_est` silently accumulate on top of the
    first run's, understating nothing but reporting a number that describes
    every past invocation, not this one. `ledger_path` is public;
    the counters file is its sibling with the same safe-session stem."""
    from memo import emitted_ledger as el

    el.reset(state_dir, session)
    entries_path = el.ledger_path(state_dir, session)
    counters_path = entries_path.with_name(f"{entries_path.stem}.counters.json")
    counters_path.unlink(missing_ok=True)


def _replay(memory: Any, calls: list[ToolCall], *, enabled: bool, session: str) -> int:
    os.environ["MEMO_EMITTED_LEDGER"] = "1" if enabled else "0"
    os.environ["MEMO_SESSION_ID"] = session
    _reset_session(memory.cfg.state_dir, session)

    total = 0
    for call in calls:
        total += _RUNNERS[call.name](memory, call.args)
    return total


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    transcript = Path(sys.argv[1]).expanduser()

    calls = _tool_calls(transcript)
    hook_tokens = _hook_tokens(transcript)
    if not calls and not hook_tokens:
        print("no participating memo activity in this transcript")
        return 1

    from memo import emitted_ledger as el
    from memo.config import Config
    from memo.memory import Memory

    # One Memory instance, reused for both passes: session id and the ledger
    # flag are both read fresh from os.environ on every call (never cached
    # at construction), so a single instance is correct and avoids a second
    # cold embedder/LLM load.
    memory = Memory(Config.from_env())

    baseline_tools = _replay(memory, calls, enabled=False, session="eval-off") if calls else 0
    treated_tools = _replay(memory, calls, enabled=True, session="eval-on") if calls else 0

    counters = el.stats(memory.cfg.state_dir, "eval-on")
    served = counters["digests_served"]
    recovered = counters["memo_get_after_digest"]

    baseline_total = hook_tokens + baseline_tools
    treated_total = hook_tokens + treated_tools
    reduction = 0.0 if not baseline_total else (baseline_total - treated_total) / baseline_total

    _reset_session(memory.cfg.state_dir, "eval-off")
    _reset_session(memory.cfg.state_dir, "eval-on")

    tool_names = ", ".join(sorted({c.name for c in calls})) or "none"
    print(f"transcript:             {transcript.name}")
    print(f"participating calls:    {len(calls)}  ({tool_names})")
    print(f"recall-hook tokens:     {hook_tokens}   (identical off/on -- see doc)")
    print(f"tool tokens, flag off:  {baseline_tools}")
    print(f"tool tokens, flag on:   {treated_tools}")
    print(f"TOTAL, flag off:        {baseline_total}")
    print(f"TOTAL, flag on:         {treated_total}")
    print(f"reduction:              {reduction:.1%}   (criterion 1: >= 25%)")
    print()
    print(f"digests served:         {served}")
    print(f"memo_get after digest:  {recovered}   (criterion 2: UNMEASURABLE by replay -- see doc)")
    print(f"net_saved_est:          {counters['net_saved_est']} tokens (this session's counters)")

    criterion_1_pass = reduction >= _REDUCTION_THRESHOLD
    print()
    print(f"criterion 1 (>=25% reduction):    {'PASS' if criterion_1_pass else 'FAIL'}")
    print("criterion 2 (<20% recovery rate):  UNMEASURABLE BY REPLAY -- see doc")
    print()
    # Promotion needs BOTH criteria to pass (design spec, "Success criteria").
    # Criterion 2 can never come from a replay -- no model is in the loop to
    # decide whether to recover a digested id -- so this harness can confirm
    # criterion 1 failing (a real KEEP-AT-0), but it can never by itself
    # justify PROMOTE: that also needs live dogfooding data for criterion 2,
    # which does not exist yet. Printing "PROMOTE" here would assert
    # something this script structurally cannot know.
    if not criterion_1_pass:
        print("VERDICT: KEEP AT MEMO_EMITTED_LEDGER=0  (criterion 1 failed)")
        return 2
    print("VERDICT: KEEP AT MEMO_EMITTED_LEDGER=0  (criterion 1 passes, but criterion 2")
    print("  is unmeasured -- promotion needs a live dogfooding period; see")
    print("  docs/eval/emission-ledger-replay.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
