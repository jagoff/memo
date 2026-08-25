"""Runs memo's already-measured L1 JSON crusher over large JSON tool_result
blocks in the live zone.

Measured (see `CLAUDE.md`, capture-plane): L1 SmartCrusher
(`maybe_crush_json_capture`, `memo.capture_core`) cuts JSON-array capture
size +44.4% on the committed token-quality gate. It is a fully general
JSON-array reducer -- score rows by IDF-weighted distinctiveness, keep the
top-K, offload the rest to cache -- and this transform reuses it EXACTLY as
capture does, over tool_result blocks instead of capture content.
`_score_rows_by_relevance` (`capture_core.py`) is documented "pure,
deterministic" and position-blind, so the same block always crushes to the
same bytes -- that is what lets this run over the WHOLE conversation, not
just the newest turns, under `MEMO_PROXY_CONTENT_SCOPE=all` (default; see
`zones.scan_scope`).
Originals land in the SAME crush cache `memo retrieve` already reads
(`maybe_crush_json_capture` writes it itself, keyed by a truncated sha256
embedded as a marker row in the returned array), so there is exactly one
recovery path for a crushed block, same as for any memory the capture path
crushes. `ccr` is not involved here -- it exists for transforms with no
recovery story of their own (see `ccr.py`'s module docstring); this one
already has one.

Capture-plane flag interaction
-------------------------------
`maybe_crush_json_capture` is internally gated by `MEMO_CRUSHER_ENABLED`
(`flags_capture.py`) -- a CAPTURE-plane flag, default OFF, that governs
whether memo's own ingest crushes JSON before indexing. That is a completely
different code path from this proxy, and flipping its process-wide default
would silently change ingest behavior for the rest of memo. This transform
has its own flag, `MEMO_PROXY_JSONCRUSH` (`flags_proxy.py`), and the two
must stay decoupled.

Because `maybe_crush_json_capture`'s signature has no `enabled` override --
it is the fixed public interface this transform is required to consume, not
something to fork or duplicate -- the only way to make it run without
touching the capture-plane default is a scoped, restored `os.environ`
mutation around the single call:

* Process-scoped: `memo proxy serve` is its own OS process. This mutation is
  invisible to the CLI, hooks, or `memo-mcp` -- each is a separate process
  with its own environ, so "the rest of memo" never observes it.
* Call-scoped: the previous state is restored in a `finally` immediately
  after `maybe_crush_json_capture` returns, so nothing about this process's
  own environment differs before the first call and after the last.
* Deference-scoped: `_capture_flag_has_an_opinion` below checks env, the
  Markdown config, AND the tuned overlay -- the same three layers
  `flags.flag()` itself consults before falling back to the registry
  default. If ANY of them already has an opinion about
  `MEMO_CRUSHER_ENABLED`, this transform backs off completely and lets
  `maybe_crush_json_capture` read whatever is already there -- including an
  explicit OFF, which must never be forced on just because the proxy
  transform is enabled. Only a genuinely unconfigured flag (one that would
  fall through to the registry default) gets the temporary override. A
  failure reading either config layer is treated the same as "an opinion
  exists" -- fail toward NOT overriding, since a silent ingest-behavior
  change is the exact failure mode this function exists to avoid.

This is not a substitute for a real per-call parameter on
`maybe_crush_json_capture`; if that function ever grows one, this whole
workaround should be deleted in favor of it.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from typing import Any

from memo.flags import flag_bool
from memo.mcp_budget import est_tokens
from memo.proxy.plan import ZONE_LIVE, Context
from memo.proxy.zones import Zones, scan_scope

_log = logging.getLogger(__name__)

_CAPTURE_FLAG = "MEMO_CRUSHER_ENABLED"


def _capture_flag_has_an_opinion() -> bool:
    """True if `MEMO_CRUSHER_ENABLED` is set anywhere in the resolution
    chain `flags.flag()` itself consults (env, Markdown config, tuned
    overlay) -- i.e. anywhere OTHER than the registry default. See the
    module docstring for why this gates the override below."""
    if os.environ.get(_CAPTURE_FLAG):
        return True
    try:
        from memo.config_md import flag_values
        from memo.tuned_overlay import overlay_values

        if _CAPTURE_FLAG in flag_values(os.environ):
            return True
        if _CAPTURE_FLAG in overlay_values(os.environ):
            return True
    except Exception:
        _log.debug("proxy: could not read capture-flag config layers", exc_info=True)
        return True  # unknown -> treat as "has an opinion", never override
    return False


@contextlib.contextmanager
def _capture_flag_forced_on() -> Iterator[None]:
    """Temporarily set `MEMO_CRUSHER_ENABLED=1` in this process's environ,
    but only when nothing already has an opinion (see module docstring).
    Always restores exactly what was there before on exit."""
    if _capture_flag_has_an_opinion():
        yield
        return
    os.environ[_CAPTURE_FLAG] = "1"
    try:
        yield
    finally:
        os.environ.pop(_CAPTURE_FLAG, None)


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


def _set_block_text(block: dict, text: str) -> None:
    if isinstance(block.get("content"), list):
        block["content"] = [{"type": "text", "text": text}]
    else:
        block["content"] = text


class JsonCrush:
    name = "jsoncrush"
    zone = ZONE_LIVE

    def enabled(self) -> bool:
        try:
            return bool(flag_bool("MEMO_PROXY_JSONCRUSH"))
        except Exception:
            return False

    def apply(self, zones: Zones, ctx: Context) -> int:
        try:
            messages = scan_scope(zones)
            if not messages or not flag_bool("MEMO_PROXY_JSONCRUSH"):
                return 0
            from memo.config import Config

            # Pin state_dir to ctx.state_dir explicitly (rather than trust
            # whatever Config.from_env() resolves on its own) so the crush
            # cache this writes to is the SAME one `ccr.py` and `memo
            # retrieve` use -- see `memo.proxy.server`, which derives
            # ctx.state_dir the same way in production.
            config = Config.from_env(state_dir=ctx.state_dir)

            saved = 0
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    saved += self._rewrite_block(block, config)
            return saved
        except Exception:
            return 0

    def _rewrite_block(self, block: object, config: Any) -> int:
        try:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                return 0
            text = _block_text(block)
            if not text:
                return 0

            from memo.capture_core import maybe_crush_json_capture

            with _capture_flag_forced_on():
                new_text, crush_hash = maybe_crush_json_capture(text, "", config)

            # crush_hash is None whenever the crusher declined (not JSON, too
            # small, or its own internal >=5% ratio gate rejected the cut) --
            # in every one of those cases new_text IS the original, but the
            # length check is the REAL, final comparison (defense in depth,
            # matching the sibling toolresults.py transform's pattern) so a
            # future change to maybe_crush_json_capture can never make this
            # transform write back a net-larger block.
            if crush_hash is None or not isinstance(new_text, str) or len(new_text) >= len(text):
                return 0

            _set_block_text(block, new_text)
            return max(0, est_tokens(text) - est_tokens(new_text))
        except Exception:
            return 0
