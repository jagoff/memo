"""Grounding detector — did the answer actually USE the recalled memories?

Runs at the Stop hook (`memo capture-stop` → `grounding.score_turn`), AFTER the
answer is generated, inside the 30s async budget. For the turn's recalled
memories (cached as `hits[].snippet` in recall.log by the recall-hook) it scores
how much each one shows up in the assistant's final answer:

  1. lexical containment (free) — salient snippet tokens present in the answer.
  2. embedding cosine (only on the ambiguous middle band) — via the already-warm
     daemon embedder (`embedder_client.embed`, symmetric/doc side, no query
     prefix). Catches paraphrase.

`used_score = max(lexical, embed_cosine)`. One `grounding.log` row per recalled
memory, keyed by `(session_id, turn, recall_id)` — the recall→use ledger that
`dashboard.grounded_rate` joins against recall_hook.log, with recall.log as a
compatibility fallback.

Design constraints (see plan / CLAUDE.md):
- Foundation leaf: stdlib + `memo.dashboard` (leaf) + lazy `memo.embedder_client`
  / `memo.session`. NO `memo.memory` import — stays import-cheap, never a brain
  verb on any surface.
- Hard budget: ≤5 snippets, one batched embed, `MEMO_GROUNDING_BUDGET_MS` (8000)
  wall-clock guard that bails writing nothing. Every failure is swallowed — Stop
  telemetry must never fail the turn.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Lexical containment ≥ this → grounded without paying for embedding.
_LEXICAL_HIGH = 0.6
# Lexical containment ≤ this with no embed signal → treat as not used.
_LEXICAL_LOW = 0.1
_MAX_SNIPPETS = 5
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
# Tiny stop set — these tokens carry no grounding signal and would inflate
# containment for any prose answer.
_STOP = frozenset(
    [
        "the",
        "and",
        "for",
        "that",
        "with",
        "this",
        "from",
        "have",
        "are",
        "was",
        "were",
        "has",
        "not",
        "but",
        "you",
        "your",
        "una",
        "los",
        "las",
        "del",
        "por",
        "con",
        "para",
        "como",
        "que",
        "está",
        "este",
        "esta",
        "más",
    ]
)


def _budget_ms() -> int:
    from memo.flags import flag_int

    v = flag_int("MEMO_GROUNDING_BUDGET_MS")
    return v if v is not None else 8000


def _salient_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP}


def _lexical_containment(snippet: str, answer_tokens: set[str]) -> float:
    snip = _salient_tokens(snippet)
    if not snip:
        return 0.0
    return len(snip & answer_tokens) / len(snip)


_CITE_RE = re.compile(r"\[([0-9a-f]{6,8})\]")


def cited_ids(answer: str) -> set[str]:
    """Short-id prefixes the assistant explicitly cited, e.g. ``[a1b2c3d4]``.

    Explicit citations are the strongest grounding signal — the model *told*
    us it used the memory (see CITE_INSTRUCTION in recall_logic).
    """
    return set(_CITE_RE.findall(answer or ""))


def match_cited(cited: set[str], session_ids: Iterable[str]) -> set[str]:
    """Full memory ids (recalled this session) matching a cited prefix.

    Membership in the session-recall set is the anti-false-positive gate:
    a random hex token in the answer never matches unless that exact memory
    was actually injected this session.
    """
    if not cited:
        return set()
    return {fid for fid in session_ids if any(fid.startswith(p) for p in cited)}


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def read_last_assistant_text(transcript_path: str | Path, *, max_chars: int = 8000) -> str:
    """Extract the final assistant message text from a Claude Code transcript
    JSONL. Best-effort: returns '' on any problem. Concatenates the text blocks
    of the last entry whose role/type is assistant."""
    try:
        p = Path(transcript_path).expanduser()
        if not p.is_file():
            return ""
        lines = p.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return ""
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") not in ("assistant", None) and obj.get("role") != "assistant":
            continue
        msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
        if (msg.get("role") or obj.get("role")) != "assistant":
            continue
        content = msg.get("content")
        parts: list[str] = []
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif isinstance(block, str):
                    parts.append(block)
        text = "\n".join(p for p in parts if p).strip()
        if text:
            return text[:max_chars]
    return ""


def collect_recent_tool_targets(
    transcript_path: str | Path, *, scan_lines: int = 40
) -> list[dict[str, str]]:
    """Extract the tool actions of the last turn from a Claude Code transcript:
    file paths (Read/Edit/Write/MultiEdit) and bash commands (Bash). Returns a
    list of {action, target} where action is opened_file|ran_command. Best-effort,
    bounded to the last `scan_lines` transcript entries. Claude-Code-shaped only;
    returns [] for clients without structured tool calls."""
    out: list[dict[str, str]] = []
    try:
        p = Path(transcript_path).expanduser()
        if not p.is_file():
            return out
        lines = p.read_text(encoding="utf-8").splitlines()[-scan_lines:]
    except (OSError, UnicodeDecodeError):
        return out
    for line in lines:
        line = line.strip()
        if not line or '"tool_use"' not in line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = str(block.get("name") or "")
            inp_raw = block.get("input")
            inp = inp_raw if isinstance(inp_raw, dict) else {}
            fp = inp.get("file_path") or inp.get("path") or inp.get("notebook_path")
            if fp:
                out.append({"action": "opened_file", "target": str(fp)})
            cmd = inp.get("command")
            if cmd and name.lower() == "bash":
                out.append({"action": "ran_command", "target": str(cmd)[:200]})
    return out


def _action_for_snippet(snippet: str, targets: list[dict[str, str]]) -> dict[str, str] | None:
    """If a recalled snippet names a path/command the turn acted on, return the
    matching action. A memory that mentions `foo/bar.py` and the turn Read
    `foo/bar.py` is a strong downstream-action signal."""
    snip = snippet or ""
    if not snip:
        return None
    snip_low = snip.lower()
    snip_tok = set(_TOKEN_RE.findall(snip_low))
    for t in targets:
        tgt = t["target"]
        if t["action"] == "opened_file":
            # match on basename or full path appearing in the snippet
            base = tgt.rsplit("/", 1)[-1].lower()
            if base and (base in snip_low or tgt.lower() in snip_low):
                return {"downstream_action": "opened_file", "action_evidence": tgt}
        else:  # ran_command — overlap of salient command tokens with the snippet
            cmd_tok = {x for x in _TOKEN_RE.findall(tgt.lower()) if x not in _STOP}
            if cmd_tok and len(cmd_tok & snip_tok) >= max(2, len(cmd_tok) // 2):
                return {"downstream_action": "ran_command", "action_evidence": tgt[:120]}
    return None


def _recalled_for_turn(state_dir: Path, session_id: str, turn: int) -> list[dict[str, Any]]:
    """The memories recalled for (session_id, turn): list of
    {id, snippet, score} pulled from recall_hook.log, falling back to the
    shared recall.log for older runtimes. No store read."""
    from memo.dashboard import read_recall_hook_log, read_recall_log

    out: dict[str, dict[str, Any]] = {}
    rows = read_recall_hook_log(state_dir, limit=2000)
    if not rows:
        rows = read_recall_log(state_dir, limit=2000)
    for row in rows:
        if row.get("session_id") != session_id or row.get("turn") != turn:
            continue
        for h in row.get("hits") or []:
            hid = h.get("id")
            if not hid:
                continue
            snippet = h.get("snippet") or h.get("title") or ""
            prev = out.get(hid)
            if prev is None or (snippet and not prev.get("snippet")):
                out[hid] = {"id": hid, "snippet": snippet, "score": h.get("score")}
    return list(out.values())


def _prompt_for_turn(state_dir: Path, session_id: str, turn: int) -> str:
    """The user prompt recorded for (session_id, turn) in recall_hook.log, for
    the topical-baseline embedding. '' when not found."""
    from memo.dashboard import read_recall_hook_log, read_recall_log

    rows = read_recall_hook_log(state_dir, limit=2000)
    if not rows:
        rows = read_recall_log(state_dir, limit=2000)
    for row in rows:
        if row.get("session_id") == session_id and row.get("turn") == turn:
            p = row.get("prompt")
            if p:
                return str(p)
    return ""


def score_turn(state_dir: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Score grounding for the just-finished exchange. Returns a small summary
    dict (for tests/logging) or None when nothing was scored. Never raises."""
    t0 = time.time()
    budget_s = _budget_ms() / 1000.0

    def _bail(reason: str, *, session_id: str | None = None, turn: int | None = None) -> dict[str, Any]:
        try:
            from memo.dashboard import append_grounding_diag_log

            append_grounding_diag_log(
                state_dir,
                reason=reason,
                session_id=session_id,
                turn=turn,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("grounding: failed to write diag log: %s", exc)
        out: dict[str, Any] = {"scored": 0, "bailed": reason}
        if session_id:
            out["session_id"] = session_id
        if turn is not None:
            out["turn"] = turn
        return out

    try:
        from memo.dashboard import append_grounding_log

        session_id = (payload.get("session_id") or "").strip()
        transcript_path = payload.get("transcript_path")
        if not session_id:
            return _bail("missing_session_id")
        if not transcript_path:
            return _bail("missing_transcript_path", session_id=session_id)

        # Resolve the turn the recall-hook stamped for this exchange.
        # Primary: recall_hook.log (written synchronously by the recall-hook,
        # never subject to the async-checkpoint race that leaves last_recall_turn=null
        # in the session snapshot). Fallback: session snapshot for older runtimes.
        from memo.dashboard import read_recall_hook_log

        _hook_rows = read_recall_hook_log(state_dir, limit=2000)
        turn: int | None = None
        for _row in reversed(_hook_rows):
            if _row.get("session_id") == session_id:
                _t = _row.get("turn")
                if isinstance(_t, int) and _t >= 0:
                    turn = _t
                    break
        if turn is None:
            from memo import session as _session

            snap = _session.get_session(state_dir, session_id) or {}
            _lrt = snap.get("last_recall_turn")
            if isinstance(_lrt, int):
                turn = _lrt
        if turn is None:
            return _bail("missing_last_recall_turn", session_id=session_id)

        recalled = _recalled_for_turn(state_dir, session_id, turn)
        if not recalled:
            return _bail("no_recalled_hits", session_id=session_id, turn=turn)
        recalled = recalled[:_MAX_SNIPPETS]

        answer = read_last_assistant_text(transcript_path)
        if not answer:
            return _bail("no_answer", session_id=session_id, turn=turn)
        answer_tokens = _salient_tokens(answer)
        # Explicit citations — validated against this session's recalled ids.
        cited_full: set[str] = set()
        cited8: set[str] = set()
        try:
            from memo import session as _session_cited

            _session_map = _session_cited.get_recalled_ids(state_dir, session_id)
            cited_full = match_cited(cited_ids(answer), _session_map.keys())
            # recall_hook.log truncates hit ids to 8 chars; normalise to 8-char
            # prefixes for comparison so the in-turn upgrade is never dead code.
            cited8 = {f[:8] for f in cited_full}
        except Exception as exc:
            import logging

            logging.getLogger(__name__).debug("grounding: cited-id match failed: %s", exc)
        _snap = locals().get("snap") or {}
        client = _snap.get("client") or "claude-code"
        question = _prompt_for_turn(state_dir, session_id, turn)
        # Downstream-action targets for this turn (Claude Code only; [] elsewhere).
        tool_targets = collect_recent_tool_targets(transcript_path)

        # Stage 1 — lexical (free).
        scored: list[dict[str, Any]] = []
        for m in recalled:
            lex = _lexical_containment(m.get("snippet") or "", answer_tokens)
            scored.append({"m": m, "lexical": lex, "embed": None, "specific": None, "method": "lexical"})

        # Stage 2 — embedding, single batch: answer + question + ALL snippets.
        # `specific = cos(answer, mem) - cos(question, mem)` separates real use
        # (answer matches the memory MORE than the topical baseline the question
        # already set) from same-topic overlap — the cause of the inflated rate.
        if (time.time() - t0) < budget_s:
            try:
                from memo import embedder_client

                snips = [recalled[i].get("snippet") or "" for i in range(len(recalled))]
                texts = [answer, question or "", *snips]
                vectors = embedder_client.embed(texts, state_dir=state_dir)
                if vectors and len(vectors) == len(texts):
                    avec, qvec = vectors[0], vectors[1]
                    has_q = bool(question and any(qvec))
                    for i in range(len(recalled)):
                        svec = vectors[i + 2]
                        cos_a = _cosine(avec, svec)
                        scored[i]["embed"] = cos_a
                        if has_q:
                            scored[i]["specific"] = max(0.0, cos_a - _cosine(qvec, svec))
                        scored[i]["method"] = "both" if scored[i]["lexical"] > 0 else "embed"
            except Exception as exc:
                import logging
                logging.getLogger(__name__).debug("grounding: embed scoring failed, using lexical-only: %s", exc)

        # Wall-clock guard: if we blew the budget, write nothing.
        if (time.time() - t0) >= budget_s:
            return _bail("budget", session_id=session_id, turn=turn)

        written = 0
        for entry in scored:
            rec = entry["m"]
            rid = str(rec.get("id") or "") if isinstance(rec, dict) else ""
            top = rec.get("score") if isinstance(rec, dict) else None
            lex_v = entry["lexical"]
            lex = float(lex_v) if isinstance(lex_v, (int, float)) else 0.0
            emb = entry["embed"]
            used = max(lex, float(emb)) if isinstance(emb, (int, float)) else lex
            method = str(entry["method"])
            if rid and rid[:8] in cited8:
                used = 1.0
                method = "cited"
            spec = entry["specific"]
            snippet = rec.get("snippet") if isinstance(rec, dict) else ""
            action = _action_for_snippet(str(snippet or ""), tool_targets)
            append_grounding_log(
                state_dir,
                session_id=session_id,
                turn=turn,
                recall_id=rid,
                used_score=used,
                method=method,
                client=str(client),
                answer_len=len(answer),
                recall_top_score=float(top) if isinstance(top, (int, float)) else None,
                specific_score=float(spec) if isinstance(spec, (int, float)) else None,
                downstream_action=action["downstream_action"] if action else None,
                action_evidence=action["action_evidence"] if action else None,
            )
            written += 1
        _scored8 = {
            s
            for s in (str(e["m"].get("id") or "")[:8] for e in scored if isinstance(e["m"], dict))
            if s
        }
        for fid in (f for f in cited_full if f[:8] not in _scored8):
            append_grounding_log(
                state_dir,
                session_id=session_id,
                turn=turn,
                recall_id=fid,
                used_score=1.0,
                method="cited",
                client=str(client),
                answer_len=len(answer),
            )
            written += 1
        return {"session_id": session_id, "turn": turn, "scored": written}
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("grounding: score_turn failed: %s", exc)
        return None
