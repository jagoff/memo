"""Grounding detector — did the answer actually USE the recalled memorias?

Runs at the Stop hook (`memo capture-stop` → `grounding.score_turn`), AFTER the
answer is generated, inside the 30s async budget. For the turn's recalled
memorias (cached as `hits[].snippet` in recall.log by the recall-hook) it scores
how much each one shows up in the assistant's final answer:

  1. lexical containment (free) — salient snippet tokens present in the answer.
  2. embedding cosine (only on the ambiguous middle band) — via the already-warm
     daemon embedder (`embedder_client.embed`, symmetric/doc side, no query
     prefix). Catches paraphrase.

`used_score = max(lexical, embed_cosine)`. One `grounding.log` row per recalled
memoria, keyed by `(session_id, turn, recall_id)` — the recall→use ledger that
`dashboard.grounded_rate` joins against recall.log.

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

    return flag_int("MEMO_GROUNDING_BUDGET_MS") or 8000


def _salient_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP}


def _lexical_containment(snippet: str, answer_tokens: set[str]) -> float:
    snip = _salient_tokens(snippet)
    if not snip:
        return 0.0
    return len(snip & answer_tokens) / len(snip)


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
    matching action. A memoria that mentions `foo/bar.py` and the turn Read
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
    """The memorias recalled for (session_id, turn): list of
    {id, snippet, score} pulled from recall.log. No store read."""
    from memo.dashboard import read_recall_log

    out: dict[str, dict[str, Any]] = {}
    for row in read_recall_log(state_dir, limit=400):
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


def score_turn(state_dir: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Score grounding for the just-finished exchange. Returns a small summary
    dict (for tests/logging) or None when nothing was scored. Never raises."""
    t0 = time.time()
    budget_s = _budget_ms() / 1000.0
    try:
        from memo.dashboard import append_grounding_log

        session_id = (payload.get("session_id") or "").strip()
        transcript_path = payload.get("transcript_path")
        if not session_id or not transcript_path:
            return None

        # Resolve the turn the recall-hook stamped for this exchange.
        from memo import session as _session

        snap = _session.get_session(state_dir, session_id) or {}
        turn = snap.get("last_recall_turn")
        if not isinstance(turn, int):
            return None

        recalled = _recalled_for_turn(state_dir, session_id, turn)
        if not recalled:
            return None
        recalled = recalled[:_MAX_SNIPPETS]

        answer = read_last_assistant_text(transcript_path)
        if not answer:
            return None
        answer_tokens = _salient_tokens(answer)
        client = snap.get("client") or "claude-code"
        # Downstream-action targets for this turn (Claude Code only; [] elsewhere).
        tool_targets = collect_recent_tool_targets(transcript_path)

        # Stage 1 — lexical (free).
        scored: list[dict[str, Any]] = []
        ambiguous: list[int] = []
        for i, m in enumerate(recalled):
            lex = _lexical_containment(m.get("snippet") or "", answer_tokens)
            entry = {"m": m, "lexical": lex, "embed": None, "method": "lexical"}
            scored.append(entry)
            if lex < _LEXICAL_HIGH and lex > _LEXICAL_LOW:
                ambiguous.append(i)
            elif lex <= _LEXICAL_LOW:
                # Still embed the near-zero band IF budget allows — paraphrase
                # can ground with little lexical overlap.
                ambiguous.append(i)

        # Stage 2 — embedding, only on the ambiguous band, single batch.
        if ambiguous and (time.time() - t0) < budget_s:
            try:
                from memo import embedder_client

                texts = [answer] + [recalled[i].get("snippet") or "" for i in ambiguous]
                vectors = embedder_client.embed(texts, state_dir=state_dir)
                if vectors and len(vectors) == len(texts):
                    avec = vectors[0]
                    for j, i in enumerate(ambiguous):
                        cos = _cosine(avec, vectors[j + 1])
                        scored[i]["embed"] = cos
                        scored[i]["method"] = "both" if scored[i]["lexical"] > 0 else "embed"
            except Exception:
                pass  # lexical-only fallback; never fail the turn

        # Wall-clock guard: if we blew the budget, write nothing.
        if (time.time() - t0) >= budget_s:
            return {"session_id": session_id, "turn": turn, "scored": 0, "bailed": "budget"}

        written = 0
        for entry in scored:
            rec = entry["m"]
            rid = str(rec.get("id") or "") if isinstance(rec, dict) else ""
            top = rec.get("score") if isinstance(rec, dict) else None
            lex_v = entry["lexical"]
            lex = float(lex_v) if isinstance(lex_v, (int, float)) else 0.0
            emb = entry["embed"]
            used = max(lex, float(emb)) if isinstance(emb, (int, float)) else lex
            snippet = rec.get("snippet") if isinstance(rec, dict) else ""
            action = _action_for_snippet(str(snippet or ""), tool_targets)
            append_grounding_log(
                state_dir,
                session_id=session_id,
                turn=turn,
                recall_id=rid,
                used_score=used,
                method=str(entry["method"]),
                client=str(client),
                answer_len=len(answer),
                recall_top_score=float(top) if isinstance(top, (int, float)) else None,
                downstream_action=action["downstream_action"] if action else None,
                action_evidence=action["action_evidence"] if action else None,
            )
            written += 1
        return {"session_id": session_id, "turn": turn, "scored": written}
    except Exception:
        return None
