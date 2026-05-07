"""Save-side ambient capture — Phase B.

Hook fires on every assistant turn (Stop event), reads the just-finished
exchange from the transcript, asks the helper LLM (Qwen2.5-3B) to
extract any actionable insights, dedups against the existing corpus,
and saves the survivors with auto-derived metadata.

## Pipeline

```
Stop event
   │
   ▼  read transcript_path JSONL → last (user, assistant) exchange
   │
   ▼  pre-filter (cheap): skip empty / too-short / pure-tool turns
   │
   ▼  helper LLM extract → JSON [{title, type, body, tags}, ...]
   │
   ▼  dedup: embed each candidate, near-search, drop if max_sim > 0.85
   │
   ▼  save survivors via Memory.save()
```

## State file

`~/.local/share/memo/last-capture.json` tracks the hash of the last
processed assistant message so re-firing on the same turn (e.g. the
user runs `/clear` mid-stream, or two Stop hooks race) doesn't
double-extract.

## Why dedup with the embedder, not the title

Two memorias with different titles can describe the same fact. The
embedder is the only signal that catches "same fact, different
phrasing". Threshold 0.85 is empirical: cosine sim between
near-paraphrases is typically 0.85-0.95 with Qwen3-Embedding;
genuinely-distinct memorias score below 0.75 even on the same topic.

## Failure modes

All swallowed silently. Capture is opportunistic — a hook that fails
to extract is no worse than the pre-Phase-B world. Exception: if the
user explicitly sets `MEMO_CAPTURE_DEBUG=1`, errors print to stderr.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Trigger keywords — pre-filter pass. Cheap regex check before paying
# the helper LLM cost. Permissive; better to send to the LLM and have
# it return [] than to skip a real insight on a false negative.
_TRIGGER_PATTERNS = (
    "decid", "fix", "bug", "error", "issue", "from now on", "siempre",
    "nunca", "regla", "preferenc", "discover", "turns out", "result",
    "shippe", "merged", "deploy", "config", "instal", "uninstal",
    "migrate", "switch to", "use ", "usá", "uso ahora", "should",
    "porque", "because", "why ", "fail", "broke", "rompi", "crash",
    "perform", "latenc", "warm", "cold", "model", "embed", "rerank",
    "commit", "branch", "test", "regress",
)


_EXTRACT_SYSTEM_PROMPT = """You read one conversation turn between a developer and an AI coding assistant. Your job: extract ACTIONABLE INSIGHTS the developer would want to remember in their personal memory archive — and ONLY those.

EXTRACT:
- Decisions made with rationale ("we'll use X because Y")
- Bugs found with root cause + fix (not just "fixed it")
- Preferences expressed ("from now on, always X" / "never Y")
- Discoveries / non-obvious facts ("X turns out to require Y")
- Commands / config that worked ("to do X, run Y")

DO NOT extract:
- Mid-process status updates ("checking…", "looking at…", "let me…")
- Speculation ("we could…", "if we wanted…")
- Code snippets shown but not adopted
- Generic tutorials, documentation summaries
- Pleasantries, conversational filler

For each insight, output a JSON object with:
- "title": ≤80 chars, no period at end, descriptive of the insight
- "type": one of "decision", "bug", "preference", "fact", "note"
- "body": 2-5 sentences. INCLUDE: what the insight is, why it matters, and how to apply it. Be specific (file paths, numbers, model names) when relevant.
- "tags": 3-6 lowercase tags (project, technology, domain)

Output ONLY a JSON array. Empty array `[]` if nothing notable.
NO markdown fences. NO commentary. NO preamble."""


def _state_file(state_dir: Path) -> Path:
    return state_dir / "last-capture.json"


def _load_state(state_dir: Path) -> dict[str, Any]:
    f = _state_file(state_dir)
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state_dir: Path, state: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    _state_file(state_dir).write_text(json.dumps(state), encoding="utf-8")


def _read_last_exchange(transcript_path: Path) -> tuple[str, str] | None:
    """Walk the JSONL transcript backwards to find the last (user,
    assistant) exchange. Returns (user_text, assistant_text) or None
    if the transcript doesn't yield a complete pair (e.g. user just
    typed and the assistant hasn't responded yet).

    A "turn" in Claude Code can contain multiple assistant messages
    interleaved with tool_use / tool_result entries. The function
    concatenates ALL assistant text blocks that come AFTER the most
    recent user message — that's the assistant's full response to the
    user's last prompt. Without this, the parser would surface only
    the trailing message of a multi-message turn (often a brief
    "running this" status update with no insight), missing the
    substantive prose earlier in the same response.

    The transcript schema is Claude Code's internal format: each line
    a JSON object with `type` ("user" / "assistant") and `message`
    containing `content` which is either a string or a list of blocks
    (tool_use, text, etc.).
    """
    if not transcript_path.is_file():
        return None
    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    # Pre-parse: keep only user/assistant entries with non-empty text.
    parsed: list[tuple[str, str]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = obj.get("type") or obj.get("role")
        if role not in ("user", "assistant"):
            continue
        msg = obj.get("message", obj)
        content = msg.get("content") if isinstance(msg, dict) else None
        text = _extract_text(content)
        if text:
            parsed.append((role, text))

    if not parsed:
        return None

    # Find the LAST user message; everything assistant after it forms
    # the response. The user msg before that is the prompt.
    last_user_idx: int | None = None
    for i in range(len(parsed) - 1, -1, -1):
        if parsed[i][0] == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return None

    user_text = parsed[last_user_idx][1]
    assistant_chunks = [t for r, t in parsed[last_user_idx + 1 :] if r == "assistant"]
    if not assistant_chunks:
        return None
    assistant_text = "\n\n".join(assistant_chunks)
    return user_text, assistant_text


def _extract_text(content: Any) -> str:
    """Pull the plain text out of a Claude Code message content. Skips
    tool_use blocks, tool_result blocks, image blocks. Concatenates all
    text/markdown content with newlines.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                t = block.get("text", "")
                if t:
                    out.append(t.strip())
            # Skip tool_use, tool_result, image, thinking — not useful
            # for insight extraction. The text blocks carry the
            # assistant's prose explanations and the user's prompts.
        return "\n\n".join(out).strip()
    return ""


def _passes_prefilter(text: str, min_chars: int = 200) -> bool:
    """Cheap keyword + length check before paying the LLM cost."""
    if len(text) < min_chars:
        return False
    lower = text.lower()
    return any(p in lower for p in _TRIGGER_PATTERNS)


def _hash_assistant(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def extract_insights(
    helper_chat: Any, helper_model: str, user_text: str, assistant_text: str,
) -> list[dict[str, Any]]:
    """Run the helper LLM. Returns a list of insight dicts; empty on
    parse failure or model refusal. The helper is the small Qwen2.5-3B
    so latency is bounded (~1-3s warm).
    """
    user_block = (
        f"USER:\n{user_text[:4000]}\n\nASSISTANT:\n{assistant_text[:8000]}"
    )
    try:
        resp = helper_chat.chat(
            model=helper_model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_block},
            ],
            options={"temperature": 0.0, "seed": 42, "num_predict": 800},
        )
        raw = (resp.get("message") or {}).get("content") or ""
    except Exception:
        return []

    raw = raw.strip()
    if not raw:
        return []
    # Handle the occasional fenced response — strip fences before parse.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json\n"):
            raw = raw[5:]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        body = (item.get("body") or "").strip()
        type_ = (item.get("type") or "note").strip()
        tags = item.get("tags") or []
        if not title or not body:
            continue
        if not isinstance(tags, list):
            tags = []
        out.append({
            "title": title[:80],
            "type": type_,
            "body": body,
            "tags": [str(t).lower().strip() for t in tags if t],
        })
    return out


def is_near_duplicate(
    memory: Any, candidate: dict[str, Any], threshold: float = 0.85,
) -> bool:
    """Return True if the candidate is semantically near a record
    already in the corpus. Uses pure vec search (not hybrid+rerank) —
    the dedup decision is about embedding similarity, not the
    reranker's joint judgement."""
    composed = f"{candidate['title']}\n\n{candidate['body']}"
    try:
        emb = memory.embedder.embed_query(composed)
        rows = memory.store.search(emb, limit=1)
    except Exception:
        return False
    if not rows:
        return False
    top_score = rows[0].get("score")
    return top_score is not None and top_score >= threshold


def run_capture(
    transcript_path: Path,
    *,
    debug: bool = False,
) -> dict[str, Any]:
    """Top-level entry: read transcript, extract, dedup, save.
    Returns a result summary dict so the CLI can print + the tests can
    assert on counts. All errors absorbed (logged to stderr in debug)."""
    from memo.config import Config
    from memo.memory import Memory

    cfg = Config.from_env()
    state = _load_state(cfg.state_dir)

    pair = _read_last_exchange(transcript_path)
    if pair is None:
        return {"status": "no_pair"}
    user_text, assistant_text = pair

    # Idempotence — same assistant message hash → skip.
    h = _hash_assistant(assistant_text)
    if state.get("last_hash") == h:
        return {"status": "duplicate_turn"}

    if not _passes_prefilter(assistant_text):
        # Stamp the state anyway so we don't keep re-checking the same turn.
        state["last_hash"] = h
        _save_state(cfg.state_dir, state)
        return {"status": "no_trigger"}

    # Lazy heavy imports: only paid past pre-filter.
    mem = Memory(cfg)
    if mem._chat is None:  # type: ignore[attr-defined]
        from memo.llm import MLXChat
        mem._chat = MLXChat()  # type: ignore[attr-defined]

    insights = extract_insights(
        mem._chat, cfg.helper_model, user_text, assistant_text,  # type: ignore[attr-defined]
    )
    if debug:
        print(f"# memo capture: {len(insights)} candidate(s)", file=sys.stderr)

    saved: list[str] = []
    skipped_dup = 0
    for cand in insights:
        if is_near_duplicate(mem, cand):
            skipped_dup += 1
            if debug:
                print(f"# memo capture: skip dup '{cand['title']}'", file=sys.stderr)
            continue
        try:
            rec = mem.save(
                content=cand["body"], title=cand["title"],
                type_=cand["type"], tags=cand["tags"],
            )
            saved.append(rec.id)
            if debug:
                print(f"# memo capture: saved [{rec.id[:8]}] {rec.title}", file=sys.stderr)
        except Exception as exc:
            if debug:
                print(f"# memo capture: save failed: {exc}", file=sys.stderr)

    state["last_hash"] = h
    _save_state(cfg.state_dir, state)
    return {
        "status": "ok",
        "candidates": len(insights),
        "saved": saved,
        "skipped_dup": skipped_dup,
    }
