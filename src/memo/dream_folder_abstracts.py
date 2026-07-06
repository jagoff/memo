"""Dream pass — one synthesis abstract per vault folder (K4, RAPTOR-lite).

Flat top-K recall can't answer "what is this vault section about". This
pass groups reference-tier rows by vault folder (the store path's parent
directory; ``#chunk-N`` rows fold into their parent file) and abstracts
each folder into one durable ``type=synthesis`` memory
(``synthesis_kind=folder_abstract``) with folder + member ids as
provenance. Existing abstracts are UPDATED in place when membership
changes (no churn) and skipped when unchanged, so each folder has at most
ONE abstract. OFF by default — MEMO_DREAM_FOLDER_ABSTRACTS_ENABLED.
Nightly only; never in the 5s recall hook.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import PurePosixPath
from typing import Any

_log = logging.getLogger(__name__)

_SYS = (
    "You summarize what one folder of a user's knowledge vault is about. "
    'Reply ONLY with JSON {"title": str, "summary": str}. The summary states '
    "the folder's subject matter and recurring themes in 3-5 sentences; "
    "no preamble, no markdown."
)


def _folder_of(store_path: str) -> str:
    """Parent folder of a reference row's store path ('' = vault root).
    Chunk rows ('<rel>#chunk-N') fold into their parent file's folder."""
    base = store_path.split("#chunk-", 1)[0]
    parent = str(PurePosixPath(base).parent)
    return "" if parent == "." else parent


def members_hash(folder: str, ids: list[str]) -> str:
    """Stable membership fingerprint. First 12 sorted ids → drift-tolerant:
    one new doc changes it (re-abstract), id ORDER never does."""
    core = [folder, *sorted(ids)[:12]]
    return hashlib.sha256("|".join(core).encode("utf-8")).hexdigest()[:16]


def collect_folders(mem: Any, *, min_members: int) -> list[dict[str, Any]]:
    """Group reference-tier rows by folder → [{folder, ids, titles}],
    biggest folder first. Chunk rows are skipped (their parent file row
    already represents the document)."""
    groups: dict[str, dict[str, Any]] = {}
    for r in mem.store.list_recent(limit=100_000):
        if r.get("type") != "reference":
            continue
        path = str(r.get("path") or "")
        if "#chunk-" in path:
            continue
        folder = _folder_of(path)
        g = groups.setdefault(folder, {"folder": folder, "ids": [], "titles": []})
        g["ids"].append(r["id"])
        g["titles"].append(str(r.get("title") or ""))
    out = [g for g in groups.values() if len(g["ids"]) >= min_members]
    out.sort(key=lambda g: (-len(g["ids"]), g["folder"]))
    return out


def _llm_abstract_folder(mem: Any, group: dict[str, Any]) -> dict[str, str] | None:
    import json as _json

    from memo.memory.record import chat_with_timeout

    titles = "\n".join(f"- {t[:120]}" for t in group["titles"][:40] if t)
    prompt = (
        f"Vault folder: {group['folder'] or '(vault root)'} "
        f"({len(group['ids'])} documents). Document titles:\n{titles}\n\n"
        "What is this folder about?"
    )
    out = chat_with_timeout(
        mem._ensure_chat(),
        timeout=30,
        model=mem.cfg.helper_model,
        messages=[{"role": "system", "content": _SYS}, {"role": "user", "content": prompt}],
        options={"temperature": 0.0, "max_tokens": 384, "thinking": False},
    )
    if out is None:
        return None
    text = ((out.get("message") or {}).get("content") or "").strip()
    try:
        obj = _json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or not obj.get("title") or not obj.get("summary"):
        return None
    return {"title": str(obj["title"])[:120], "body": str(obj["summary"])}


def _existing_abstract(mem: Any, folder: str) -> dict[str, Any] | None:
    import json as _json

    row = mem.store._conn.execute(
        "SELECT id, extra_json FROM meta WHERE type = 'synthesis' "
        "AND json_extract(extra_json, '$.abstract_folder') = ? LIMIT 1",
        (folder,),
    ).fetchone()
    if row is None:
        return None
    try:
        extra = _json.loads(row["extra_json"]) if row["extra_json"] else {}
    except (ValueError, TypeError):
        extra = {}
    return {"id": row["id"], "extra": extra if isinstance(extra, dict) else {}}


def run_folder_abstracts(
    cfg: Any,
    mem: Any,
    *,
    min_members: int = 5,
    max_folders: int = 5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One per-folder abstract pass. Never raises. OFF unless flagged."""
    from memo.flags import flag_bool

    res: dict[str, Any] = {"status": "noop", "abstracts": []}
    if not flag_bool("MEMO_DREAM_FOLDER_ABSTRACTS_ENABLED"):
        res["status"] = "disabled"
        return res
    try:
        groups = collect_folders(mem, min_members=min_members)
        if not groups:
            res["status"] = "skipped"
            return res
        for g in groups[:max_folders]:
            folder = g["folder"]
            mhash = members_hash(folder, g["ids"])
            existing = _existing_abstract(mem, folder)
            if existing and existing["extra"].get("abstract_members_hash") == mhash:
                res["abstracts"].append({"folder": folder, "status": "skip_unchanged"})
                continue
            if dry_run:
                res["abstracts"].append(
                    {"folder": folder, "status": "would_save", "members": len(g["ids"])}
                )
                continue
            synth = _llm_abstract_folder(mem, g)
            if not synth:
                res["abstracts"].append({"folder": folder, "status": "synth_failed"})
                continue
            extra = {
                "synthesis_kind": "folder_abstract",
                "abstract_folder": folder,
                "abstract_members_hash": mhash,
                "synthesis_sources": sorted(g["ids"])[:20],
            }
            try:
                if existing:
                    mem.update(
                        existing["id"], title=synth["title"], content=synth["body"], extra=extra
                    )
                    res["abstracts"].append({"folder": folder, "status": "updated"})
                else:
                    mem.save(
                        content=synth["body"],
                        type="synthesis",
                        title=synth["title"],
                        extra=extra,
                    )
                    res["abstracts"].append({"folder": folder, "status": "saved"})
            except Exception as exc:
                res["abstracts"].append(
                    {
                        "folder": folder,
                        "status": "save_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        res["status"] = "done"
    except Exception as exc:
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res
