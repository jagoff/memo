"""`memo invalidate <pattern> --reason <text>` — reversible bulk weakening.

One event (a stack migration, a port change) can invalidate a whole class of
memories at once; today each needs its own individual contradiction. This
command bulk-weakens every memory matching a case-insensitive substring
pattern (title, tags, body):

  * confidence penalty in memory_health (ranking-side: score x confidence)
  * `_invalidated` tag + reason/date stamp in extra (markdown-side, survives
    reindex and syncs with the .md)

Reversible: every applied run writes a receipt to
`state_dir/invalidate/<ts>.json` with each memory's prior confidence;
`memo invalidate --undo` restores confidences and removes the stamps.

Without `--yes` the command only PREVIEWS matches — it never mutates and
never prompts (safe under MEMO_NONINTERACTIVE).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config
from memo.flags import flag_float

_log = logging.getLogger(__name__)

INVALIDATED_TAG = "_invalidated"
_SCAN_CAP = 10_000  # corpus scan ceiling; --limit caps the MATCHES


def _receipt_dir(cfg: Config) -> Path:
    return cfg.state_dir / "invalidate"


def _match(pattern: str, title: str, tags: list[str], body: str) -> bool:
    """Case-insensitive substring match over title, tags and body."""
    p = pattern.lower()
    return (
        p in (title or "").lower()
        or any(p in (t or "").lower() for t in tags)
        or p in (body or "").lower()
    )


@click.command(name="invalidate")
@click.argument("pattern", required=False)
@click.option("--reason", default="", help="Why these memories are being weakened (stored in extra).")
@click.option("--yes", is_flag=True, help="Apply. Without it the command only previews matches.")
@click.option("--limit", default=500, type=int, help="Max memories weakened per run (default 500).")
@click.option(
    "--undo",
    "undo_ts",
    is_flag=False,
    flag_value="latest",
    default=None,
    help="Revert a previous run (bare --undo = the latest receipt; or pass its <ts>).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit result as JSON.")
def invalidate_cmd(
    pattern: str | None,
    reason: str,
    yes: bool,
    limit: int,
    undo_ts: str | None,
    as_json: bool,
) -> None:
    """Bulk-weaken all memories matching PATTERN (reversible via --undo)."""
    cfg = Config.from_env()
    mem = _get_memory(cfg)
    if undo_ts is not None:
        _do_undo(mem, cfg, undo_ts, as_json)
        return
    if not pattern:
        raise click.UsageError("PATTERN is required (or pass --undo).")
    if yes and not reason:
        raise click.UsageError("--reason is required when applying with --yes.")

    matched = [
        r for r in mem.list(limit=_SCAN_CAP) if _match(pattern, r.title, r.tags, r.body or "")
    ][:limit]

    if not matched:
        console.print(json.dumps({"pattern": pattern, "matched": 0}) if as_json else "No memories match.", highlight=False)
        return

    if not yes:
        if as_json:
            console.print(
                json.dumps(
                    {"pattern": pattern, "matched": len(matched), "applied": False,
                     "ids": [r.id for r in matched]}
                ),
                highlight=False,
            )
        else:
            console.print(
                f"Would weaken {len(matched)} memories (re-run with --yes --reason to apply):",
                highlight=False,
            )
            for r in matched:
                console.print(f"  [{r.id[:8]}] {r.title}", highlight=False)
        return

    ids = [r.id for r in matched]
    prior = mem.store.get_health_batch(ids)
    now = datetime.now(UTC).isoformat()
    penalty = flag_float("MEMO_INVALIDATE_PENALTY")
    penalty = 0.3 if penalty is None else penalty
    entries: list[dict[str, Any]] = [
        {"id": i, "prev_confidence": (prior.get(i) or {}).get("confidence", 1.0)} for i in ids
    ]

    mem.store.penalize_confidence_batch(ids, delta=penalty)
    stamped = 0
    for r in matched:
        new_extra = dict(r.extra or {})
        new_extra["invalidated_reason"] = reason
        new_extra["invalidated_at"] = now
        try:
            # pure tags/extra update — Memory.update() skips the embedder
            mem.update(r.id, tags=[*r.tags, INVALIDATED_TAG], extra=new_extra)
            stamped += 1
        except Exception as exc:
            _log.warning("invalidate: stamp failed for %s: %s", r.id[:8], exc)

    ts = now.replace("-", "").replace(":", "")[:15]  # filesystem-safe run id
    rd = _receipt_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    receipt = {"ts": now, "pattern": pattern, "reason": reason, "penalty": penalty, "entries": entries}
    (rd / f"{ts}.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    payload = {"pattern": pattern, "matched": len(matched), "stamped": stamped,
               "penalty": penalty, "receipt": ts, "applied": True}
    console.print(
        json.dumps(payload)
        if as_json
        else f"Weakened {len(matched)} memories (penalty {penalty}). Undo: memo invalidate --undo {ts}",
        highlight=False,
    )


def _do_undo(mem: Any, cfg: Config, undo_ts: str, as_json: bool) -> None:
    rd = _receipt_dir(cfg)
    receipts = sorted(rd.glob("*.json")) if rd.is_dir() else []
    if not receipts:
        raise click.ClickException("No invalidate receipts found.")
    path = receipts[-1] if undo_ts == "latest" else rd / f"{undo_ts}.json"
    if not path.is_file():
        raise click.ClickException(f"No receipt {path.name}.")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    entries = receipt.get("entries") or []

    mem.store.set_confidence_batch(
        [(e["id"], float(e["prev_confidence"])) for e in entries]
    )
    restored = 0
    for e in entries:
        rec = mem.get(e["id"])
        if rec is None:
            continue
        new_extra = dict(rec.extra or {})
        new_extra.pop("invalidated_reason", None)
        new_extra.pop("invalidated_at", None)
        try:
            mem.update(
                rec.id,
                tags=[t for t in rec.tags if t != INVALIDATED_TAG],
                extra=new_extra,
            )
            restored += 1
        except Exception as exc:
            _log.warning("invalidate --undo: unstamp failed for %s: %s", rec.id[:8], exc)

    path.rename(path.with_suffix(".json.undone"))
    payload = {"undone": path.stem, "restored": restored}
    console.print(json.dumps(payload) if as_json else f"Restored {restored} memories from {path.stem}.", highlight=False)
