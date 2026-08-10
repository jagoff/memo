"""What memo has already put into this session's context window.

memo produces the same memory bodies from two places — the recall hook on every
``UserPromptSubmit`` and the MCP read tools — pulling from one corpus with one
embedder. Nothing tracked emissions across those surfaces, so a body could enter
the window three or four times in a session and stay there.

This module is the ledger. It records what was *emitted*, not what is stored:
the hash is over the text that actually went out, plus its length. That
distinction is the whole correctness argument — see ``partition``.

Leaf module by contract, like ``recall_dedup``: stdlib only, no store access, no
MLX, no flag reads beyond its own. The recall hook imports it inside a 5s budget
and every call from there is fail-open.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from memo.flags import flag_int

_DIRNAME = "emitted"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


@dataclass(frozen=True)
class Entry:
    """One emission: memory ``id``, hash ``h`` and length ``n`` of the text that
    went out, the batch ``ref`` it went out under, unix seconds ``t``, and
    ``src`` (``hook`` | ``mcp``)."""

    id: str
    h: str
    n: int
    ref: str
    t: int
    src: str

    @classmethod
    def for_text(cls, memory_id: str, text: str, ref: str, t: int, src: str) -> Entry:
        """Build an entry from the text ACTUALLY emitted — the only correct source
        for ``h`` and ``n``. ``h`` and ``n`` must describe the same string or the
        monotonic-emission rule digests content the model never saw; this is the
        one call site that can't get that wrong. Prefer this over the plain
        constructor everywhere except rebuilding an ``Entry`` from a disk line
        (``read``), where ``h``/``n`` are already-computed, trusted fields."""
        return cls(id=memory_id, h=emitted_hash(text), n=len(text), ref=ref, t=t, src=src)


def emitted_hash(text: str) -> str:
    """8 hex chars over the emitted text. Collisions only ever matter between
    versions of the *same* id, so the space is per-memory and tiny."""
    return hashlib.blake2s(text.encode("utf-8"), digest_size=4).hexdigest()


def mint_ref(ids: Sequence[str], t: int, *, prefix: str = "memo-r") -> str:
    """A short token naming one emission batch, echoed in the payload so a
    digest can point at a specific earlier message without turn numbers.
    Order-insensitive: the same set of ids at the same second is the same ref."""
    seed = ",".join(sorted(ids)) + f"@{t}"
    return f"{prefix}/{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:6]}"


def _safe(session_id: str) -> str:
    """Session ids reach us from an env var and a hook payload — both external
    input, and both used to build a path. Sanitise rather than trust.

    ``.`` alone stays allowed (it is common in real session ids), but a run of
    two or more dots collapses to a single ``_`` — otherwise the path-separator
    substitution below can leave a literal ``..`` sitting in the filename even
    though it can no longer reach a parent directory (no ``/`` survives)."""
    cleaned = _UNSAFE.sub("_", session_id or "unknown")
    cleaned = re.sub(r"\.\.+", "_", cleaned)
    return cleaned[:120] or "unknown"


def ledger_path(state_dir: Path, session_id: str) -> Path:
    return Path(state_dir) / _DIRNAME / f"{_safe(session_id)}.jsonl"


def _cap() -> int:
    """Entry cap, FIFO. Wrapped in its own try: flag resolution reads memo's
    Markdown config off disk, and a corrupt config file (e.g. non-UTF-8 bytes)
    must not make ``read()``/``append()`` raise — every caller here counts on
    this never doing anything but returning an int."""
    try:
        value = flag_int("MEMO_EMITTED_LEDGER_MAX")
    except Exception:
        return 500
    return 500 if value is None else max(0, value)


def append(state_dir: Path, session_id: str, entries: Sequence[Entry]) -> None:
    """Append entries. Fail-open: a ledger that cannot be written costs tokens,
    never correctness, so every failure here is silent by design.

    No locking. The recall hook and the MCP server both write, but each line is
    a single short ``O_APPEND`` write, which is atomic; ``read`` tolerates a
    torn tail regardless.
    """
    if not entries:
        return
    try:
        path = ledger_path(state_dir, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(asdict(entry), separators=(",", ":")) + "\n")
        _trim(path)
    except Exception:
        return


def _trim(path: Path) -> None:
    """FIFO the file back under the cap.

    Rewrites via a temp file + ``os.replace`` in the same directory, so a
    concurrent reader/writer never observes a half-written file and a crash
    mid-rewrite can't truncate it — ``os.replace`` is atomic on POSIX.

    This does NOT close the write race between the recall hook and the MCP
    server: a concurrent ``append`` landed between our read and our replace is
    silently dropped by this rewrite. That is intentionally left open — the
    5s hook budget can't afford a lock, and the cost of losing an entry is
    just a re-emitted body (the pre-feature baseline), never a correctness
    problem.
    """
    cap = _cap()
    if cap <= 0:
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= cap * 2:  # amortise: only rewrite once we are well over
            return
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(lines[-cap:]) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        return


def read(state_dir: Path, session_id: str) -> dict[str, Entry]:
    """Latest entry per memory id. Unparseable lines are skipped: a concurrent
    writer can leave a torn tail, and a torn tail must not blind the reader."""
    out: dict[str, Entry] = {}
    try:
        raw = ledger_path(state_dir, session_id).read_text(encoding="utf-8")
    except Exception:
        return out
    cap = _cap()
    lines = raw.splitlines()
    if cap > 0:
        lines = lines[-cap:]
    for line in lines:
        try:
            obj = json.loads(line)
            out[str(obj["id"])] = Entry(
                id=str(obj["id"]),
                h=str(obj["h"]),
                n=int(obj["n"]),
                ref=str(obj["ref"]),
                t=int(obj["t"]),
                src=str(obj.get("src") or ""),
            )
        except Exception:  # noqa: S112  # torn/malformed line — skip it, keep reading
            continue
    return out


def reset(state_dir: Path, session_id: str) -> bool:
    """Drop this session's ledger. Returns whether a file was actually removed.

    Called at the compaction boundary: once the window is rewritten, memo can no
    longer claim anything is in it. Idempotent — PreCompact double-fires against
    the plugin copy.
    """
    try:
        path = ledger_path(state_dir, session_id)
        if path.is_file():
            path.unlink()
            return True
    except Exception:  # noqa: S110  # fail-open: a removal failure just leaves the file
        pass
    return False


def prune(state_dir: Path, *, max_age_s: int) -> int:
    """Remove ledgers whose session ended long ago. Sessions leave no close
    signal, so age is the only available liveness proxy."""
    removed = 0
    now = time.time()
    try:
        entries = list((Path(state_dir) / _DIRNAME).glob("*.jsonl"))
    except Exception:
        return 0
    for path in entries:
        try:
            if now - os.stat(path).st_mtime > max_age_s:
                path.unlink()
                removed += 1
        except Exception:  # noqa: S112  # one file's stat/unlink failure skips only that file
            continue
    return removed
