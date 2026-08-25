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
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from memo.flags import flag_int

_DIRNAME = "emitted"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_PREFIX_CHARS = 200


@dataclass(frozen=True)
class Entry:
    """One emission: memory ``id``, hash ``h`` and length ``n`` of the text that
    went out, prefix hash ``hp`` over its first ``_PREFIX_CHARS`` characters (see
    ``partition``), the batch ``ref`` it went out under, unix seconds ``t``, and
    ``src`` (``hook`` | ``mcp``).

    ``hp`` is last and defaults to ``None`` so entries written before this field
    existed still deserialize in ``read()`` — an older entry simply carries no
    prefix hash, which ``partition`` treats as unsafe-to-digest-on-length-alone."""

    id: str
    h: str
    n: int
    ref: str
    t: int
    src: str
    hp: str | None = None

    @classmethod
    def for_text(cls, memory_id: str, text: str, ref: str, t: int, src: str) -> Entry:
        """Build an entry from the text ACTUALLY emitted — the only correct source
        for ``h``, ``n`` and ``hp``. All three must describe the same string or the
        monotonic-emission rule digests content the model never saw; this is the
        one call site that can't get that wrong. Prefer this over the plain
        constructor everywhere except rebuilding an ``Entry`` from a disk line
        (``read``), where ``h``/``n``/``hp`` are already-computed, trusted fields."""
        return cls(
            id=memory_id,
            h=emitted_hash(text),
            n=len(text),
            ref=ref,
            t=t,
            src=src,
            hp=emitted_hash(text[:_PREFIX_CHARS]),
        )


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

    The temp filename includes our own pid: a fixed ``path.name + ".tmp"``
    would let two processes racing into ``_trim`` at once interleave writes
    into the same temp path before either reaches ``os.replace``. Per-pid
    names make that impossible without adding any locking.
    """
    cap = _cap()
    if cap <= 0:
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= cap * 2:  # amortise: only rewrite once we are well over
            return
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
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
            hp = obj.get("hp")
            out[str(obj["id"])] = Entry(
                id=str(obj["id"]),
                h=str(obj["h"]),
                n=int(obj["n"]),
                ref=str(obj["ref"]),
                t=int(obj["t"]),
                src=str(obj.get("src") or ""),
                hp=str(hp) if hp is not None else None,
            )
        except Exception:  # noqa: S112  # torn/malformed line — skip it, keep reading
            continue
    return out


def reset(state_dir: Path, session_id: str) -> bool:
    """Drop this session's ledger. Returns whether a file was actually removed.

    Called at the compaction boundary: once the window is rewritten, memo can no
    longer claim anything is in it. Idempotent — PreCompact double-fires against
    the plugin copy.

    Also drops the digested-ids sidecar (`record_digested`/`digested_ids`): a
    digest pointer's claim ("this body is already up there") expires at
    compaction exactly like the main ledger's does, so a `memo_get` after
    compaction on an id digested BEFORE it must not still count as a recovery
    from a digest the model can no longer see. Keeping the two files' lifetimes
    tied here is the correctness argument, not just consistency.

    Deliberately does NOT touch the counters sidecar (`bump`/`stats`, see
    `_counters_path`) — reviewed and confirmed in task-8-findings-r1.md F2.
    The ledger's claim expires at compaction, but the counters are the
    promotion gate's per-session measurement and must span compactions; only
    `prune`'s age-based GC may remove them, once the session itself is gone.
    Do not "fix" this asymmetry.
    """
    removed = False
    try:
        path = ledger_path(state_dir, session_id)
        if path.is_file():
            path.unlink()
            removed = True
    except Exception:  # noqa: S110  # fail-open: a removal failure just leaves the file
        pass
    try:
        digested_path = _digested_path(state_dir, session_id)
        if digested_path.is_file():
            digested_path.unlink()
            removed = True
    except Exception:  # noqa: S110  # fail-open: a removal failure just leaves the file
        pass
    return removed


@dataclass(frozen=True)
class Partition:
    """``full`` keeps its hits verbatim; ``digest`` are hits the caller should
    render as {id, title, ref}. ``suppressed_chars`` is the character count of
    the digested hits' text (``text_of(hit)`` only, not the whole serialized
    row).

    NOT the promotion gate's numerator, despite an earlier version of this
    docstring's claim: the gate's real ``tokens_suppressed`` is computed
    independently in ``server_common.apply_ledger``, over the whole
    serialized hit (id/title/tags/timestamps/extra/... included), because
    charging only the text field undercounted the real saving by roughly 16x
    on a real payload (see ``apply_ledger``'s own F1 comment). Nothing in this
    codebase currently reads ``suppressed_chars`` outside this module's own
    tests -- kept as a cheap, informational figure for a caller that wants a
    body-text-only view, not as a wired-up measurement."""

    full: list[Any]
    digest: list[Any]
    suppressed_chars: int


def partition(
    hits: Sequence[Any],
    known: dict[str, Entry],
    *,
    text_of: Callable[[Any], str],
    id_of: Callable[[Any], str],
) -> Partition:
    """Split hits into what still needs sending and what the caller should digest
    to a pointer. ``partition`` performs no I/O of its own — it only classifies;
    the caller is the one that sends the full hits and appends fresh ledger
    entries for them.

    A hit is digested only when the ledger proves the model has already seen at
    least this much of it — the *monotonic-emission rule*:

        same hash                                  -> identical text already up there
        new_len <= known.n AND prefix hash matches  -> a safe shortening of text
                                                        already up there

    An exact hash match is logically subsumed by the length+prefix check below it
    (equal text has equal length and an equal prefix) but is kept as its own fast
    path — it is the common case and skips the prefix hash computation. Do not
    "simplify" it away.

    The length arm alone is not enough: a body that was EDITED and happens to be
    shorter would pass ``new_len <= known.n`` while describing text the model
    never saw. The prefix hash closes that hole — it catches an edit that changes
    the start of a body, while still accepting a prefix-preserving shortening
    (trailing truncation, where the model has already seen a superset). An entry
    with no recorded prefix hash (``hp is None`` — an older ledger line written
    before this field existed) is always sent in full: unknown means unsafe means
    full, and that direction costs tokens, never correctness. A residual gap is
    accepted and left unclosed: an edit that changes only the middle or end of a
    body AND shortens it can still be digested; that is bounded by the ledger
    reset at the compaction boundary.

    Anything that fails both checks, including a *longer* rendering of the same
    memory, is sent in full. The length/prefix asymmetry against a longer
    rendering is the point: emitting less than the model has seen is free, but
    digesting past content it has never seen is silent data loss. The recall hook
    truncates to MEMO_RECALL_BODY_CHARS (400) while memo_ask may emit far more,
    so a longer re-rendering of the same memory is routine, not hypothetical.
    """
    full: list[Any] = []
    digest: list[Any] = []
    suppressed = 0
    for hit in hits:
        text = text_of(hit) or ""
        prior = known.get(id_of(hit))
        can_digest = prior is not None and (
            emitted_hash(text) == prior.h
            or (
                len(text) <= prior.n
                and prior.hp is not None
                and emitted_hash(text[:_PREFIX_CHARS]) == prior.hp
            )
        )
        if can_digest:
            digest.append(hit)
            suppressed += len(text)
        else:
            full.append(hit)
    return Partition(full=full, digest=digest, suppressed_chars=suppressed)


def _digested_path(state_dir: Path, session_id: str) -> Path:
    return Path(state_dir) / _DIRNAME / f"{_safe(session_id)}.digested.jsonl"


def record_digested(state_dir: Path, session_id: str, ids: Sequence[str]) -> None:
    """Record ids actually rendered as an ``already_in_context`` pointer in some
    tool response this session -- i.e. ``partition`` put them in ``digest``, not
    ``full``.

    Deliberately separate from ``append``/``read``: those record every EMISSION
    (a body actually sent, whether by the recall hook or an MCP tool sending it
    in full), not which of those emissions were later suppressed to a pointer.
    An id can sit in ``read()``'s result because the hook injected it, or
    because a tool sent it in full, without the model ever having seen a digest
    for it -- a `memo_get` on such an id is an ordinary first read, not a
    recovery from a digest, and must not be counted as one (see
    `server_core_records._record_ledger_recovery`).

    Same fail-open envelope as `append`: a counter that cannot be written costs
    the promotion gate a measurement, never a caller's response.
    """
    if not ids:
        return
    try:
        path = _digested_path(state_dir, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        now = int(time.time())
        with path.open("a", encoding="utf-8") as fh:
            for memory_id in ids:
                fh.write(json.dumps({"id": memory_id, "t": now}, separators=(",", ":")) + "\n")
        _trim(path)
    except Exception:
        return


def digested_ids(state_dir: Path, session_id: str) -> set[str]:
    """Ids that have been rendered as a digest pointer (an ``already_in_context``
    entry) at least once this session -- the correct membership test for
    "was this id ever digested", unlike `read()` (see `record_digested`).

    Unparseable lines are skipped, matching `read()`'s tolerance of a torn tail
    from a concurrent writer."""
    out: set[str] = set()
    try:
        raw = _digested_path(state_dir, session_id).read_text(encoding="utf-8")
    except Exception:
        return out
    cap = _cap()
    lines = raw.splitlines()
    if cap > 0:
        lines = lines[-cap:]
    for line in lines:
        try:
            obj = json.loads(line)
            out.add(str(obj["id"]))
        except Exception:  # noqa: S112  # torn/malformed line -- skip it, keep reading
            continue
    return out


def _counters_path(state_dir: Path, session_id: str) -> Path:
    return Path(state_dir) / _DIRNAME / f"{_safe(session_id)}.counters.json"


def bump(
    state_dir: Path,
    session_id: str,
    *,
    digests_served: int = 0,
    tokens_suppressed: int = 0,
    tokens_digest: int = 0,
    get_after_digest: int = 0,
    tokens_recovered: int = 0,
) -> None:
    """Accumulate the numbers ``stats()`` reports for the promotion gate.

    Every argument is a DELTA to add to this session's running total, already
    converted to tokens by the caller (``mcp_budget.est_tokens``, the house
    4-chars-per-token estimate) -- this module stays stdlib-only per its
    leaf-module contract (see the module docstring), so it never re-derives
    that conversion itself; ``server_common.apply_ledger`` and
    ``server_core_records.memo_get`` are the two call sites that hold real
    text and do the conversion.

    Read-modify-write against a small per-session JSON file, same fail-open
    envelope as ``append``: a counter this cannot persist costs the
    promotion gate a measurement, never a caller's response.
    """
    try:
        path = _counters_path(state_dir, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cur = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
        deltas = {
            "digests_served": digests_served,
            "tokens_suppressed": tokens_suppressed,
            "tokens_digest": tokens_digest,
            "get_after_digest": get_after_digest,
            "tokens_recovered": tokens_recovered,
        }
        for key, delta in deltas.items():
            cur[key] = int(cur.get(key, 0)) + delta
        path.write_text(json.dumps(cur, separators=(",", ":")), encoding="utf-8")
    except Exception:
        return


def stats(state_dir: Path, session_id: str) -> dict[str, int]:
    """This session's emission-ledger scorecard, in tokens.

    ``net_saved_est`` is the number the promotion gate reads::

        tokens_suppressed - tokens_digest - tokens_recovered

    what was NOT sent, minus what the digest stubs themselves cost, minus
    what recovering from them via ``memo_get`` cost. It can go negative on
    purpose: a session that calls ``memo_get`` on every digested id pays the
    recovery cost on top of a suppression that was never actually realised,
    and the gate has to be able to see that rather than have it netted away.
    """
    try:
        cur = json.loads(_counters_path(state_dir, session_id).read_text(encoding="utf-8"))
    except Exception:
        cur = {}
    suppressed = int(cur.get("tokens_suppressed", 0))
    digest_cost = int(cur.get("tokens_digest", 0))
    recovery = int(cur.get("tokens_recovered", 0))
    return {
        "entries": len(read(state_dir, session_id)),
        "digests_served": int(cur.get("digests_served", 0)),
        "tokens_suppressed": suppressed,
        "tokens_digest": digest_cost,
        "memo_get_after_digest": int(cur.get("get_after_digest", 0)),
        "net_saved_est": suppressed - digest_cost - recovery,
    }


def prune(state_dir: Path, *, max_age_s: int) -> int:
    """Remove ledger files whose session ended long ago. Sessions leave no
    close signal, so mtime age is the only available liveness proxy.

    F2 (task-8 review): collects all three file shapes `emitted/` can hold,
    not just the `.jsonl` ledger itself -- `bump`'s counters sidecar
    (``<sid>.counters.json``) and any crash-mid-rewrite leftover from
    `_trim` (``<sid>.jsonl.<pid>.tmp``) matched no glob before this fix and
    leaked forever, one pair of files per Claude Code session id ever
    started. Same liveness proxy (mtime age) applies to all three.

    Does NOT touch ``reset()``, and never should: the ledger's
    already-in-context claim expires at compaction (what the window can
    still see), but the counters file is the promotion gate's per-session
    measurement and must survive a compaction reset -- deleting it there
    would zero out `net_saved_est` mid-session, before the session that
    earned it has even ended. Only age-based pruning, here, once the
    session itself is long gone, may remove the counters file.
    """
    removed = 0
    now = time.time()
    try:
        base = Path(state_dir) / _DIRNAME
        entries = [
            *base.glob("*.jsonl"),
            *base.glob("*.counters.json"),
            *base.glob("*.jsonl.*.tmp"),
        ]
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
