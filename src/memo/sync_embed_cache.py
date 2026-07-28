"""Cross-machine embedding-cache export/import (sync bootstrap accelerator).

memo's `.md` memories sync between machines via git, but their EMBEDDINGS do
not — a fresh clone (`memo sync bootstrap`) or a machine catching up on pulls
must re-embed every new memory through MLX, which is the exact cold cost the
sync is supposed to spare. These helpers snapshot each memory's document (and
chunk) embedding as content-addressed rows next to the memories
(`embed_cache/<machine_id>.json`, so git carries them) and import a peer's
snapshot into `repo_embedding_cache` BEFORE the post-pull reindex — the
reindex then resolves through `_embed_cached` and issues ~zero embedder calls.

Design:
- Pairs are derived from the LIVE index, not the cache table: `meta.title` +
  `fts.body` are exactly what `_compose_for_embed` embedded at index time and
  the `vec` row is that embedding, so the pairs are correct by construction —
  and memories indexed through `save()` (which never populates the cache)
  export too.
- One shard per machine (filename = persisted device_id). Concurrent Macs
  never write the same file, so the sync rebase cannot conflict on it. The
  shard mirrors the N most-recently-updated durable parents
  (`MEMO_SYNC_EMBED_CACHE_MAX_ROWS`, chunks ride along) — a rolling window
  that bounds the file (a 2560-dim vector is ~13.7KB in base64; the full
  mature corpus would be tens of MB). Peers that sync regularly converge to
  full coverage anyway because their local cache persists; deletions and
  aged-out rows drop off on the next export.
- Reference-tier rows (bulk vault ingest) are excluded: the vault is not part
  of the sync corpus and its embeddings must not leak into the sync repo.
  Chunks OF durable memories (`extra.parent_id`) are included — the post-pull
  reindex re-embeds those too.
- Vectors travel as base64(packed float32) — the same bytes the `vec` table
  stores; JSON float lists would roughly triple the payload.
- Everything here is derived data. A missing, corrupt, or foreign-model shard
  degrades to today's behavior (re-embed locally) — never to an error.
- Export composes with the PURE `record._compose_for_embed` (never the facade
  override) so it can never trigger LLM context generation inside the sync
  hook, and it SKIPS entirely when this machine shows contextual-retrieval
  evidence (flag on, or a non-empty context cache — proof some stored vectors
  may carry an LLM context prefix the pure hash wouldn't match). The context
  cache lives in the same state_dir as memvec.db, so a state wipe resets both
  together and the guard stays truthful.
- Import validates every vector (finite, ~unit norm) — a poisoned or corrupt
  shard row is skipped, never stored.

Trust boundary (accepted residual, not a gap): import validates vector SHAPE
only — it does not cryptographically bind a shard row's content-hash key to its
vector value, so a peer with WRITE access to the shared sync git remote could
substitute a shape-valid vector and shift where a memory lands in embedding
space. That is the SAME trust boundary as the `.md` files themselves: an
attacker who can write the sync repo already controls memory content directly
(and more visibly). An HMAC would need a per-fleet secret shared out-of-band
across the user's machines — a setup decision, not a code fix, and it adds no
protection against the only attacker in scope. Operators who want the smaller
trust boundary set `MEMO_SYNC_EMBED_CACHE=0` to skip cache import entirely and
re-embed locally (derived data — correctness is unaffected).
"""

from __future__ import annotations

import base64
import json
import math
import re
import struct
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from memo.memory.record import _compose_for_embed as _compose_plain
from memo.util import sha256_full

if TYPE_CHECKING:
    from pathlib import Path

    from memo.config import Config
    from memo.memory import Memory
    from memo.store import VecStore

EMBED_CACHE_SCHEMA = "memo.sync.embed_cache.v1"

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def embed_cache_dir_for(cfg: Config) -> Path:
    """The `embed_cache/` directory — sibling of the memories dir, like `signal/`."""
    return cfg.memory_dir.parent / "embed_cache"


def _shard_path(cfg: Config, cache_dir: Path) -> Path:
    machine = _SAFE_NAME_RE.sub("_", str(getattr(cfg, "device_id", "") or "")) or "machine"
    return cache_dir / f"{machine}.json"


def _contextual_evidence(cfg: Config) -> bool:
    """True when stored vectors on this machine MAY carry an LLM context prefix
    — contextual retrieval is enabled now, or its context cache has entries
    (it was enabled at some indexing in this state_dir's lifetime). Exporting
    pure-compose hashes for such vectors would mint wrong (hash → vector)
    pairs, so the exporter must skip."""
    from memo.contextual_retrieval import context_cache_path, contextual_retrieval_enabled

    if contextual_retrieval_enabled():
        return True
    try:
        path = context_cache_path(cfg.state_dir)
        if path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            return bool(cached)
    except (OSError, ValueError):
        return True  # unreadable evidence — assume tainted, skip export
    return False


def _decode_shard_row(
    encoded: object, dims: int, shard_quant: str, shard_bpd: int
) -> list[float] | None:
    """Decode one base64 shard vector to a validated ~unit-norm float list.

    Returns None to skip the row (non-string, corrupt base64, wrong width, or a
    poisoned NaN/Inf/zero-norm vector that would fail the vec-table write and
    de-index the memory on reindex). int8 shards are dequantized (÷127) back to
    the ~unit float range; float32 bytes round-trip exactly (LE serialize_float32).
    """
    if not isinstance(encoded, str):
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return None
    if len(raw) != dims * shard_bpd:
        return None
    if shard_quant == "int8":
        vec = [x / 127.0 for x in struct.unpack(f"{dims}b", raw)]
    else:
        vec = list(struct.unpack(f"{dims}f", raw))
    norm = math.fsum(x * x for x in vec) ** 0.5
    if not math.isfinite(norm) or not (0.5 < norm < 1.5):
        return None
    return vec


def _shard_model_id(store: VecStore) -> str:
    """Shard identity used for cross-machine matching. Appends `+int8` under
    quantization so a lossy int8 shard is never merged into a float32 index
    (and vice-versa). The LOCAL `repo_embedding_cache` row key stays the plain
    `store.embedder_model`, so a warm rebuild still hits it with ~zero embeds."""
    return f"{store.embedder_model}+int8" if store.vec_quant == "int8" else store.embedder_model


def export_embed_cache(mem: Memory, cache_dir: Path) -> dict:
    """Write this machine's shard: `{sha256(embed_text): b64(f32 vector)}` for
    every durable memory (and durable-parented chunk) in the live index.

    The hash is computed with the PURE compose (title + body, no contextual
    prefix, no LLM) so export never generates anything; when this machine has
    contextual-retrieval evidence the export is skipped instead (see
    `_contextual_evidence`). Deterministic (sorted keys) and write-if-changed,
    so an unchanged corpus produces zero git churn. Returns
    ``{"rows", "written", "path"}`` (plus ``"skipped"`` when guarded).
    """
    from memo.flags import flag_int

    store = mem.store
    if _contextual_evidence(mem.cfg):
        return {"rows": 0, "written": False, "path": "", "skipped": "contextual-retrieval"}
    # Under int8 the vec blob is 1 B/dim, so the shard is ~4x smaller for free.
    bytes_per_dim = 1 if store.vec_quant == "int8" else 4
    pairs: dict[str, str] = {}
    for row in store.export_embed_rows(limit=flag_int("MEMO_SYNC_EMBED_CACHE_MAX_ROWS")):
        blob = store.get_embedding_blob(str(row["id"]))
        if blob is None or len(blob) != store.dims * bytes_per_dim:
            continue  # embed-pending or dims/quant drift — nothing exportable for this row
        text = _compose_plain(str(row["title"]), str(row["body"]))
        pairs[sha256_full(text)] = base64.b64encode(bytes(blob)).decode("ascii")

    payload = {
        "schema": EMBED_CACHE_SCHEMA,
        # `+int8` isolates a quantized shard so a float32 index never imports
        # lossy int8 vectors (and vice-versa) — cross-precision shards are
        # skipped whole and re-embedded locally.
        "model": _shard_model_id(store),
        "dims": store.dims,
        "quant": store.vec_quant,
        "rows": pairs,
    }
    data = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"

    shard = _shard_path(mem.cfg, cache_dir)
    try:
        unchanged = shard.read_text(encoding="utf-8") == data
    except OSError:
        unchanged = False
    if not unchanged:
        cache_dir.mkdir(parents=True, exist_ok=True)
        shard.write_text(data, encoding="utf-8")
    return {"rows": len(pairs), "written": not unchanged, "path": str(shard)}


def import_embed_cache(store: VecStore, cache_dir: Path) -> dict:
    """Merge every peer shard under `cache_dir` into `repo_embedding_cache`.

    Only shards matching the local `(model, dims)` import (a foreign-profile
    shard would only bloat the DB with vectors this machine can never serve).
    Rows already cached are skipped. Returns
    ``{"imported", "shards", "skipped_shards"}``.
    """
    out = {"imported": 0, "shards": 0, "skipped_shards": 0}
    if not cache_dir.is_dir():
        return out
    now = datetime.now(UTC).isoformat()
    for shard in sorted(cache_dir.glob("*.json")):
        try:
            doc = json.loads(shard.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out["skipped_shards"] += 1
            continue
        rows = doc.get("rows") if isinstance(doc, dict) else None
        if (
            not isinstance(rows, dict)
            or doc.get("schema") != EMBED_CACHE_SCHEMA
            or doc.get("model") != _shard_model_id(store)
            or doc.get("dims") != store.dims
        ):
            # A float32 shard vs an int8 index (or vice-versa) fails the
            # `+int8`-tagged model match and is skipped whole → re-embed locally.
            out["skipped_shards"] += 1
            continue
        shard_quant = "int8" if doc.get("quant") == "int8" else "off"
        shard_bpd = 1 if shard_quant == "int8" else 4
        out["shards"] += 1
        hashes = [h for h in rows if isinstance(h, str)]
        present = store.get_repo_embedding_cache(
            model=store.embedder_model, dims=store.dims, input_hashes=hashes
        )
        to_add: list[tuple[str, list[float]]] = []
        for input_hash in hashes:
            if input_hash in present:
                continue
            vec = _decode_shard_row(rows[input_hash], store.dims, shard_quant, shard_bpd)
            if vec is not None:
                to_add.append((input_hash, vec))
        if to_add:
            store.upsert_repo_embedding_cache(
                model=store.embedder_model,
                dims=store.dims,
                embeddings=to_add,
                created_at=now,
            )
            out["imported"] += len(to_add)
    return out
