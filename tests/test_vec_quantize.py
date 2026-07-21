"""int8 vector quantization (MEMO_VEC_QUANTIZE=int8).

Exercises the vec0 store, the DDL-derived quant guard, the quant-flip rebuild,
and the sync-shard int8 round-trip — all without the MLX runtime (pure
sqlite-vec + hand-built unit vectors).
"""

from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path

import pytest

from memo.errors import StorageError
from memo.sqlite_compat import import_sqlite_vec
from memo.store import VecStore

serialize_float32 = import_sqlite_vec().serialize_float32

DIMS = 8


def _unit(*xs: float) -> list[float]:
    padded = list(xs) + [0.0] * (DIMS - len(xs))
    norm = math.sqrt(sum(x * x for x in padded)) or 1.0
    return [x / norm for x in padded]


def _row(store: VecStore, id_: str, emb: list[float]) -> None:
    store.upsert(
        id_=id_,
        path=f"memory/{id_}.md",
        title=id_,
        type_="fact",
        tags=[],
        created="2026-01-01",
        updated="2026-01-01",
        body_hash=id_,
        embedding=emb,
        body_text=f"body {id_}",
    )


# --------------------------------------------------------------------------- #
# helper unit tests                                                           #
# --------------------------------------------------------------------------- #
def test_bind_helpers_collapse_to_noop_when_off(tmp_path: Path) -> None:
    s = VecStore(tmp_path / "off.db", dims=DIMS, vec_quant="off")
    try:
        assert s._vec_dtype_ddl() == "FLOAT"
        assert s._vec_bind_new() == "?"
        assert s._vec_bind_stored() == "?"
    finally:
        s.close()


def test_bind_helpers_int8(tmp_path: Path) -> None:
    s = VecStore(tmp_path / "i8.db", dims=DIMS, vec_quant="int8")
    try:
        assert s._vec_dtype_ddl() == "int8"
        assert s._vec_bind_new() == "vec_quantize_int8(vec_f32(?), 'unit')"
        assert s._vec_bind_stored() == "vec_int8(?)"
        assert s._vec_table_dtype("vec") == "int8"
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# round-trip: int8 stores 1 B/dim and preserves cosine ranking               #
# --------------------------------------------------------------------------- #
def test_int8_blob_is_one_byte_per_dim(tmp_path: Path) -> None:
    off = VecStore(tmp_path / "off.db", dims=DIMS, vec_quant="off")
    i8 = VecStore(tmp_path / "i8.db", dims=DIMS, vec_quant="int8")
    try:
        _row(off, "a", _unit(1, 0, 0))
        _row(i8, "a", _unit(1, 0, 0))
        off_blob = off.get_embedding_blob("a")
        i8_blob = i8.get_embedding_blob("a")
        assert off_blob is not None and i8_blob is not None
        assert len(off_blob) == DIMS * 4
        assert len(i8_blob) == DIMS * 1
    finally:
        off.close()
        i8.close()


def test_int8_search_ranking_matches_float32(tmp_path: Path) -> None:
    vectors = {"a": _unit(1, 0.1), "b": _unit(0, 1), "c": _unit(0.9, 0.4)}
    query = _unit(0.95, 0.1)

    def top_ids(mode: str) -> list[str]:
        s = VecStore(tmp_path / f"{mode}.db", dims=DIMS, vec_quant=mode)
        try:
            for id_, v in vectors.items():
                _row(s, id_, v)
            return [h["id"] for h in s.search(query, limit=3)]
        finally:
            s.close()

    assert top_ids("int8") == top_ids("off")


def test_int8_dequant_roundtrip_is_unit_norm(tmp_path: Path) -> None:
    s = VecStore(tmp_path / "i8.db", dims=DIMS, vec_quant="int8")
    try:
        _row(s, "a", _unit(0.6, 0.5, 0.4, 0.3))
        blob = s.get_embedding_blob("a")
        assert blob is not None
        doc = s.unpack_embedding(blob)
        assert len(doc) == DIMS
        assert 0.5 < math.sqrt(sum(x * x for x in doc)) < 1.5
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# DDL-derived guard: a precision mismatch at open raises (both directions)    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("built", "reopened"),
    [("off", "int8"), ("int8", "off")],
)
def test_quant_mismatch_open_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, built: str, reopened: str
) -> None:
    # A real (non-stub) model + no SKIP env so the guard is not bypassed.
    monkeypatch.delenv("MEMO_SKIP_MODEL_VERSION_CHECK", raising=False)
    db = tmp_path / "vec.db"
    s = VecStore(db, dims=DIMS, embedder_model="realmodel", vec_quant=built)
    _row(s, "a", _unit(1, 0))
    s.close()

    with pytest.raises(StorageError, match="storage precision mismatch"):
        VecStore(db, dims=DIMS, embedder_model="realmodel", vec_quant=reopened)


def test_rebuild_flip_bypasses_guard_via_skip_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "vec.db"
    s = VecStore(db, dims=DIMS, embedder_model="realmodel", vec_quant="off")
    s.close()
    # `memo reindex --rebuild` sets this so it can open a mismatched store.
    monkeypatch.setenv("MEMO_SKIP_MODEL_VERSION_CHECK", "1")
    s2 = VecStore(db, dims=DIMS, embedder_model="realmodel", vec_quant="int8")
    s2.close()  # no raise


# --------------------------------------------------------------------------- #
# quant-flip rebuild recreates ONLY `vec`; repo_vec (reference tier) survives #
# --------------------------------------------------------------------------- #
def test_rebuild_flip_preserves_repo_vec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "vec.db"
    off = VecStore(db, dims=DIMS, embedder_model="realmodel", vec_quant="off")
    _row(off, "m1", _unit(1, 0))
    # Seed a reference-tier repo_vec row directly (bypassing the repo API).
    with off._tx() as cx:
        cx.execute(
            "INSERT INTO repo_vec (id, repo_id, embedding) VALUES (?, ?, ?)",
            ("r1", "repo1", serialize_float32(_unit(0, 1))),
        )
    off.close()

    monkeypatch.setenv("MEMO_SKIP_MODEL_VERSION_CHECK", "1")
    i8 = VecStore(db, dims=DIMS, embedder_model="realmodel", vec_quant="int8")
    try:
        i8.replace_memory_index(
            [
                {
                    "id_": "m1",
                    "path": "memory/m1.md",
                    "title": "m1",
                    "type_": "fact",
                    "tags": [],
                    "created": "2026-01-01",
                    "updated": "2026-01-01",
                    "body_hash": "m1",
                    "embedding": _unit(1, 0),
                    "body_text": "body m1",
                }
            ]
        )
        assert i8._vec_table_dtype("vec") == "int8"
        # repo_vec untouched by a quant-only flip.
        repo_n = i8._conn.execute("SELECT COUNT(*) FROM repo_vec").fetchone()[0]
        assert repo_n == 1
        assert i8._vec_table_dtype("repo_vec") == "off"
        # `vec` repopulated from the markdown row.
        m1_blob = i8.get_embedding_blob("m1")
        assert m1_blob is not None
        assert len(m1_blob) == DIMS * 1
    finally:
        i8.close()


# --------------------------------------------------------------------------- #
# sync shard: int8 round-trips; cross-precision shards are skipped whole      #
# --------------------------------------------------------------------------- #
def _int8_shard(cache_dir: Path, *, model: str, ihash: str, vec: list[float]) -> None:
    from memo.sync_embed_cache import EMBED_CACHE_SCHEMA

    q = [max(-127, min(127, round(x * 127))) for x in vec]
    raw = struct.pack(f"{len(q)}b", *q)
    payload = {
        "schema": EMBED_CACHE_SCHEMA,
        "model": f"{model}+int8",
        "dims": DIMS,
        "quant": "int8",
        "rows": {ihash: base64.b64encode(raw).decode("ascii")},
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "peer.json").write_text(json.dumps(payload), encoding="utf-8")


def test_sync_int8_shard_imports_into_int8_store(tmp_path: Path) -> None:
    from memo.sync_embed_cache import import_embed_cache

    cache_dir = tmp_path / "embed_cache"
    _int8_shard(cache_dir, model="realmodel", ihash="h1", vec=_unit(0.6, 0.5, 0.4))

    store = VecStore(tmp_path / "i8.db", dims=DIMS, embedder_model="realmodel", vec_quant="int8")
    try:
        out = import_embed_cache(store, cache_dir)
        assert out["shards"] == 1
        assert out["imported"] == 1
        cached = store.get_repo_embedding_cache(
            model="realmodel", dims=DIMS, input_hashes=["h1"]
        )
        assert "h1" in cached
        assert 0.5 < math.sqrt(sum(x * x for x in cached["h1"])) < 1.5
    finally:
        store.close()


def test_sync_int8_shard_skipped_by_float32_store(tmp_path: Path) -> None:
    from memo.sync_embed_cache import import_embed_cache

    cache_dir = tmp_path / "embed_cache"
    _int8_shard(cache_dir, model="realmodel", ihash="h1", vec=_unit(0.6, 0.5, 0.4))

    store = VecStore(tmp_path / "off.db", dims=DIMS, embedder_model="realmodel", vec_quant="off")
    try:
        out = import_embed_cache(store, cache_dir)
        # `realmodel+int8` != `realmodel` → skipped whole (re-embed locally).
        assert out["shards"] == 0
        assert out["skipped_shards"] == 1
        assert out["imported"] == 0
    finally:
        store.close()


def test_shard_model_id_tags_int8(tmp_path: Path) -> None:
    from memo.sync_embed_cache import _shard_model_id

    off = VecStore(tmp_path / "off.db", dims=DIMS, embedder_model="m", vec_quant="off")
    i8 = VecStore(tmp_path / "i8.db", dims=DIMS, embedder_model="m", vec_quant="int8")
    try:
        assert _shard_model_id(off) == "m"
        assert _shard_model_id(i8) == "m+int8"
    finally:
        off.close()
        i8.close()
