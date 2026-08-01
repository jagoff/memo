"""Embed-cache sync — A's memory embeddings reach B via git so B's post-pull
reindex issues ZERO embedder calls (the cross-machine bootstrap accelerator).

Same harness as test_sync_git.py: local bare remote + two clones, embedder
stubbed to 4-dim, no network, no MLX.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from memo.config import Config
from memo.memory import Memory
from memo.sync_embed_cache import (
    EMBED_CACHE_SCHEMA,
    embed_cache_dir_for,
    export_embed_cache,
    import_embed_cache,
)
from memo.sync_git import sync_once, sync_pull
from tests.operational_authority import authorize_test_config


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)


def _stub_embed(self, inputs):
    out = []
    for s in inputs:
        h = sum(ord(c) for c in s) % 4
        v = [0.0] * 4
        v[h] = 1.0
        out.append(v)
    return out


def _make_clone(remote: Path, where: Path) -> Path:
    subprocess.run(
        ["git", "clone", str(remote), str(where)], check=True, capture_output=True, text=True
    )
    _git(where, "config", "user.email", "t@t.t")
    _git(where, "config", "user.name", "t")
    (where / "memorias").mkdir(exist_ok=True)
    return where


def _mem_for(clone: Path, state: Path, monkeypatch) -> Memory:
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    cfg = authorize_test_config(
        Config(
            data_dir=clone / "memorias",
            state_dir=state,
            embedder_dims=4,
            embedder_model="stub",
            reranker_enabled=False,
        )
    )
    return Memory(cfg)


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    r = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(r)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(r), str(seed)], check=True, capture_output=True)
    _git(seed, "config", "user.email", "t@t.t")
    _git(seed, "config", "user.name", "t")
    (seed / "memorias").mkdir()
    (seed / "memorias" / ".gitkeep").write_text("")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "push", "origin", "main")
    return r


def _shard_files(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob("*.json"))


@pytest.mark.float32_precision  # asserts doc["model"] == "stub"; int8 stores tag shards "stub+int8" by design
def test_export_import_roundtrip(remote: Path, tmp_path: Path, monkeypatch):
    """Export derives (hash → vector) pairs from A's live index; importing them
    into a fresh store makes the exact vectors available under the same keys."""
    a = _make_clone(remote, tmp_path / "a")
    mem_a = _mem_for(a, tmp_path / "state_a", monkeypatch)
    mem_a.save(content="usar sqlite-vec como store", title="Decision store", type="decision")
    mem_a.save(content="los tests corren sin MLX", title="Fact tests", type="fact")

    cache_dir = embed_cache_dir_for(mem_a.cfg)
    out = export_embed_cache(mem_a, cache_dir)
    assert out["rows"] == 2
    assert out["written"] is True

    shards = _shard_files(cache_dir)
    assert len(shards) == 1
    doc = json.loads(shards[0].read_text())
    assert doc["schema"] == EMBED_CACHE_SCHEMA
    assert doc["model"] == "stub"
    assert doc["dims"] == 4
    assert len(doc["rows"]) == 2

    b = _make_clone(remote, tmp_path / "b")
    mem_b = _mem_for(b, tmp_path / "state_b", monkeypatch)
    imported = import_embed_cache(mem_b.store, cache_dir)
    assert imported["imported"] == 2
    assert imported["shards"] == 1

    # Identity, not just shape: every hash must map to the EXACT vector A's
    # index holds for the text that produces that hash (catches key/vector
    # pairing bugs and pack/unpack order bugs that permutation-blind checks
    # would miss).
    import struct

    from memo.memory.record import _compose_for_embed
    from memo.util import sha256_full

    expected = {}
    for row in mem_a.store.export_embed_rows():
        h = sha256_full(_compose_for_embed(row["title"], row["body"]))
        blob = mem_a.store.get_embedding_blob(row["id"])
        expected[h] = list(struct.unpack("4f", blob))
    hashes = list(doc["rows"])
    assert set(hashes) == set(expected)
    cached = mem_b.store.get_repo_embedding_cache(model="stub", dims=4, input_hashes=hashes)
    assert cached == expected


def test_pull_reindexes_with_zero_embed_calls(remote: Path, tmp_path: Path, monkeypatch):
    """The money path: B pulls A's new memory and indexes it entirely from the
    imported cache — the embedder is never called."""
    a = _make_clone(remote, tmp_path / "a")
    mem_a = _mem_for(a, tmp_path / "state_a", monkeypatch)
    saved = mem_a.save(content="commits en español", title="Pref commits", type="preference")
    assert sync_once(mem_a.cfg, mem_a.store, mem_a)["pushed"] is True

    calls = {"n": 0}

    def _counting_embed(self, inputs):
        calls["n"] += 1
        return _stub_embed(self, inputs)

    b = _make_clone(remote, tmp_path / "b")
    mem_b = _mem_for(b, tmp_path / "state_b", monkeypatch)
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _counting_embed)

    out = sync_pull(mem_b.cfg, mem_b.store, mem_b)
    assert out["pulled"] is True
    assert out["reindexed"]["added"] >= 1
    assert out["embed_cache_imported"] >= 1
    assert mem_b.store.get(saved.id) is not None
    assert calls["n"] == 0


def test_import_skips_foreign_model_shard(remote: Path, tmp_path: Path, monkeypatch):
    a = _make_clone(remote, tmp_path / "a")
    mem_a = _mem_for(a, tmp_path / "state_a", monkeypatch)
    cache_dir = embed_cache_dir_for(mem_a.cfg)
    cache_dir.mkdir(parents=True)
    (cache_dir / "other.json").write_text(
        json.dumps(
            {
                "schema": EMBED_CACHE_SCHEMA,
                "model": "some-other-model",
                "dims": 4,
                "rows": {"ab" * 32: "AAAAAAAAAAAAAAAAAAAAAA=="},
            }
        )
    )
    out = import_embed_cache(mem_a.store, cache_dir)
    assert out["imported"] == 0
    assert out["skipped_shards"] == 1


def test_import_survives_corrupt_shard(remote: Path, tmp_path: Path, monkeypatch):
    a = _make_clone(remote, tmp_path / "a")
    mem_a = _mem_for(a, tmp_path / "state_a", monkeypatch)
    cache_dir = embed_cache_dir_for(mem_a.cfg)
    cache_dir.mkdir(parents=True)
    (cache_dir / "corrupt.json").write_text("{not json")
    out = import_embed_cache(mem_a.store, cache_dir)
    assert out["imported"] == 0
    assert out["skipped_shards"] == 1


def test_export_excludes_reference_tier(remote: Path, tmp_path: Path, monkeypatch):
    """Vault-ingested reference rows must never leak into the sync repo —
    only durable memories (and their chunks) export."""
    a = _make_clone(remote, tmp_path / "a")
    mem_a = _mem_for(a, tmp_path / "state_a", monkeypatch)
    mem_a.save(content="nota durable", title="Nota durable")
    mem_a.store.upsert(
        id_="ref00000000000000000000000000000000",
        path="vault/some-doc.md#chunk-0",
        title="vault doc §1/9",
        type_="reference",
        tags=[],
        created="2026-07-20T00:00:00Z",
        updated="2026-07-20T00:00:00Z",
        body_hash="x" * 16,
        embedding=[1.0, 0.0, 0.0, 0.0],
        extra=None,
        body_text="bulk vault content",
    )
    out = export_embed_cache(mem_a, embed_cache_dir_for(mem_a.cfg))
    assert out["rows"] == 1


def test_export_covers_chunks_of_durable_parents(remote: Path, tmp_path: Path, monkeypatch):
    """With chunk ingest on, a long durable memory's chunk rows export too —
    the post-pull reindex re-embeds those as well, so they must hit the cache."""
    monkeypatch.setenv("MEMO_CHUNK_INGEST", "1")
    a = _make_clone(remote, tmp_path / "a")
    mem_a = _mem_for(a, tmp_path / "state_a", monkeypatch)
    body = "\n\n".join(f"## Sección {i}\n\n" + (f"contenido {i} " * 120) for i in range(4))
    mem_a.save(content=body, title="Larga")
    mem_a.reindex(force=True)  # chunk emission happens on reindex
    n_rows = len(mem_a.store.export_embed_rows())
    assert n_rows > 1  # parent + at least one chunk
    out = export_embed_cache(mem_a, embed_cache_dir_for(mem_a.cfg))
    assert out["rows"] == n_rows


@pytest.mark.float32_precision  # shard doc is hand-crafted model="stub" (float32); int8 store expects
# "stub+int8" and by-design skips the whole (foreign-profile) shard, not just poisoned rows
def test_import_skips_invalid_vectors(remote: Path, tmp_path: Path, monkeypatch):
    """A poisoned shard row (zero-norm / NaN) is skipped; valid rows import."""
    import base64
    import struct

    a = _make_clone(remote, tmp_path / "a")
    mem_a = _mem_for(a, tmp_path / "state_a", monkeypatch)
    cache_dir = embed_cache_dir_for(mem_a.cfg)
    cache_dir.mkdir(parents=True)
    good = base64.b64encode(struct.pack("4f", 0.0, 1.0, 0.0, 0.0)).decode()
    zero = base64.b64encode(struct.pack("4f", 0.0, 0.0, 0.0, 0.0)).decode()
    nan = base64.b64encode(struct.pack("4f", float("nan"), 1.0, 0.0, 0.0)).decode()
    (cache_dir / "peer.json").write_text(
        json.dumps(
            {
                "schema": EMBED_CACHE_SCHEMA,
                "model": "stub",
                "dims": 4,
                "rows": {"aa" * 32: good, "bb" * 32: zero, "cc" * 32: nan},
            }
        )
    )
    out = import_embed_cache(mem_a.store, cache_dir)
    assert out["imported"] == 1
    cached = mem_a.store.get_repo_embedding_cache(
        model="stub", dims=4, input_hashes=["aa" * 32, "bb" * 32, "cc" * 32]
    )
    assert list(cached) == ["aa" * 32]


def test_export_skipped_under_contextual_retrieval(remote: Path, tmp_path: Path, monkeypatch):
    """Contextual retrieval makes stored vectors carry an LLM prefix the pure
    hash wouldn't match — the exporter must refuse to mint those pairs."""
    monkeypatch.setattr("memo.contextual_retrieval.contextual_retrieval_enabled", lambda: True)
    a = _make_clone(remote, tmp_path / "a")
    mem_a = _mem_for(a, tmp_path / "state_a", monkeypatch)
    mem_a.save(content="con contexto", title="Ctx")
    out = export_embed_cache(mem_a, embed_cache_dir_for(mem_a.cfg))
    assert out["skipped"] == "contextual-retrieval"
    assert out["rows"] == 0
    assert not embed_cache_dir_for(mem_a.cfg).exists()


def test_export_is_idempotent(remote: Path, tmp_path: Path, monkeypatch):
    a = _make_clone(remote, tmp_path / "a")
    mem_a = _mem_for(a, tmp_path / "state_a", monkeypatch)
    mem_a.save(content="nota idempotente", title="Nota")
    cache_dir = embed_cache_dir_for(mem_a.cfg)
    first = export_embed_cache(mem_a, cache_dir)
    second = export_embed_cache(mem_a, cache_dir)
    assert first["written"] is True
    assert second["written"] is False
    assert first["rows"] == second["rows"]


def test_export_caps_to_most_recent_rows(remote: Path, tmp_path: Path, monkeypatch):
    """MEMO_SYNC_EMBED_CACHE_MAX_ROWS bounds the shard to the newest durable
    parents — the shard-size guard for mature corpora."""
    monkeypatch.setenv("MEMO_SYNC_EMBED_CACHE_MAX_ROWS", "2")
    a = _make_clone(remote, tmp_path / "a")
    mem_a = _mem_for(a, tmp_path / "state_a", monkeypatch)
    mem_a.save(content="la más vieja", title="Vieja", created="2026-01-01T00:00:00Z")
    mem_a.save(content="la del medio", title="Media")
    mem_a.save(content="la más nueva", title="Nueva")
    out = export_embed_cache(mem_a, embed_cache_dir_for(mem_a.cfg))
    assert out["rows"] == 2


def test_flag_off_skips_export_on_sync(remote: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMO_SYNC_EMBED_CACHE", "0")
    a = _make_clone(remote, tmp_path / "a")
    mem_a = _mem_for(a, tmp_path / "state_a", monkeypatch)
    mem_a.save(content="nota sin cache", title="Nota sin cache")
    assert sync_once(mem_a.cfg, mem_a.store, mem_a)["pushed"] is True
    assert not embed_cache_dir_for(mem_a.cfg).exists()
