"""High-level Memory — save/search/list/get/delete with stub embedder.

These tests stub out `MLXEmbedder.embed` so they run on any platform
without loading the real model. The actual MLX-on-Metal smoke is in
`test_smoke_mlx.py` (gated by `requires_mlx`).
"""

from __future__ import annotations

import pytest

from mem_lmx.config import Config
from mem_lmx.memory import Memory


@pytest.fixture
def mem_with_stub(tmp_cfg: Config, monkeypatch) -> Memory:
    """`Memory` with a deterministic 4-dim embedder that hashes the
    input text into one of 4 buckets — same text always lands in the
    same vector quadrant, different texts collide deterministically.
    Good enough to exercise the index roundtrip without real MLX."""
    cfg = Config(
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )

    def _stub_embed(self, inputs):
        out = []
        for s in inputs:
            h = sum(ord(c) for c in s) % 4
            v = [0.0] * 4
            v[h] = 1.0
            out.append(v)
        return out

    monkeypatch.setattr("mem_lmx.embedder.MLXEmbedder.embed", _stub_embed)
    return Memory(cfg)


def test_save_writes_md_and_indexes(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="primer memo del test", title="Test 1", type_="note")
    abs_path = mem_with_stub.cfg.vault_path / rec.path
    assert abs_path.is_file()
    text = abs_path.read_text(encoding="utf-8")
    assert "title: Test 1" in text
    assert "primer memo del test" in text
    assert mem_with_stub.store.count() == 1


def test_save_rejects_invalid_type(mem_with_stub: Memory):
    with pytest.raises(ValueError, match="not in valid set"):
        mem_with_stub.save(content="x", type_="bogus")


def test_save_rejects_empty_content(mem_with_stub: Memory):
    with pytest.raises(ValueError, match="non-empty"):
        mem_with_stub.save(content="   ")


def test_search_returns_matching(mem_with_stub: Memory):
    mem_with_stub.save(content="alpha", title="A")
    mem_with_stub.save(content="beta", title="B")
    hits = mem_with_stub.search("alpha", limit=2)
    assert any(h.title == "A" for h in hits)


def test_get_returns_record_with_body(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="cuerpo del memo", title="X")
    fetched = mem_with_stub.get(rec.id)
    assert fetched is not None
    assert fetched.title == "X"
    assert "cuerpo del memo" in fetched.body


def test_get_missing_returns_none(mem_with_stub: Memory):
    assert mem_with_stub.get("nope") is None


def test_list_orders_recent_first(mem_with_stub: Memory):
    a = mem_with_stub.save(content="primero", title="A")
    # Force monotonic timestamp ordering for the test (sub-second
    # collisions would otherwise tie-break in insertion order).
    import time

    time.sleep(0.001)
    b = mem_with_stub.save(content="segundo", title="B")
    items = mem_with_stub.list(limit=10)
    titles = [r.title for r in items]
    # B was saved second → should appear first under `updated DESC`.
    assert titles.index("B") < titles.index("A")
    assert {r.id for r in items} == {a.id, b.id}


def test_delete_removes_disk_and_index(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="borrar este", title="X")
    assert (mem_with_stub.cfg.vault_path / rec.path).is_file()
    assert mem_with_stub.delete(rec.id) is True
    assert mem_with_stub.store.count() == 0
    assert not (mem_with_stub.cfg.vault_path / rec.path).is_file()


def test_delete_missing_returns_false(mem_with_stub: Memory):
    assert mem_with_stub.delete("nope") is False


def test_tags_lower_dedup(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="x", title="X", tags=["MLX", "mlx", "Local"])
    assert rec.tags == ["mlx", "local"]


def test_title_derived_from_first_line(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="# Encabezado\n\nbody")
    assert rec.title == "Encabezado"


def test_save_truncates_huge_body(tmp_cfg: Config, monkeypatch):
    cfg = Config(
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
        max_content_size=100,
    )
    monkeypatch.setattr(
        "mem_lmx.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    mem = Memory(cfg)
    huge = "x" * 10_000
    rec = mem.save(content=huge, title="huge")
    # Body on disk should be truncated to `max_content_size`.
    on_disk = (cfg.vault_path / rec.path).read_text()
    assert on_disk.count("x") <= 100
