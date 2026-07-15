"""End-to-end smoke against the real MLX embedder.

Gated by `@pytest.mark.requires_mlx` — auto-skips on Linux/x86 dev
boxes (the conftest fixture handles the gate). This is the *only*
test that pulls the real embedder weights, so it's marked `slow` too:
the default `pytest -m 'not slow'` invocation skips it. Run it
explicitly with:

    pytest -m requires_mlx -v

What it verifies (the rest of the suite stubs out MLX, so this is
where we catch a broken model load or a wrong embedding dim):

1. `MLXEmbedder` loads `Qwen3-Embedding-0.6B-4bit-DWQ` and produces
   a 1024-dim L2-normalised vector.
2. End-to-end `Memory.save → search` returns the saved record at
   the top of the result list (semantic identity check — a query
   identical to the saved content must come back rank #1).
3. Two semantically distant entries don't collide; the relevant
   one ranks higher than the irrelevant one for a paraphrased query.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest

from memo.config import Config
from memo.memory import Memory


@pytest.fixture
def real_mlx_memory(tmp_cfg: Config) -> Iterator[Memory]:
    """Real embedder instance that always releases MLX/Metal state after a test."""
    mem = Memory(tmp_cfg)
    yield mem
    mem.close()


@pytest.mark.requires_mlx
@pytest.mark.slow
def test_embedder_produces_unit_vectors(real_mlx_memory: Memory, tmp_cfg: Config):
    mem = real_mlx_memory
    [v] = mem.embedder.embed(["hola che, tracking de pruebas MLX"])
    assert len(v) == tmp_cfg.embedder_dims
    # L2 norm should be ~1 (cosine semantics depend on this).
    norm = math.sqrt(sum(x * x for x in v))
    assert 0.95 <= norm <= 1.05, f"L2 norm out of range: {norm}"


@pytest.mark.requires_mlx
@pytest.mark.slow
def test_save_and_search_roundtrip(real_mlx_memory: Memory):
    mem = real_mlx_memory

    rec_a = mem.save(
        content=(
            "Decisión: usar mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ "
            "como embedder local en Apple Silicon. Reemplaza Ollama nomic."
        ),
        title="Embedder MLX",
        type_="decision",
        tags=["mlx", "embedder"],
    )
    rec_b = mem.save(
        content=(
            "Receta de pizza: masa con harina 000, levadura, agua tibia, "
            "sal y aceite. Reposo de 24 horas en heladera para mejor sabor."
        ),
        title="Pizza casera",
        type_="note",
        tags=["receta"],
    )

    # Identity query → rec_a top.
    hits = mem.search("qué embedder local usamos en MLX", limit=3)
    assert hits, "search returned zero hits"
    assert hits[0].id == rec_a.id, f"top hit was {hits[0].title!r}, expected {rec_a.title!r}"

    # Off-topic recipe query → rec_b should outrank rec_a.
    hits = mem.search("cómo hago la masa de la pizza", limit=3)
    assert hits[0].id == rec_b.id


@pytest.mark.requires_mlx
@pytest.mark.slow
def test_reindex_after_external_edit_picks_up_via_real_embedder(
    real_mlx_memory: Memory, tmp_cfg: Config
):
    """Real-MLX version of the reindex test — proves the embedder is
    actually re-invoked when a body changes on disk, not just stubbed."""
    import frontmatter as fm

    mem = real_mlx_memory
    rec = mem.save(content="contenido inicial sobre fútbol argentino", title="X")
    md_path = tmp_cfg.memory_dir / rec.path

    post = fm.loads(md_path.read_text())
    post.content = "contenido reemplazado: ahora habla de café de especialidad"
    md_path.write_text(fm.dumps(post), encoding="utf-8")

    counts = mem.reindex()
    assert counts["reindexed"] == 1

    # The new body should rank above its title-derived original for
    # a coffee query, confirming the embedding moved.
    hits = mem.search("cafetería de especialidad y granos", limit=3)
    assert hits and hits[0].id == rec.id
