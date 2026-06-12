"""embed-cache for memory records: save()/update() reuse the
content-addressed embedding cache instead of always re-embedding.

The cache is keyed on (embedder_model, dims, sha256(compose_text)); a model
swap changes the key, so a stale vector can never be served. This mirrors
the reindex path, which already used `_embed_cached`.
"""

from __future__ import annotations


def _count_embeds(mock_memory):
    """Wrap embedder.embed to count document-embedding forward passes."""
    state = {"n": 0}
    orig = mock_memory.embedder.embed

    def counting(inputs):
        state["n"] += 1
        return orig(inputs)

    mock_memory.embedder.embed = counting
    return state


def test_update_revert_hits_embedding_cache(mock_memory):
    rec = mock_memory.save(content="v1 body", title="T")
    mock_memory.update(rec.id, content="v2 body")  # warms cache for v2 + v1

    state = _count_embeds(mock_memory)
    mock_memory.update(rec.id, content="v1 body")  # compose text seen before
    assert state["n"] == 0, "reverting to previously-embedded content must hit cache"


def test_save_populates_embedding_cache(mock_memory):
    from memo.memory.maintain_ops import _sha256_short

    mock_memory.save(content="cache me please", title="Title")
    text = mock_memory._compose_for_embed("Title", "cache me please")
    cached = mock_memory.store.get_repo_embedding_cache(
        model=mock_memory.cfg.embedder_model,
        dims=mock_memory.cfg.embedder_dims,
        input_hashes=[_sha256_short(text)],
    )
    assert cached, "save should populate the content-addressed embedding cache"
