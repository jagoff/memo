"""Save-side capture (Phase B) — pipeline tests.

Stubs the helper LLM so tests run on any platform. The real-MLX path
is exercised end-to-end via `tests/test_smoke_mlx.py`-style markers
when needed; here we focus on parsing, prefilter, dedup, and
idempotence — the parts most likely to break under refactor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memo.capture import (
    _extract_text,
    _hash_assistant,
    _passes_prefilter,
    _read_last_exchange,
    extract_insights,
    is_near_duplicate,
    run_capture,
)
from memo.config import Config
from memo.memory import Memory


def _write_transcript(p: Path, lines: list[dict]) -> None:
    p.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")


def test_extract_text_handles_string_content():
    assert _extract_text("hello world") == "hello world"


def test_extract_text_concatenates_text_blocks_skips_tools():
    blocks = [
        {"type": "text", "text": "Decidí migrar a MLX."},
        {"type": "tool_use", "name": "Read", "input": {}},
        {"type": "text", "text": "Razón: latency 30% menor."},
        {"type": "tool_result", "content": "..."},
    ]
    out = _extract_text(blocks)
    assert "Decidí migrar a MLX." in out
    assert "latency 30% menor" in out
    assert "tool_use" not in out
    assert "Read" not in out


def test_read_last_exchange_picks_latest_pair(tmp_path: Path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, [
        {"type": "user", "message": {"content": "old user msg"}},
        {"type": "assistant", "message": {"content": "old assistant msg"}},
        {"type": "user", "message": {
            "content": [{"type": "text", "text": "qué decidí sobre MLX?"}]
        }},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Decidiste usar MLX porque la latencia bajó 30%."},
            {"type": "tool_use", "name": "Read", "input": {}},
        ]}},
    ])
    pair = _read_last_exchange(transcript)
    assert pair is not None
    user, asst = pair
    assert "MLX" in user
    assert "latencia bajó 30%" in asst
    assert "tool_use" not in asst  # tool blocks stripped


def test_read_last_exchange_returns_none_on_user_only(tmp_path: Path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, [
        {"type": "user", "message": {"content": "lonely user msg"}},
    ])
    assert _read_last_exchange(transcript) is None


def test_prefilter_skips_too_short():
    assert not _passes_prefilter("decidí")  # < 200 chars


def test_prefilter_skips_no_triggers():
    text = "x" * 500
    assert not _passes_prefilter(text)


def test_prefilter_passes_with_trigger_and_length():
    text = (
        "Hoy estuvimos analizando el flujo de retrieval del sistema. "
        "Decidí cambiar el threshold del reranker a 0.4 porque con 0.6 "
        "se filtraban hits relevantes en queries diffuse. La razón es "
        "que los scores fusionados tienen distribución diferente al "
        "cosine puro y necesitan un piso más permisivo."
    )
    assert _passes_prefilter(text)


def test_extract_insights_parses_clean_json():
    """Stub helper that returns valid JSON. The pipeline should
    surface every well-formed item and reject malformed entries."""
    class _StubChat:
        def chat(self, model, messages, options):
            return {"message": {"content": json.dumps([
                {
                    "title": "Use MLX over Ollama",
                    "type": "decision",
                    "body": "MLX shaved 30% off prefill latency vs Ollama for the same model.",
                    "tags": ["mlx", "ollama", "latency"],
                },
                {"title": "", "body": "missing title — should drop"},
                {
                    "title": "Reranker threshold 0.4 for hybrid mode",
                    "type": "preference",
                    "body": "Fused scores cluster around 0.3-0.7; 0.4 is the empirical knee.",
                    "tags": ["reranker", "threshold"],
                },
            ])}}

    out = extract_insights(_StubChat(), "any-model", "user", "assistant")
    assert len(out) == 2
    assert out[0]["title"] == "Use MLX over Ollama"
    assert out[1]["type"] == "preference"


def test_extract_insights_strips_markdown_fences():
    class _StubChat:
        def chat(self, model, messages, options):
            return {"message": {"content": (
                "```json\n"
                '[{"title": "T", "type": "note", "body": "B", "tags": []}]\n'
                "```"
            )}}

    out = extract_insights(_StubChat(), "m", "u", "a")
    assert len(out) == 1


def test_extract_insights_returns_empty_on_garbage():
    class _StubChat:
        def chat(self, model, messages, options):
            return {"message": {"content": "I cannot extract anything from this."}}

    out = extract_insights(_StubChat(), "m", "u", "a")
    assert out == []


def test_extract_insights_returns_empty_on_llm_exception():
    class _StubChat:
        def chat(self, model, messages, options):
            raise RuntimeError("MLX hiccup")

    out = extract_insights(_StubChat(), "m", "u", "a")
    assert out == []


@pytest.fixture
def mem_with_stub(tmp_cfg: Config, monkeypatch) -> Memory:
    """Real Memory with a stub embedder. Same shape as the test_memory
    fixture but exposed to capture tests."""
    # Use 64-dim buckets here (vs 4 in test_memory) so unrelated test
    # strings rarely collide. Dedup tests need distinct embeddings to
    # produce distinct vectors; the 4-bucket stub gives a 25% collision
    # probability, which fails ~1-in-4 runs.
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=64,
    )

    def _stub_embed(self, inputs):
        out = []
        for s in inputs:
            h = sum(ord(c) for c in s) % 64
            v = [0.0] * 64
            v[h] = 1.0
            out.append(v)
        return out

    def _stub_embed_query(self, query: str):
        # Match the bucket of `embed([f"{title}\\n\\n{body}"])` so dedup
        # tests can assert exact-match candidates collide. Real embedder
        # adds an instruction prefix; we skip it here so the stub stays
        # symmetric with the indexing path.
        h = sum(ord(c) for c in query) % 64
        v = [0.0] * 64
        v[h] = 1.0
        return v

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed_query", _stub_embed_query)
    return Memory(cfg)


def test_is_near_duplicate_flags_existing(mem_with_stub):
    mem_with_stub.save(content="a body that exists", title="alpha")
    candidate = {
        "title": "alpha",
        "body": "a body that exists",
        "tags": [],
        "type": "note",
    }
    # The 4-dim hash-stub maps identical strings to identical buckets,
    # so cosine = 1.0 and the dedup must catch it.
    assert is_near_duplicate(mem_with_stub, candidate, threshold=0.9)


def test_is_near_duplicate_lets_distinct_through(mem_with_stub):
    mem_with_stub.save(content="completely separate content here", title="alpha")
    candidate = {
        "title": "very different topic",
        "body": "z" * 50,
        "tags": [],
        "type": "note",
    }
    # Different bucket → cosine < threshold.
    assert not is_near_duplicate(mem_with_stub, candidate, threshold=0.9)


def test_hash_assistant_idempotent():
    a = "hola mundo"
    assert _hash_assistant(a) == _hash_assistant(a)
    assert _hash_assistant("a") != _hash_assistant("b")


def test_run_capture_skips_duplicate_turn(tmp_path: Path, monkeypatch):
    """If we re-fire on the same assistant message hash, nothing
    happens. Real-world: two Stop hooks racing or a transient
    Claude Code re-invoke must not produce duplicate memorias."""
    state_dir = tmp_path / "state"
    vault = tmp_path / "vault"
    (vault / "Obsidian" / "AI" / "memory").mkdir(parents=True)
    state_dir.mkdir()
    monkeypatch.setenv("MEMO_VAULT_PATH", str(vault))
    monkeypatch.setenv("MEMO_STATE_DIR", str(state_dir))
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed_query",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )

    transcript = tmp_path / "t.jsonl"
    _write_transcript(transcript, [
        {"type": "user", "message": {"content": "decidí algo"}},
        {"type": "assistant", "message": {"content": (
            "Decidiste cambiar la config porque el bug del reranker se "
            "manifestaba cuando el body excedía 4096 tokens. Fix: truncar "
            "a 1200 chars antes del rerank. Latencia bajó 3x."
        )}},
    ])

    run_capture(transcript)  # first call processes the turn
    out2 = run_capture(transcript)
    assert out2["status"] == "duplicate_turn"
