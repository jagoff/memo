"""High-level Memory write-path tests with stub embedder."""

from __future__ import annotations

import pytest

from memo.config import Config
from memo.memory import Memory


def test_save_writes_md_and_indexes(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="primer memo del test", title="Test 1", type_="note")
    abs_path = mem_with_stub.cfg.memory_dir / rec.path
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


def test_save_index_failure_keeps_md_and_marks_pending(mem_with_stub: Memory, monkeypatch):
    def _explode(self, inputs):
        raise RuntimeError("embedder down")

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _explode)
    rec = mem_with_stub.save(content="cuerpo recuperable", title="Recuperable")

    abs_path = mem_with_stub.cfg.memory_dir / rec.path
    assert abs_path.is_file()
    text = abs_path.read_text(encoding="utf-8")
    assert "_memo_embed_pending" in text
    assert rec.extra.get("_memo_embed_pending") is True
    assert mem_with_stub.store.get(rec.id) is not None


def test_tags_lower_dedup(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="x", title="X", tags=["MLX", "mlx", "Local"])
    assert rec.tags == ["mlx", "local"]


def test_title_derived_from_first_line(mem_with_stub: Memory):
    rec = mem_with_stub.save(content="# Encabezado\n\nbody")
    assert rec.title == "Encabezado"


def test_embed_batch_preserves_order_and_handles_empty(tmp_cfg: Config, monkeypatch):
    seen: list[int] = []

    def _spy(self, inputs):
        seen.append(len(inputs))
        out = []
        for s in inputs:
            if not s:
                out.append([0.0] * 4)
            else:
                v = [0.0] * 4
                v[len(s) % 4] = 1.0
                out.append(v)
        return out

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _spy)
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )
    mem = Memory(cfg)
    rec = mem.save(content="cuerpo", title="X")
    assert rec.title == "X"
    assert seen == [1]


def test_auto_derive_fills_missing_fields(mem_with_stub: Memory, monkeypatch):
    seen_messages: list[list[dict]] = []

    def _stub_chat(self, model, messages, options=None):
        seen_messages.append(messages)
        return {"message": {"content": '{"title": "Derived Title", "type": "decision", "tags": ["alpha", "beta", "gamma"]}'}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    rec = mem_with_stub.save(content="long body about something", auto_derive=True)
    assert rec.title == "Derived Title"
    assert rec.type == "decision"
    assert rec.tags == ["alpha", "beta", "gamma"]
    assert len(seen_messages) == 1
    assert seen_messages[0][0]["role"] == "system"
    assert "long body about something" in seen_messages[0][1]["content"]


def test_auto_derive_does_not_override_caller(mem_with_stub: Memory, monkeypatch):
    def _stub_chat(self, model, messages, options=None):
        return {"message": {"content": '{"title": "LLM Title", "type": "bug", "tags": ["llm"]}'}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    rec = mem_with_stub.save(
        content="x", title="Mine", type_="fact", tags=["mine"], auto_derive=True,
    )
    assert rec.title == "Mine"
    assert rec.type == "fact"
    assert rec.tags == ["mine"]


def test_auto_derive_tolerates_bad_llm_output(mem_with_stub: Memory, monkeypatch):
    def _stub_chat(self, model, messages, options=None):
        return {"message": {"content": "this is not json at all sorry"}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    rec = mem_with_stub.save(content="primer línea\n\nmás contenido", auto_derive=True)
    assert rec.title == "primer línea"
    assert rec.type == "note"
    assert rec.tags == []


def test_save_truncates_huge_body(tmp_cfg: Config, monkeypatch):
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
        max_content_chars=100,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    mem = Memory(cfg)
    huge = "x" * 10_000
    rec = mem.save(content=huge, title="huge")
    on_disk = (cfg.memory_dir / rec.path).read_text()
    assert on_disk.count("x") <= 110


def test_save_rejects_wrong_dim_embedding(tmp_cfg: Config, monkeypatch):
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0] * 7 for _ in inputs],
    )
    mem = Memory(cfg)
    with pytest.raises(ValueError, match="dim mismatch"):
        mem.save(content="x", title="t")


def test_high_signal_detector_rescues_pin_notes():
    from memo.cli_ingest import _is_high_signal

    real_case = "# Link de pago escuela Grecia\n\nhttps://sit.educacionadventista.org.ar/"
    assert _is_high_signal(real_case, ["grecia", "escuela", "pagos", "links"])
    assert _is_high_signal("https://example.com", None)
    assert _is_high_signal("```bash\nls\n```", None)
    assert _is_high_signal("CBU 0001234567890", ["dato"])
    assert not _is_high_signal(
        "#hipotesis #pendiente\n¿qué iba a hacer mañana?",
        ["hipotesis", "pendiente"],
    )
    assert not _is_high_signal("algo corto sin nada especial", ["random"])


def test_save_rejects_zero_norm_embedding(tmp_cfg: Config, monkeypatch):
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[0.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    mem = Memory(cfg)
    with pytest.raises(ValueError, match="norm out of"):
        mem.save(content="x", title="t")
