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


def test_extract_text_keeps_prose_clean_and_appends_tool_activity():
    blocks = [
        {"type": "text", "text": "Decidí migrar a MLX."},
        {"type": "tool_use", "name": "Edit", "input": {"file_path": "src/memo/embedder.py"}},
        {"type": "text", "text": "Razón: latency 30% menor."},
        {"type": "tool_result", "content": "..."},
    ]
    out = _extract_text(blocks)
    # Prose is concatenated and never polluted by raw tool block structure.
    assert "Decidí migrar a MLX." in out
    assert "latency 30% menor" in out
    assert "tool_use" not in out
    # Default-on: a compact TOOL ACTIVITY projection grounds the extraction.
    assert "TOOL ACTIVITY:" in out
    assert "Edit(src/memo/embedder.py)" in out


def test_extract_text_tool_evidence_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MEMO_CAPTURE_TOOL_EVIDENCE", "0")
    blocks = [
        {"type": "text", "text": "Decidí migrar a MLX."},
        {"type": "tool_use", "name": "Read", "input": {}},
    ]
    out = _extract_text(blocks)
    assert out == "Decidí migrar a MLX."
    assert "TOOL ACTIVITY" not in out


def test_read_last_exchange_picks_latest_pair(tmp_path: Path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {"type": "user", "message": {"content": "old user msg"}},
            {"type": "assistant", "message": {"content": "old assistant msg"}},
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "qué decidí sobre MLX?"}]},
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Decidiste usar MLX porque la latencia bajó 30%."},
                        {"type": "tool_use", "name": "Read", "input": {}},
                    ]
                },
            },
        ],
    )
    pair = _read_last_exchange(transcript)
    assert pair is not None
    user, asst = pair
    assert "MLX" in user
    assert "latencia bajó 30%" in asst
    assert "tool_use" not in asst  # tool blocks stripped


def test_read_last_exchange_returns_none_on_user_only(tmp_path: Path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {"type": "user", "message": {"content": "lonely user msg"}},
        ],
    )
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
            return {
                "message": {
                    "content": json.dumps(
                        [
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
                        ]
                    )
                }
            }

    out = extract_insights(_StubChat(), "any-model", "user", "assistant")
    assert len(out) == 2
    assert out[0]["title"] == "Use MLX over Ollama"
    assert out[1]["type"] == "preference"


def test_extract_insights_strips_markdown_fences():
    class _StubChat:
        def chat(self, model, messages, options):
            return {
                "message": {
                    "content": (
                        '```json\n[{"title": "T", "type": "note", "body": "B", "tags": []}]\n```'
                    )
                }
            }

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
    mem = Memory(cfg)
    yield mem
    mem.close()


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


def test_extract_and_save_drops_near_identical_paraphrase(mem_with_stub, monkeypatch):
    import memo.capture as capture_mod

    cand = {"title": "t", "body": "b" * 80, "type": "note", "tags": []}
    monkeypatch.setattr(capture_mod, "extract_insights", lambda *a, **kw: [cand])
    monkeypatch.setattr(capture_mod, "_passes_quality", lambda *a, **kw: True)
    # >= drop threshold (0.97) → a paraphrase with no new info → drop.
    monkeypatch.setattr(
        capture_mod,
        "find_near_duplicate",
        lambda *a, **kw: {"id": "x" * 32, "score": 0.99, "title": "t"},
    )
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert out["skipped_dup"] == 1
    assert out["reconciled"] == 0
    assert out["saved"] == []


def test_extract_and_save_admits_same_topic_evolution(mem_with_stub, monkeypatch):
    import memo.capture as capture_mod

    cand = {"title": "t", "body": "b" * 80, "type": "decision", "tags": []}
    monkeypatch.setattr(capture_mod, "extract_insights", lambda *a, **kw: [cand])
    monkeypatch.setattr(capture_mod, "_passes_quality", lambda *a, **kw: True)
    # In the 0.85–0.97 band → same topic evolving → ADMIT as new so the
    # supersede pass can run (the old behaviour silently dropped it).
    monkeypatch.setattr(
        capture_mod,
        "find_near_duplicate",
        lambda *a, **kw: {"id": "x" * 32, "score": 0.90, "title": "t"},
    )
    out = capture_mod._extract_and_save(mem_with_stub, mem_with_stub.cfg, "u", "a")
    assert out["reconciled"] == 1
    assert out["skipped_dup"] == 0
    assert len(out["saved"]) == 1


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
    _write_transcript(
        transcript,
        [
            {"type": "user", "message": {"content": "decidí algo"}},
            {
                "type": "assistant",
                "message": {
                    "content": (
                        "Decidiste cambiar la config porque el bug del reranker se "
                        "manifestaba cuando el body excedía 4096 tokens. Fix: truncar "
                        "a 1200 chars antes del rerank. Latencia bajó 3x."
                    )
                },
            },
        ],
    )

    run_capture(transcript)  # first call processes the turn
    out2 = run_capture(transcript)
    assert out2["status"] == "duplicate_turn"


def test_write_capture_notification_lists_titles(tmp_path: Path) -> None:
    from memo.cli_capture import _write_capture_notification

    _write_capture_notification(tmp_path, ["Falso negativo en grounding", "Floor de tokens"])
    notif = (tmp_path / "pending_idle_notification.txt").read_text(encoding="utf-8")
    assert notif == "※ MEMO auto-saved\n"  # simplified to single muted line


def test_write_capture_notification_idle_tag(tmp_path: Path) -> None:
    from memo.cli_capture import _write_capture_notification

    _write_capture_notification(tmp_path, ["Insight"], idle=True)
    notif = (tmp_path / "pending_idle_notification.txt").read_text(encoding="utf-8")
    # idle flag is accepted but notification format is the same
    assert notif == "※ MEMO auto-saved\n"


def test_write_capture_notification_truncates_and_counts(tmp_path: Path) -> None:
    from memo.cli_capture import _write_capture_notification

    _write_capture_notification(tmp_path, [f"t{i}" for i in range(5)])
    notif = (tmp_path / "pending_idle_notification.txt").read_text(encoding="utf-8")
    # simplified notification format doesn't include individual title details
    assert notif == "※ MEMO auto-saved\n"


def test_write_capture_notification_noop_on_empty(tmp_path: Path) -> None:
    from memo.cli_capture import _write_capture_notification

    _write_capture_notification(tmp_path, [])
    assert not (tmp_path / "pending_idle_notification.txt").exists()


def test_capture_stop_writes_notification_when_saved(tmp_path: Path, monkeypatch) -> None:
    """capture-stop outputs a notification when memories are saved."""
    from click.testing import CliRunner

    import memo.capture as capture_mod
    from memo.cli_capture import capture_stop

    state = tmp_path / "state"
    state.mkdir()
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setattr(
        capture_mod,
        "run_capture",
        lambda *a, **k: {"status": "ok", "saved": ["id1"], "saved_titles": ["Insight uno"]},
    )

    payload = json.dumps({"transcript_path": str(transcript), "session_id": "s1"})
    result = CliRunner().invoke(capture_stop, input=payload)

    assert result.exit_code == 0
    # capture_stop prints to console when memories are saved
    assert "auto-saved" in result.output


def test_capture_stop_no_notification_when_nothing_saved(tmp_path: Path, monkeypatch) -> None:
    from click.testing import CliRunner

    import memo.capture as capture_mod
    from memo.cli_capture import capture_stop

    state = tmp_path / "state"
    state.mkdir()
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setattr(
        capture_mod,
        "run_capture",
        lambda *a, **k: {"status": "duplicate_turn", "saved": [], "saved_titles": []},
    )

    payload = json.dumps({"transcript_path": str(transcript), "session_id": "s1"})
    result = CliRunner().invoke(capture_stop, input=payload)

    assert result.exit_code == 0
    assert not (state / "pending_idle_notification.txt").exists()


def test_capture_stop_recovers_transcript_path_from_session_id(tmp_path: Path, monkeypatch) -> None:
    """Regression: 2026-06-27 onward some Stop-hook payloads omit
    transcript_path outright, so capture-stop used to no-op — including its
    grounding.score_turn call, the source of `memo tokens`' data. When
    session_id is present, it must recover transcript_path before bailing."""
    from click.testing import CliRunner

    import memo.capture as capture_mod
    import memo.session as session_mod
    from memo.cli_capture import capture_stop

    state = tmp_path / "state"
    state.mkdir()
    transcript = tmp_path / "recovered.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")

    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setattr(session_mod, "find_transcript_path", lambda sid: str(transcript))

    captured_paths: list[Path] = []
    monkeypatch.setattr(
        capture_mod,
        "run_capture",
        lambda path, **k: (
            captured_paths.append(path),
            {"status": "ok", "saved": [], "saved_titles": []},
        )[1],
    )

    payload = json.dumps({"session_id": "s-missing-transcript"})
    result = CliRunner().invoke(capture_stop, input=payload)

    assert result.exit_code == 0
    assert captured_paths == [transcript]


def test_capture_stop_still_noops_when_session_id_also_missing(tmp_path: Path, monkeypatch) -> None:
    """No session_id at all → nothing to recover by, must stay a silent no-op."""
    from click.testing import CliRunner

    import memo.capture as capture_mod
    from memo.cli_capture import capture_stop

    state = tmp_path / "state"
    state.mkdir()

    monkeypatch.setenv("MEMO_STATE_DIR", str(state))
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")

    called = []
    monkeypatch.setattr(capture_mod, "run_capture", lambda *a, **k: called.append(1))

    result = CliRunner().invoke(capture_stop, input=json.dumps({}))

    assert result.exit_code == 0
    assert not called


# ── <private> span stripping (A2) ─────────────────────────────────────────────


def test_extract_text_strips_private_spans(monkeypatch):
    monkeypatch.delenv("MEMO_PRIVATE_MARKERS", raising=False)
    from memo.capture import _extract_text

    out = _extract_text("public fact <private>my api key is hunter2</private> tail")
    assert "hunter2" not in out
    assert "public fact" in out and "tail" in out


def test_extract_text_private_flag_off_keeps_text(monkeypatch):
    monkeypatch.setenv("MEMO_PRIVATE_MARKERS", "0")
    from memo.capture import _extract_text

    out = _extract_text("keep <private>visible when disabled</private>")
    assert "visible when disabled" in out


def test_extract_text_strips_private_in_block_content(monkeypatch):
    monkeypatch.delenv("MEMO_PRIVATE_MARKERS", raising=False)
    from memo.capture import _extract_text

    content = [{"type": "text", "text": "decision: use X <private>token abc</private>"}]
    assert "token abc" not in _extract_text(content)


def test_miner_iter_exchanges_honors_private_spans(tmp_path: Path, monkeypatch):
    """mine-history reads transcripts through the same _extract_text, so
    <private> content must never reach the extractor from history either."""
    monkeypatch.delenv("MEMO_PRIVATE_MARKERS", raising=False)
    from memo.transcript_miner import iter_exchanges

    t = tmp_path / "s.jsonl"
    lines = [
        {"type": "user", "message": {"content": "set up deploy key"}},
        {
            "type": "assistant",
            "message": {
                "content": "done. <private>the key is ghp_secret</private> committed the config"
            },
        },
    ]
    t.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    pairs = list(iter_exchanges(t))
    assert len(pairs) == 1
    assert "ghp_secret" not in pairs[0][1]
    assert "committed the config" in pairs[0][1]


# ── secret masking in capture (A1) ────────────────────────────────────────────


class _SecretChat:
    """Helper-LLM stub whose extraction output carries a GitHub token."""

    def chat(self, model, messages, options=None):
        tok = "ghp_" + "a" * 32 + "WXYZ"
        return {
            "message": {
                "content": json.dumps(
                    [
                        {
                            "title": "deploy token configured",
                            "type": "fact",
                            "body": f"the deploy uses {tok} against origin",
                            "tags": ["deploy"],
                        }
                    ]
                )
            }
        }


def test_extract_insights_masks_secrets_and_tags_redacted(monkeypatch):
    monkeypatch.delenv("MEMO_REDACT_SECRETS", raising=False)
    out = extract_insights(_SecretChat(), "m", "u", "a")
    assert len(out) == 1
    assert "ghp_" not in out[0]["body"]
    assert "****WXYZ" in out[0]["body"]
    assert "_redacted" in out[0]["tags"]


def test_extract_insights_redaction_flag_off_keeps_raw(monkeypatch):
    monkeypatch.setenv("MEMO_REDACT_SECRETS", "0")
    out = extract_insights(_SecretChat(), "m", "u", "a")
    assert "ghp_" in out[0]["body"]
    assert "_redacted" not in out[0]["tags"]


def test_extract_and_save_text_verbatim_fallback_redacts(mem_with_stub, monkeypatch):
    """The verbatim fallback (extractor yields zero candidates) is the one
    save that bypasses extract_insights — it must mask too."""
    import memo.capture as capture_mod

    monkeypatch.delenv("MEMO_REDACT_SECRETS", raising=False)
    monkeypatch.setattr(capture_mod, "extract_insights", lambda *a, **kw: [])
    tok = "sk-ant-" + "k" * 24 + "1234"
    out = capture_mod.extract_and_save_text(
        mem_with_stub,
        mem_with_stub.cfg,
        f"api key rotated to {tok} today",
        title="rotation",
    )
    assert out["status"] == "verbatim"
    rec = mem_with_stub.get(out["saved"][0])
    assert tok not in rec.body
    assert "****1234" in rec.body
    assert "_redacted" in rec.tags
