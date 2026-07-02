"""CLI: `memo embed` — single (asymmetric query) + batch (symmetric)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli import cli


def _patch_embedder(monkeypatch: pytest.MonkeyPatch, dims: int = 4) -> dict[str, list[str]]:
    """Stub MLXEmbedder. Returns a dict tracking which methods were called."""
    calls: dict[str, list[str]] = {"embed": [], "embed_query": []}

    def _embed(self, inputs):
        calls["embed"].extend(list(inputs))
        out = []
        for s in inputs:
            v = [0.0] * dims
            v[sum(ord(c) for c in s) % dims] = 1.0
            out.append(v)
        return out

    def _embed_query(self, query):
        calls["embed_query"].append(query)
        v = [0.0] * dims
        v[sum(ord(c) for c in query) % dims] = 1.0
        return v

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _embed)
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed_query", _embed_query)
    return calls


def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point memo at an isolated tmp data dir + state dir."""
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_path / "vault"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    (tmp_path / "data").mkdir()
    (tmp_path / "vault" / "Obsidian" / "AI" / "memory").mkdir(parents=True)
    (tmp_path / "state").mkdir()


def test_memo_embed_single_returns_vector_with_dim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _env(monkeypatch, tmp_path)
    calls = _patch_embedder(monkeypatch, dims=4)

    runner = CliRunner()
    result = runner.invoke(cli, ["embed", "hablame de Grecia"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output.strip())
    assert "vector" in payload
    assert payload["dim"] == 4
    assert len(payload["vector"]) == 4
    assert "model" in payload
    # Asymmetric path used (query prefix), NOT symmetric batch embed.
    assert calls["embed_query"] == ["hablame de Grecia"]
    assert calls["embed"] == []


def test_memo_embed_batch_via_stdin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _env(monkeypatch, tmp_path)
    calls = _patch_embedder(monkeypatch, dims=4)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["embed", "--batch-json", "-"],
        input='["alpha", "beta", "gamma"]',
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output.strip())
    assert "vectors" in payload
    assert len(payload["vectors"]) == 3
    assert payload["dim"] == 4
    # Symmetric path: embed() called with the full batch in one shot.
    assert calls["embed"] == ["alpha", "beta", "gamma"]
    assert calls["embed_query"] == []


def test_memo_embed_query_uses_asymmetric_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Single TEXT always goes through `embed_query` (query prefix).
    Critical invariant — mixing the symmetric path here would degrade
    cosine similarity vs documents indexed at retrieval time."""
    _env(monkeypatch, tmp_path)
    calls = _patch_embedder(monkeypatch, dims=4)

    runner = CliRunner()
    result = runner.invoke(cli, ["embed", "anything"])
    assert result.exit_code == 0
    assert calls["embed_query"] == ["anything"]
    assert calls["embed"] == []


def test_memo_embed_no_text_and_no_batch_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _env(monkeypatch, tmp_path)
    _patch_embedder(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(cli, ["embed"])
    assert result.exit_code != 0
    assert "provide TEXT or --batch-json" in result.output


def test_memo_embed_text_and_batch_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _env(monkeypatch, tmp_path)
    _patch_embedder(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["embed", "hello", "--batch-json", "-"],
        input='["a"]',
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_memo_embed_batch_invalid_json_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _env(monkeypatch, tmp_path)
    _patch_embedder(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(cli, ["embed", "--batch-json", "-"], input="not json")
    assert result.exit_code != 0
    assert "invalid JSON" in result.output


def test_memo_embed_batch_non_string_list_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _env(monkeypatch, tmp_path)
    _patch_embedder(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(cli, ["embed", "--batch-json", "-"], input="[1, 2, 3]")
    assert result.exit_code != 0
    assert "expected JSON list of strings" in result.output
