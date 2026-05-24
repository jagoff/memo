from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

from click.testing import CliRunner

from memo.cli import cli
from memo.config import Config
from memo.memory import Memory
from memo.repo_index import _chunk_lines


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _make_repo(root: Path) -> Path:
    repo = root / "sample-repo"
    (repo / "src").mkdir(parents=True)
    (repo / "node_modules").mkdir()
    (repo / "src" / "app.py").write_text(
        "def alpha():\n"
        "    return 'needle-value'\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text(
        "# Sample\n\nThis repo documents a searchable zebra token.\n",
        encoding="utf-8",
    )
    (repo / "node_modules" / "skip.js").write_text("needle-value\n", encoding="utf-8")
    (repo / "image.bin").write_bytes(b"\x00\x01\x02")

    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, text=True)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


def _make_text_repo(root: Path, name: str, files: dict[str, str]) -> Path:
    repo = root / name
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, text=True)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


def _cfg(tmp_path: Path) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        embedder_dims=4,
        reranker_enabled=False,
    )


def _patch_embedder(monkeypatch) -> None:
    def _embed(self, inputs):
        out = []
        for text in inputs:
            if "zebra" in text:
                out.append([0.0, 1.0, 0.0, 0.0])
            else:
                out.append([1.0, 0.0, 0.0, 0.0])
        return out

    def _embed_query(self, text):
        return [0.0, 1.0, 0.0, 0.0] if "zebra" in text else [1.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _embed)
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed_query", _embed_query)


def test_repo_index_indexes_lines_chunks_and_gets_ranges(tmp_path: Path, monkeypatch):
    _patch_embedder(monkeypatch)
    repo = _make_repo(tmp_path)
    mem = Memory(_cfg(tmp_path))

    out = mem.repo_index(str(repo), name="sample")

    assert out["name"] == "sample"
    assert out["indexed_files"] == 2
    assert out["indexed_lines"] == 5
    assert out["skipped_binary"] == 1
    assert out["skipped_excluded"] >= 1
    assert out["indexed_chunks"] == 2
    assert out["embedded_chunks"] == 2
    assert out["pending_chunks"] == 0
    assert out["semantic_status"] == "semantic_ready"

    line_hits = mem.repo_search("needle-value", repo="sample", mode="line")
    assert line_hits
    assert line_hits[0].path == "src/app.py"
    assert line_hits[0].line_start == 2

    vec_hits = mem.repo_search("zebra", repo="sample", mode="vec")
    assert vec_hits
    assert vec_hits[0].path == "README.md"

    body = mem.repo_get_file("sample", "src/app.py", start=1, end=2)
    assert body is not None
    assert "def alpha" in body["text"]
    assert "needle-value" in body["text"]

    unchanged = mem.repo_index(str(repo), name="sample")
    assert unchanged["skipped_repo_unchanged"] is True
    assert unchanged["unchanged_files"] == 2
    assert unchanged["semantic_status"] == "semantic_ready"


def test_repo_chunker_splits_single_huge_line():
    chunks = _chunk_lines(["x" * 10_500], target_chars=3500)

    assert len(chunks) == 3
    assert all(line_start == 1 and line_end == 1 for _, line_start, line_end, _ in chunks)
    assert all(len(body) <= 3500 for _, _, _, body in chunks)
    assert "".join(body for _, _, _, body in chunks) == "x" * 10_500


def test_repo_index_never_embeds_single_huge_line(tmp_path: Path, monkeypatch):
    repo = tmp_path / "huge-line-repo"
    repo.mkdir()
    (repo / "minified.js").write_text("const payload = '" + ("x" * 12_000) + "';\n", encoding="utf-8")
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, text=True)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")

    max_seen = 0

    def _embed(self, inputs):
        nonlocal max_seen
        max_seen = max(max_seen, *(len(text) for text in inputs))
        return [[1.0, 0.0, 0.0, 0.0] for _ in inputs]

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _embed)
    mem = Memory(_cfg(tmp_path))

    out = mem.repo_index(str(repo), name="huge")

    assert out["indexed_files"] == 1
    assert out["indexed_lines"] == 1
    assert out["indexed_chunks"] >= 4
    # Includes the repo/path/line header plus a bounded chunk body.
    assert max_seen < 3800


def test_repo_index_can_skip_embeddings_then_embed_later(tmp_path: Path, monkeypatch):
    repo = _make_repo(tmp_path)
    mem = Memory(_cfg(tmp_path))

    def _unexpected_embed(self, inputs):
        raise AssertionError("embedder should not run when with_embeddings=False")

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _unexpected_embed)
    out = mem.repo_index(str(repo), name="sample", with_embeddings=False)

    assert out["indexed_chunks"] == 2
    assert out["embedded_chunks"] == 0
    assert out["pending_chunks"] == 2
    assert out["semantic_status"] == "semantic_pending"
    assert mem.repo_search("zebra", repo="sample", mode="vec") == []

    _patch_embedder(monkeypatch)
    embedded = mem.repo_embed("sample")

    assert embedded["embedded_chunks"] == 2
    assert embedded["pending_chunks"] == 0
    assert embedded["semantic_status"] == "semantic_ready"
    status = mem.repo_status("sample")
    assert status is not None
    assert status["semantic_status"] == "semantic_ready"
    assert mem.repo_search("zebra", repo="sample", mode="vec")[0].path == "README.md"


def test_repo_embed_reuses_cache_on_force(tmp_path: Path, monkeypatch):
    repo = _make_repo(tmp_path)
    mem = Memory(_cfg(tmp_path))
    calls = 0

    def _embed(self, inputs):
        nonlocal calls
        calls += len(inputs)
        return [[1.0, 0.0, 0.0, 0.0] for _ in inputs]

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _embed)
    mem.repo_index(str(repo), name="sample", with_embeddings=False)

    first = mem.repo_embed("sample")

    assert first["model_chunks"] == 2
    assert first["cached_chunks"] == 0
    assert calls == 2

    def _unexpected_embed(self, inputs):
        raise AssertionError("force re-embed should be satisfied from cache")

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _unexpected_embed)
    second = mem.repo_embed("sample", force=True)

    assert second["embedded_chunks"] == 2
    assert second["model_chunks"] == 0
    assert second["cached_chunks"] == 2
    assert second["semantic_status"] == "semantic_ready"


def test_repo_embed_sorts_batches_by_input_length(tmp_path: Path, monkeypatch):
    repo = _make_text_repo(
        tmp_path,
        "size-repo",
        {
            "long.txt": "l" * 1000,
            "short.txt": "s\n",
            "medium.txt": "m" * 200,
        },
    )
    mem = Memory(_cfg(tmp_path))
    seen_lengths: list[int] = []
    monkeypatch.setenv("MEMO_REPO_EMBED_BATCH", "10")

    def _embed(self, inputs):
        seen_lengths.extend(len(text) for text in inputs)
        return [[1.0, 0.0, 0.0, 0.0] for _ in inputs]

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _embed)

    mem.repo_index(str(repo), name="sizes")

    assert len(seen_lengths) == 3
    assert seen_lengths == sorted(seen_lengths)


def test_repo_embed_reduces_batch_size_after_runtime_error(tmp_path: Path, monkeypatch):
    repo = _make_text_repo(
        tmp_path,
        "fallback-repo",
        {
            "a.txt": "alpha\n",
            "b.txt": "bravo\n",
            "c.txt": "charlie\n",
        },
    )
    mem = Memory(_cfg(tmp_path))
    calls: list[int] = []
    monkeypatch.setenv("MEMO_REPO_EMBED_BATCH", "4")

    def _embed(self, inputs):
        calls.append(len(inputs))
        if len(inputs) > 1:
            raise RuntimeError("simulated MLX out of memory")
        return [[1.0, 0.0, 0.0, 0.0] for _ in inputs]

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _embed)

    out = mem.repo_index(str(repo), name="fallback")

    assert out["embedded_chunks"] == 3
    assert out["model_chunks"] == 3
    assert out["semantic_status"] == "semantic_ready"
    assert calls == [3, 2, 1, 1, 1]


def test_repo_index_resumes_after_partial_flush(tmp_path: Path, monkeypatch):
    files = {f"src/f{i}.txt": f"value_{i}\n" for i in range(6)}
    repo = _make_text_repo(tmp_path, "resume-repo", files)
    mem = Memory(_cfg(tmp_path))
    monkeypatch.setenv("MEMO_REPO_FLUSH_BATCH", "1")

    from memo.store import VecStore

    original = VecStore.upsert_repo_files
    flush_calls = {"n": 0}

    def _flaky_flush(self, *, repo_id, repo_name, indexed_at, files):
        flush_calls["n"] += 1
        original(self, repo_id=repo_id, repo_name=repo_name, indexed_at=indexed_at, files=files)
        if flush_calls["n"] == 3:
            raise KeyboardInterrupt("simulated mid-run interrupt")

    monkeypatch.setattr(VecStore, "upsert_repo_files", _flaky_flush)

    try:
        mem.repo_index(str(repo), name="resume", with_embeddings=False)
        interrupted = False
    except KeyboardInterrupt:
        interrupted = True

    assert interrupted, "monkeypatched flush should have raised mid-scan"

    status = mem.repo_status("resume")
    assert status is not None
    assert status["files"] == 3
    assert status["status"] == "indexing"

    monkeypatch.setattr(VecStore, "upsert_repo_files", original)

    out = mem.repo_index(str(repo), name="resume", with_embeddings=False)

    assert out["resumed_partial"] is True
    assert out["unchanged_files"] == 3
    assert out["indexed_files"] == 3
    assert out["deleted_files"] == 0
    assert out["skipped_repo_unchanged"] is False

    final = mem.repo_status("resume")
    assert final is not None
    assert final["files"] == 6
    assert final["status"] in {"semantic_pending", "semantic_ready"}


def test_repo_index_uses_git_tracked_files_not_untracked_clone_files(tmp_path: Path, monkeypatch):
    repo = _make_repo(tmp_path)
    mem = Memory(_cfg(tmp_path))

    def _unexpected_embed(self, inputs):
        raise AssertionError("this test exercises exact indexing only")

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _unexpected_embed)
    first = mem.repo_index(str(repo), name="sample", with_embeddings=False)
    generated = Path(first["clone_path"]) / "generated" / "heavy.txt"
    generated.parent.mkdir()
    generated.write_text("untracked-token\n", encoding="utf-8")

    second = mem.repo_index(str(repo), name="sample", force=True, with_embeddings=False)

    assert second["checked_files"] == 3
    assert mem.repo_search("untracked-token", repo="sample", mode="line") == []


def test_repo_index_emits_progress_events(tmp_path: Path, monkeypatch):
    _patch_embedder(monkeypatch)
    repo = _make_repo(tmp_path)
    mem = Memory(_cfg(tmp_path))
    events: list[str] = []

    mem.repo_index(
        str(repo),
        name="sample",
        progress=lambda event, data: events.append(event),
    )

    assert "clone_start" in events
    assert "scan_start" in events
    assert "semantic_start" in events
    assert "semantic_batch" in events
    assert "semantic_done" in events
    assert "file_indexed" in events
    assert "write_start" in events
    assert "write_done" in events


def test_repo_cli_index_search_and_get(tmp_path: Path, monkeypatch):
    _patch_embedder(monkeypatch)
    repo = _make_repo(tmp_path)
    cfg_file = tmp_path / "memo.toml"
    env = {
        "MEMO_CONFIG_FILE": str(cfg_file),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
        "MEMO_RERANKER_ENABLED": "0",
    }
    runner = CliRunner()

    result = runner.invoke(cli, ["repo", "index", str(repo), "--name", "sample"], env=env)
    assert result.exit_code == 0, result.output
    assert "repo=sample" in result.output

    result = runner.invoke(
        cli,
        ["repo", "search", "needle-value", "--repo", "sample", "--mode", "line", "--json"],
        env=env,
    )
    assert result.exit_code == 0, result.output
    assert "src/app.py" in result.output
    assert "needle-value" in result.output

    result = runner.invoke(
        cli,
        ["repo", "get", "sample", "src/app.py", "--start", "2", "--end", "2", "--json"],
        env=env,
    )
    assert result.exit_code == 0, result.output
    assert "needle-value" in result.output


def test_repo_cli_index_emits_memflow_receipt(tmp_path: Path, monkeypatch):
    repo = _make_repo(tmp_path)
    memflow_root = tmp_path / "memflow"
    memflow_root.mkdir()
    cfg_file = tmp_path / "memo.toml"
    env = {
        "MEMO_CONFIG_FILE": str(cfg_file),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
        "MEMO_RERANKER_ENABLED": "0",
        "MEMFLOW_PROJECT_ROOT": str(memflow_root),
        "MEMO_MEMFLOW_BIN": "memflow-test",
    }
    calls: list[list[str]] = []

    def _fake_receipt(command, *, cwd, env):
        calls.append(list(command))
        assert cwd == memflow_root
        assert env["MEMFLOW_PROJECT_ROOT"] == str(memflow_root)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=".memflow/events/memories/fact/receipt.md\n",
            stderr="",
        )

    monkeypatch.setattr("memo.cli._run_memflow_receipt_command", _fake_receipt)

    result = CliRunner().invoke(
        cli,
        ["repo", "index", str(repo), "--name", "sample", "--no-embeddings"],
        env=env,
    )

    assert result.exit_code == 0, result.output
    assert "memflow receipt:" in result.output
    command = calls[0]
    assert command[:3] == ["memflow-test", "write", "fact"]
    assert "status=ok" in command
    assert "operation=repo_index" in command
    assert "repo_name=sample" in command
    assert "indexed_files=2" in command


def test_repo_cli_index_json_includes_memflow_receipt(tmp_path: Path, monkeypatch):
    repo = _make_repo(tmp_path)
    memflow_root = tmp_path / "memflow"
    memflow_root.mkdir()
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
        "MEMO_RERANKER_ENABLED": "0",
        "MEMFLOW_PROJECT_ROOT": str(memflow_root),
        "MEMO_MEMFLOW_BIN": "memflow-test",
    }

    def _fake_receipt(command, *, cwd, env):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=".memflow/events/memories/fact/receipt.md\n",
            stderr="",
        )

    monkeypatch.setattr("memo.cli._run_memflow_receipt_command", _fake_receipt)

    result = CliRunner().invoke(
        cli,
        ["repo", "index", str(repo), "--name", "sample", "--no-embeddings", "--json"],
        env=env,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["name"] == "sample"
    assert payload["memflow_receipt"] == {
        "ok": True,
        "path": ".memflow/events/memories/fact/receipt.md",
    }


def test_repo_cli_index_can_skip_memflow_receipt(tmp_path: Path, monkeypatch):
    repo = _make_repo(tmp_path)
    memflow_root = tmp_path / "memflow"
    memflow_root.mkdir()
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
        "MEMO_RERANKER_ENABLED": "0",
        "MEMFLOW_PROJECT_ROOT": str(memflow_root),
        "MEMO_MEMFLOW_BIN": "memflow-test",
    }
    calls: list[list[str]] = []

    def _fake_receipt(command, *, cwd, env):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="receipt.md\n", stderr="")

    monkeypatch.setattr("memo.cli._run_memflow_receipt_command", _fake_receipt)

    result = CliRunner().invoke(
        cli,
        [
            "repo",
            "index",
            str(repo),
            "--name",
            "sample",
            "--no-embeddings",
            "--no-memflow-receipt",
            "--json",
        ],
        env=env,
    )

    assert result.exit_code == 0, result.output
    assert calls == []
    payload = json.loads(result.output)
    assert payload["memflow_receipt"] == {
        "ok": False,
        "skipped": True,
        "reason": "disabled",
    }


def test_repo_cli_index_survives_memflow_receipt_failure(tmp_path: Path, monkeypatch):
    repo = _make_repo(tmp_path)
    memflow_root = tmp_path / "memflow"
    memflow_root.mkdir()
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
        "MEMO_RERANKER_ENABLED": "0",
        "MEMFLOW_PROJECT_ROOT": str(memflow_root),
        "MEMO_MEMFLOW_BIN": "memflow-test",
    }

    def _fake_receipt(command, *, cwd, env):
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="memflow down")

    monkeypatch.setattr("memo.cli._run_memflow_receipt_command", _fake_receipt)

    result = CliRunner().invoke(
        cli,
        ["repo", "index", str(repo), "--name", "sample", "--no-embeddings", "--json"],
        env=env,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["memflow_receipt"] == {"ok": False, "error": "memflow down"}


def test_repo_cli_index_error_emits_memflow_receipt(tmp_path: Path, monkeypatch):
    memflow_root = tmp_path / "memflow"
    memflow_root.mkdir()
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
        "MEMO_RERANKER_ENABLED": "0",
        "MEMFLOW_PROJECT_ROOT": str(memflow_root),
        "MEMO_MEMFLOW_BIN": "memflow-test",
    }
    calls: list[list[str]] = []

    def _raise_repo_index(self, *args, **kwargs):
        raise RuntimeError("index failed")

    def _fake_receipt(command, *, cwd, env):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="receipt.md\n", stderr="")

    monkeypatch.setattr("memo.memory.Memory.repo_index", _raise_repo_index)
    monkeypatch.setattr("memo.cli._run_memflow_receipt_command", _fake_receipt)

    result = CliRunner().invoke(
        cli,
        ["repo", "index", "https://example.test/repo.git", "--name", "sample"],
        env=env,
    )

    assert result.exit_code != 0
    command = calls[0]
    assert "status=error" in command
    assert "error_type=RuntimeError" in command
    assert "error=index failed" in command


def test_repo_mcp_tools(tmp_path: Path, monkeypatch):
    import asyncio

    from memo.server import build_server

    _patch_embedder(monkeypatch)
    repo = _make_repo(tmp_path)
    mem = Memory(_cfg(tmp_path))
    server = build_server(memory=mem)

    async def _tool(name: str) -> Any:
        tool = await server.get_tool(name)
        assert tool is not None
        return cast(Any, tool).fn

    index = asyncio.run(_tool("memory_repo_index"))
    search = asyncio.run(_tool("memory_repo_search"))
    get_file = asyncio.run(_tool("memory_repo_get_file"))

    indexed = index(url=str(repo), name="sample")
    assert indexed["indexed_files"] == 2
    assert indexed["semantic_status"] == "semantic_ready"

    hits = search(query="needle-value", repo="sample", mode="line")
    assert hits[0]["path"] == "src/app.py"
    assert hits[0]["line_start"] == 2

    file_out = get_file(repo="sample", path="src/app.py", start=2, end=2)
    assert "needle-value" in file_out["text"]

    embed = asyncio.run(_tool("memory_repo_embed"))
    status = asyncio.run(_tool("memory_repo_status"))

    embedded = embed(repo="sample")
    assert embedded["pending_chunks"] == 0
    assert status(repo="sample")["semantic_status"] == "semantic_ready"
