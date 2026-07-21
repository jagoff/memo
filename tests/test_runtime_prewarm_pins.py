from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from click.testing import CliRunner

from memo.runtime.daemon import prewarm


def test_download_all_uses_exact_chat_and_helper_revisions(monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, str]] = []
    llm_sha = "1" * 40
    helper_sha = "2" * 40
    cfg = SimpleNamespace(
        llm_model="someone/chat",
        llm_revision=llm_sha,
        helper_model="someone/helper",
        helper_revision=helper_sha,
        reranker_enabled=False,
        state_dir=tmp_path,
    )

    class _Embedder:
        def embed(self, _inputs):
            return [[1.0]]

    hf = ModuleType("huggingface_hub")

    def snapshot_download(*, repo_id: str, revision: str) -> str:
        calls.append((repo_id, revision))
        return f"/cache/{repo_id}"

    hf.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)
    monkeypatch.setattr("memo.config.Config.from_env", lambda: cfg)
    monkeypatch.setattr("memo.embedder_select.make_embedder", lambda _cfg: _Embedder())
    monkeypatch.setenv("MEMO_RECALL_DISABLE", "0")

    result = CliRunner().invoke(prewarm, ["--download-all"])

    assert result.exit_code == 0
    assert calls == [
        ("someone/chat", llm_sha),
        ("someone/helper", helper_sha),
    ]
