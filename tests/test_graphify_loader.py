from __future__ import annotations

import subprocess
from pathlib import Path

from memo import graphify_loader


def test_refresh_uses_headless_graphify_update(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    graphify_out = repo_root / "graphify-out"
    graphify_out.mkdir(parents=True)
    monkeypatch.setattr(graphify_loader, "GRAPHIFY_OUT", graphify_out)
    monkeypatch.setattr(graphify_loader, "GRAPHIFY_JSON", graphify_out / "graph.json")
    monkeypatch.setattr(graphify_loader, "_graph", ({"old": set()}, {}))

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="updated", stderr="")

    monkeypatch.setattr(graphify_loader.subprocess, "run", fake_run)

    assert graphify_loader.refresh(force=True) is True
    assert calls == [["graphify", "update", str(repo_root), "--force"]]
    assert graphify_loader.node_count() == 0
