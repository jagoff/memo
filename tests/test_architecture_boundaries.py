from __future__ import annotations

import re
from pathlib import Path

from memo.cli import cli

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_brain_like_cli_groups_are_not_public() -> None:
    """Memo exposes corpus primitives; Synapse owns orchestration surfaces."""
    forbidden = {"agent", "cognitive", "federation", "lifecycle", "suggest"}

    assert forbidden.isdisjoint(cli.commands)


def test_brain_like_mcp_tools_are_not_registered() -> None:
    source = (REPO_ROOT / "src" / "memo" / "server.py").read_text(encoding="utf-8")
    forbidden = ("agent", "cognitive", "federation", "lifecycle", "suggest")

    for prefix in forbidden:
        pattern = rf"@server\.tool\(\)\s+def memory_{re.escape(prefix)}"
        assert re.search(pattern, source) is None


def test_repo_index_does_not_write_memflow_receipts() -> None:
    source = (REPO_ROOT / "src" / "memo" / "cli.py").read_text(encoding="utf-8")

    assert "memflow_receipt" not in source
    assert "MEMO_MEMFLOW" not in source
    assert "--no-memflow-receipt" not in source
