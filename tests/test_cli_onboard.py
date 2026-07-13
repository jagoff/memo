"""Tests for the `memo onboard` Day-0 wizard."""
from __future__ import annotations

import os
import time


def _env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }


def test_onboard_backfill_days_flag_registered():
    from memo.flags import REGISTRY

    spec = REGISTRY["MEMO_ONBOARD_BACKFILL_DAYS"]
    assert spec.default == 90


def test_recent_memories_orders_by_mtime_and_skips_buckets(tmp_path):
    from memo.cli_onboard import _recent_memories

    root = tmp_path / "mem"
    root.mkdir()
    for i, name in enumerate(["old", "mid", "new"]):
        p = root / f"{name}.md"
        p.write_text(f"---\nid: {'a' * 32}\n---\n# titulo {name}\n", encoding="utf-8")
        os.utime(p, (time.time() - 100 + i, time.time() - 100 + i))
    bucket = root / "_profile"
    bucket.mkdir()
    (bucket / "profile.md").write_text("# not a memory\n", encoding="utf-8")

    out = _recent_memories(root, n=2)
    assert [m["title"] for m in out] == ["titulo new", "titulo mid"]


def test_recent_memories_empty_dir(tmp_path):
    from memo.cli_onboard import _recent_memories

    assert _recent_memories(tmp_path / "nope") == []
