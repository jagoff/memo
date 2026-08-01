from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from memo.cli_definitive import definitive_group


def test_definitive_integration_writes_verified_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "integration-receipt.json"
    result = CliRunner().invoke(
        definitive_group,
        ["integration", "--receipt", str(receipt)],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["ok"] is True
    assert all(report["checks"].values())
    assert report["checks"]["terminal_mesh_bidirectional"] is True
    assert report["checks"]["terminal_a_observes_b"] is True
    assert report["checks"]["terminal_b_observes_a"] is True
    assert report["checks"]["acks_bidirectional"] is True
    assert report["scope"].endswith("physical Mac reboot excluded")
    assert receipt.is_file()
    assert Path(report["roster_path"]).is_file()
    stored = json.loads(receipt.read_text(encoding="utf-8"))
    assert stored["receipt_sha256"] == report["receipt_sha256"]
    assert stored["signature"]["signature"]
