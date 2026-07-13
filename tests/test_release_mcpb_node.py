from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BOOTSTRAP = _REPO_ROOT / "packaging" / "mcpb-node" / "bootstrap.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_bootstrap_js_syntax() -> None:
    result = subprocess.run(
        ["node", "--check", str(_BOOTSTRAP)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_bootstrap_js_is_zero_dep() -> None:
    source = _BOOTSTRAP.read_text(encoding="utf8")
    requires = re.findall(r"require\(\s*[\"']([^\"']+)[\"']\s*\)", source)
    assert requires, "expected at least one require() call in bootstrap.js"
    for module_name in requires:
        assert module_name.startswith("node:"), (
            f"require({module_name!r}) is not stdlib-prefixed — bootstrap.js must be zero-dep"
        )


def test_bootstrap_reads_pin_from_manifest() -> None:
    source = _BOOTSTRAP.read_text(encoding="utf8")
    assert "manifest.json" in source
    assert ".version" in source
