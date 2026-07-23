from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "memo"
FORBIDDEN_PACKAGES = {
    "consciousness_contracts",
    "memflow",
    "synapse",
}
REMOVED_MODULES = {
    "memo._trace",
    "memo.cli_crossdedup",
    "memo.consciousness_ledger",
    "memo.receipts",
    "memo.synapse_backend",
    "memo.synapse_client",
}


def test_runtime_has_no_external_memory_package_imports() -> None:
    violations: list[str] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                root = name.split(".", 1)[0]
                if root in FORBIDDEN_PACKAGES:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert violations == []


def test_removed_runtime_adapters_are_not_importable() -> None:
    assert {name for name in REMOVED_MODULES if importlib.util.find_spec(name) is not None} == set()


def test_package_metadata_has_no_private_contract_or_memory_daemon_dependency() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8").casefold()
    assert "consciousness-contracts" not in metadata
    assert '"memflow' not in metadata
    assert '"synapse' not in metadata
