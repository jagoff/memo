from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from memo.cli import cli

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "memo"

# Foundation layer: building blocks that everything else composes. They must
# not import any *other* memo module, which keeps them free of import cycles
# and safe to reuse anywhere (incl. the god-modules memory.py / cli.py).
# (config.py is foundation too but lazily imports memo.setup for file IO, so
# it's intentionally excluded from this strict leaf set.)
FOUNDATION_MODULES = ["util", "store", "embedder", "graph"]


def _memo_imports(module_file: Path) -> set[str]:
    """Top-level memo submodules imported by `module_file` (via AST)."""
    tree = ast.parse(module_file.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("memo"):
            parts = node.module.split(".")
            if len(parts) >= 2:
                found.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "memo" and len(parts) >= 2:
                    found.add(parts[1])
    return found


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


@pytest.mark.parametrize("module", FOUNDATION_MODULES)
def test_foundation_modules_import_no_other_memo_module(module: str) -> None:
    """Foundation modules stay leaf-level — no memo->memo imports, no cycles."""
    imports = _memo_imports(SRC / f"{module}.py")
    imports.discard(module)
    assert imports == set(), (
        f"{module}.py must not import other memo modules, found: {sorted(imports)}"
    )


def test_store_never_imports_memory() -> None:
    """The store is below the Memory API; importing it back would be a cycle."""
    assert "memory" not in _memo_imports(SRC / "store.py")


def test_util_is_pure_stdlib_leaf() -> None:
    """memo.util is the bottom of the stack — it depends on nothing in memo."""
    assert _memo_imports(SRC / "util.py") == set()
