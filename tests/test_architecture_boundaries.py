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


def _module_imports(module: str) -> set[str]:
    """memo submodules imported by `module`, whether it's a single-file
    module (`SRC / f"{module}.py"`) or a package directory (`SRC / module`).
    For a package, unions `_memo_imports` over every `*.py` inside it."""
    single = SRC / f"{module}.py"
    if single.exists():
        return _memo_imports(single)
    pkg = SRC / module
    if pkg.is_dir():
        found: set[str] = set()
        for py in sorted(pkg.glob("*.py")):
            found |= _memo_imports(py)
        return found
    return _memo_imports(single)


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
    imports = _module_imports(module)
    imports.discard(module)
    assert imports == set(), (
        f"{module}.py must not import other memo modules, found: {sorted(imports)}"
    )


def test_store_never_imports_memory() -> None:
    """The store is below the Memory API; importing it back would be a cycle."""
    assert "memory" not in _module_imports("store")


def test_util_is_pure_stdlib_leaf() -> None:
    """memo.util is the bottom of the stack — it depends on nothing in memo."""
    assert _memo_imports(SRC / "util.py") == set()


# Optional Memory subsystems that MUST stay behind lazy @property accessors —
# constructing Memory() must not eagerly build any of them (cold-start cost +
# import-cycle risk). Guards the god-object's 3b decomposition contract.
LAZY_SUBSYSTEMS = [
    "temporal", "consolidator", "contradict_store", "contradict_scanner",
    "navigator", "contextual", "crossref", "link_suggester", "lifecycle",
    "proactive", "versioning", "query_composer", "federation", "backup",
    "sync", "encryption", "sharing", "analytics", "dashboard",
    "import_export", "multimodal", "collaborative",
]


@pytest.mark.parametrize("name", LAZY_SUBSYSTEMS)
def test_subsystem_is_a_property_not_eager_attr(name: str) -> None:
    """Each optional subsystem is a property descriptor on Memory — so it is
    only built on access, never as a plain __init__ attribute."""
    from memo.memory import Memory

    attr = getattr(Memory, name, None)
    assert isinstance(attr, property), f"Memory.{name} must stay a lazy @property"


def test_constructing_memory_builds_no_subsystem(mock_memory) -> None:
    """A freshly built Memory has every cached subsystem backing field unset —
    construction is cheap and triggers no subsystem cold-start."""
    for name in LAZY_SUBSYSTEMS:
        backing = f"_{name}"
        if hasattr(mock_memory, backing):
            assert getattr(mock_memory, backing) is None, (
                f"Memory.__init__ eagerly built {name} (backing {backing} not None)"
            )


def test_lazy_property_caches_after_access(mock_memory) -> None:
    """Accessing a cached lazy property builds it once and memoizes."""
    assert mock_memory._temporal is None
    first = mock_memory.temporal
    assert mock_memory._temporal is first
    assert mock_memory.temporal is first
