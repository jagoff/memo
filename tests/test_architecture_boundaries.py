from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from memo.cli import cli

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "memo"

# Foundation layer: building blocks that everything else composes. They must
# not import any *higher-level* memo module, which keeps them free of import
# cycles and safe to reuse anywhere (incl. memory.py / cli.py).
# (config.py is foundation too but lazily imports memo.setup for file IO, so
# it's intentionally excluded from this strict leaf set.)
FOUNDATION_MODULES = ["util", "store", "embedder", "graph"]

# Pure-stdlib leaves at the very bottom of the stack. Foundation modules may
# depend on these (they import nothing from memo, so no cycle is possible).
# embed_base = the shared EmbedderBase contract (collections.abc + typing only);
# MLXEmbedder/STEmbedder inherit it, so embedder.py imports it.
# tiers = the durable/reference/eviction-protected type registry (re + typing
# only, imports nothing from memo); store/signal_queries reads
# EVICTION_PROTECTED_TYPES from it, so store depends on this leaf.
PURE_LEAF_MODULES = {
    "util",
    "mlx_gpu",
    "graph_canonical",
    "embed_base",
    "model_pins",
    "tiers",
}

# Store migrations rebuild derived metadata from the Markdown/index source of
# truth. Reusing the product's canonical identity and privacy policies here is
# safer than copying those rules into the foundation layer. Both dependencies
# are acyclic leaves with respect to store; the separate store->memory guard
# below preserves the critical layering boundary.
FOUNDATION_ALLOWED_IMPORTS: dict[str, set[str]] = {
    "store": {"errors", "identity", "redact"},
}


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


def test_operational_surfaces_are_memo_owned() -> None:
    """Continuity and evidence are first-class Memo surfaces."""
    assert {"operational", "evidence"}.issubset(cli.commands)


def test_brain_like_mcp_tools_are_not_registered() -> None:
    """Memo keeps cognition OFF its MCP surface: no agent/cognitive/suggest
    verbs. MCP tools live across server.py + server_*.py (registered via
    ``@server.tool() def memo_*``) and mcp_tools.py (``ToolSpec name="memo_*"``);
    scan all of them so the guard keeps biting after the monolith split.

    Scope is the reduced post-#85 banned set: ``federation``/``lifecycle`` are
    now legitimate Memo-owned operational/evidence tools (e.g.
    ``memo_federation_preview``) and are no longer forbidden.
    """
    sources = [
        SRC / "server.py",
        SRC / "mcp_tools.py",
        *sorted(SRC.glob("server_*.py")),
    ]
    forbidden = ("agent", "cognitive", "suggest")

    for path in sources:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for prefix in forbidden:
            esc = re.escape(prefix)
            # Cover BOTH tool prefixes (memo_* and the session-pattern mem_*)
            # so a mem_cognitive_* tool cannot evade the guard by prefix.
            # mem_suggest_topic_key is exempt: a pure string-formatting helper
            # (type->family + kebab-case), no LLM/retrieval/orchestration.
            for match in re.finditer(rf"\bdef (?:memo|mem)_({esc}\w*)", text):
                assert f"mem_{match.group(1)}" == "mem_suggest_topic_key", (
                    f"brain-like MCP tool {match.group(0)[4:]}* defined in {path.name}"
                )
            assert re.search(rf'name="(?:memo|mem)_{esc}', text) is None, (
                f"brain-like MCP tool (memo|mem)_{prefix}* registered in {path.name}"
            )


def test_runtime_does_not_import_retired_memory_packages() -> None:
    retired = {"synapse", "memflow", "consciousness_contracts"}
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        imports = _memo_imports(path)
        external = retired & imports
        if external:
            violations.append(f"{path.relative_to(REPO_ROOT)}: {sorted(external)}")
    assert violations == []


def test_supported_windsurf_project_files_are_ignored() -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".windsurf/" in gitignore
    assert ".windsurfrules" in gitignore


@pytest.mark.parametrize("module", FOUNDATION_MODULES)
def test_foundation_modules_import_no_other_memo_module(module: str) -> None:
    """Foundation modules use only explicit acyclic lower-level dependencies."""
    imports = _module_imports(module)
    imports.discard(module)
    imports -= PURE_LEAF_MODULES  # depending on a pure-stdlib leaf can't cycle
    imports -= FOUNDATION_ALLOWED_IMPORTS.get(module, set())
    assert imports == set(), f"{module}.py imports undeclared memo modules: {sorted(imports)}"


def test_store_never_imports_memory() -> None:
    """The store is below the Memory API; importing it back would be a cycle."""
    assert "memory" not in _module_imports("store")


def test_memory_write_ops_uses_store_public_api() -> None:
    """The save path must not reach through VecStore's private connection."""
    source = (SRC / "memory" / "write_ops.py").read_text(encoding="utf-8")

    assert "store._conn" not in source


def test_util_is_pure_stdlib_leaf() -> None:
    """memo.util is the bottom of the stack — it depends on nothing in memo."""
    assert _memo_imports(SRC / "util.py") == set()


def test_mlx_gpu_is_pure_stdlib_leaf() -> None:
    """memo.mlx_gpu is a bottom-of-stack GPU-serialization leaf (threading
    only) — it must import nothing from memo so every MLX caller can depend
    on it without risking a cycle."""
    assert _memo_imports(SRC / "mlx_gpu.py") == set()


def test_graph_canonical_is_pure_stdlib_leaf() -> None:
    """memo.graph_canonical is a pure (re + unicodedata) canonicalization leaf —
    it must import nothing from memo so graph.py can depend on it cycle-free."""
    assert _memo_imports(SRC / "graph_canonical.py") == set()


# Optional Memory subsystems that MUST stay behind lazy @property accessors —
# constructing Memory() must not eagerly build any of them (cold-start cost +
# import-cycle risk). Guards the god-object's 3b decomposition contract.
LAZY_SUBSYSTEMS = [
    "temporal",
    "consolidator",
    "contradict_store",
    "contradict_scanner",
    "navigator",
    "contextual",
    "crossref",
    "link_suggester",
    "lifecycle",
    "versioning",
    "query_composer",
    "backup",
    "sync",
    "analytics",
    "dashboard",
    "import_export",
    "collaborative",
    "federation",
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


def test_optional_memory_capabilities_live_in_registry(mock_memory) -> None:
    """Experimental/advanced Memory subsystems are declared in one registry."""
    from memo.memory.capabilities import OPTIONAL_CAPABILITIES

    expected = {
        "analytics",
        "backup",
        "collaborative",
        "dashboard",
        "federation",
        "import_export",
        "lifecycle",
        "query_composer",
        "sync",
        "versioning",
    }

    assert expected.issubset(OPTIONAL_CAPABILITIES)
    assert mock_memory.capability("analytics") is mock_memory.analytics
    assert mock_memory.capability("lifecycle") is mock_memory.lifecycle


def test_behavior_flags_are_not_read_directly_from_environ() -> None:
    """Behavioral MEMO_* reads go through memo.flags, not ad-hoc env parsing."""
    allowed_modules = {
        SRC / "config.py",
        SRC / "flags.py",
        SRC / "flags_base.py",
        SRC / "flags_behavior.py",
        SRC / "flags_ingest.py",
        SRC / "flags_misc.py",
        SRC / "flags_recall.py",
        SRC / "flags_search.py",
    }
    config_owned = {
        "MEMO_CONFIG_FILE",
        "MEMO_DATA_DIR",
        "MEMO_EMBEDDER_DIMS",
        "MEMO_EMBEDDER_MODEL",
        "MEMO_EMBEDDER_REVISION",
        "MEMO_HELPER_MODEL",
        "MEMO_HELPER_REVISION",
        "MEMO_LLM_MODEL",
        "MEMO_LLM_REVISION",
        "MEMO_MAX_CONTENT_CHARS",
        "MEMO_MEMORY_SUBDIR",
        "MEMO_MODEL_PROFILE",
        "MEMO_RERANKER_ENABLED",
        "MEMO_RERANKER_MODEL",
        "MEMO_RERANKER_REVISION",
        "MEMO_RERANK_FUSION_ALPHA",
        "MEMO_RERANK_INPUT_K",
        "MEMO_SEARCH_DEFAULT_LIMIT",
        "MEMO_SINGLE_DB",
        "MEMO_STATE_DIR",
        "MEMO_VAULT_PATH",
    }
    pure_leaf_owned = {
        SRC / "embedder.py": {"MEMO_QUERY_CACHE_SIZE"},
        SRC / "mlx_gpu.py": {"MEMO_GPU_LOCK_PATH", "MEMO_GPU_XPROC_LOCK"},
        # Hot-path leaf (recall assoc): raw read keeps codegraph_loader free of
        # flags-registry imports; spec registered in flags_misc.py for validate.
        SRC / "codegraph_loader.py": {"MEMO_CODEGRAPH_DISCOVERY", "MEMO_CODEGRAPH_MAX_EDGES"},
        SRC / "store" / "schema.py": {"MEMO_SKIP_MODEL_VERSION_CHECK"},
        # MEMO_AGENT_TTY is set by the shim, not user-configurable; read here for IPC.
        SRC / "cli_session.py": {"MEMO_AGENT_TTY"},
        # autoupdate reads directly for the setdefault pattern (env check before flag default)
        SRC / "runtime" / "autoupdate.py": {"MEMO_AUTO_UPDATE"},
        # Three-way check (explicit 1 / explicit 0 / auto-detect) cannot use flag_bool
        # because it collapses unset → False; raw env read is the only way to distinguish
        # "not set" from "explicitly off".
        SRC / "memory" / "facade.py": {"MEMO_EMBEDDER_VIA_DAEMON"},
    }
    violations: list[str] = []

    for path in sorted(SRC.rglob("*.py")):
        if path in allowed_modules:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_env_get = (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id in {"os", "_os", "_os_min"}
            )
            is_getenv = (
                isinstance(func, ast.Attribute)
                and func.attr == "getenv"
                and isinstance(func.value, ast.Name)
                and func.value.id in {"os", "_os", "_os_min"}
            )
            if (
                (is_env_get or is_getenv)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value.startswith("MEMO_")
                and node.args[0].value not in config_owned
                and node.args[0].value not in pure_leaf_owned.get(path, set())
            ):
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {node.args[0].value}"
                )

    assert violations == []
