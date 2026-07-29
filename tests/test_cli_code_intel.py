"""Tests for `memo code-nudge` + `memo code-health` — graph↔memory awareness.

All tests run against a synthetic codegraph.db under tmp_path (never the
real `.codegraph/codegraph.db`) and an isolated `Config` via `tmp_cfg`.
git is never invoked for real: the diff-tree subprocess is monkeypatched.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.config import Config
from memo.memory import Memory

_STUB_DIMS = 8

# --- synthetic codegraph.db (shape of test_dream_code_drift._seed_graph) --------

_NODES = [
    # (id, kind, name, qualified_name, file_path, start_line, end_line)
    ("function:save", "function", "save", "memo.store.save", "src/memo/store.py", 10, 42),
    ("function:helper", "function", "helper", "memo.store.helper", "src/memo/store.py", 50, 60),
    ("function:caller", "function", "caller", "memo.cli.caller", "src/memo/cli.py", 5, 30),
    ("file:src/memo/store.py", "file", "store.py", None, "src/memo/store.py", None, None),
]

# `save` receives one call edge (the only hub); `helper` has zero callers.
_EDGES = [("function:caller", "function:save", "calls")]


def _seed_graph(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY, kind TEXT, name TEXT, qualified_name TEXT,
            file_path TEXT, start_line INTEGER, end_line INTEGER
        );
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);
        """
    )
    conn.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)", _NODES)
    conn.executemany("INSERT INTO edges VALUES (?, ?, ?)", _EDGES)
    conn.commit()
    conn.close()


def _ref(file_path: str, label: str = "", qualified: str = "", kind: str = "function") -> dict:
    # repo_id "" = no repo claim → verifiable against any DB (see code_intel).
    return {
        "uri": f"codegraph://testrepo/{label or file_path}",
        "repo_id": "",
        "kind": kind,
        "label": label,
        "qualified_name": qualified,
        "file_path": file_path,
        "relation": "modified",
        "confidence": 0.9,
    }


# --- isolation: stub embedder + no vec-similarity dedup on hash embeddings ------


@pytest.fixture(autouse=True)
def _stub_embedder(monkeypatch):
    """Deterministic per-text embeddings so seeding saves never load MLX."""

    def _embed(self, inputs):
        out = []
        for text in inputs:
            digest = hashlib.sha256((text or "").encode("utf-8")).digest()
            values = [((digest[i] / 255.0) * 2.0) - 1.0 for i in range(_STUB_DIMS)]
            norm = sum(v * v for v in values) ** 0.5
            out.append([v / norm for v in values])
        return out

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _embed)
    monkeypatch.setenv("MEMO_SAVE_DEDUP_CHECK", "0")


def _env(cfg: Config) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),
        "MEMO_STATE_DIR": str(cfg.state_dir),
        "MEMO_EMBEDDER_DIMS": str(_STUB_DIMS),
        "MEMO_EMBEDDER_MODEL": "stub",
        "MEMO_SAVE_DEDUP_CHECK": "0",
        "MEMO_AUTO_PROJECT_TAG": "0",
    }


def _seed_memories(cfg: Config, entries: list[tuple[str, list[dict]]]) -> list[str]:
    """Persist one type=fact memory per (title, code_refs) entry; return ids."""
    mem = Memory(
        Config(
            data_dir=cfg.data_dir,
            state_dir=cfg.state_dir,
            embedder_dims=_STUB_DIMS,
            reranker_enabled=False,
        )
    )
    try:
        return [
            mem.save(content=title, title=title, type_="fact", extra={"code_refs": refs}).id
            for title, refs in entries
        ]
    finally:
        mem.close()


def _patch_git(
    monkeypatch,
    *,
    stdout: str = "",
    returncode: int = 0,
    exc: Exception | None = None,
) -> list[list[str]]:
    """Intercept the `git diff-tree` subprocess; delegate anything else."""
    calls: list[list[str]] = []
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd[:2] == ["git", "diff-tree"]:
            calls.append(list(cmd))
            if exc is not None:
                raise exc
            return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr("memo.cli_code_intel.subprocess.run", fake_run)
    return calls


# --- code-nudge -------------------------------------------------------------------


def test_nudge_prints_memories_citing_commit_files(tmp_cfg: Config, monkeypatch) -> None:
    [mid] = _seed_memories(
        tmp_cfg,
        [("save() writes md first", [_ref("src/memo/store.py", "save", "memo.store.save")])],
    )
    _patch_git(monkeypatch, stdout="src/memo/store.py\ndocs/notes.md\n")

    res = CliRunner().invoke(cli, ["code-nudge"], env=_env(tmp_cfg))

    assert res.exit_code == 0, res.output
    assert f"[{mid[:8]}]" in res.output
    assert "save() writes md first" in res.output
    assert "🧠" in res.output


def test_nudge_forwards_commit_rev_to_git(tmp_cfg: Config, monkeypatch) -> None:
    calls = _patch_git(monkeypatch, stdout="")

    res = CliRunner().invoke(cli, ["code-nudge", "--commit", "abc123"], env=_env(tmp_cfg))

    assert res.exit_code == 0, res.output
    assert res.output == ""
    assert calls and calls[0][-1] == "abc123"


def test_nudge_without_match_is_silent(tmp_cfg: Config, monkeypatch) -> None:
    _seed_memories(
        tmp_cfg,
        [("about another file", [_ref("src/memo/other.py", "other", "memo.other.other")])],
    )
    _patch_git(monkeypatch, stdout="src/memo/unrelated.py\n")

    res = CliRunner().invoke(cli, ["code-nudge"], env=_env(tmp_cfg))

    assert res.exit_code == 0, res.output
    assert res.output == ""


def test_nudge_git_failure_is_silent(tmp_cfg: Config, monkeypatch) -> None:
    _patch_git(monkeypatch, exc=OSError("no git available"))

    res = CliRunner().invoke(cli, ["code-nudge"], env=_env(tmp_cfg))

    assert res.exit_code == 0, res.output
    assert res.output == ""


def test_nudge_caps_output_at_three_lines(tmp_cfg: Config, monkeypatch) -> None:
    ref = [_ref("src/memo/store.py", "save", "memo.store.save")]
    _seed_memories(tmp_cfg, [(f"memory number {i} about save", list(ref)) for i in range(4)])
    _patch_git(monkeypatch, stdout="src/memo/store.py\n")

    res = CliRunner().invoke(cli, ["code-nudge"], env=_env(tmp_cfg))

    assert res.exit_code == 0, res.output
    lines = [line for line in res.output.splitlines() if line.strip()]
    assert len(lines) == 3
    assert all(line.startswith("🧠") for line in lines)


def test_nudge_json_output(tmp_cfg: Config, monkeypatch) -> None:
    [mid] = _seed_memories(
        tmp_cfg,
        [("save() writes md first", [_ref("src/memo/store.py", "save", "memo.store.save")])],
    )
    _patch_git(monkeypatch, stdout="src/memo/store.py\n")

    res = CliRunner().invoke(cli, ["code-nudge", "--json"], env=_env(tmp_cfg))

    assert res.exit_code == 0, res.output
    assert json.loads(res.output) == [{"id": mid, "title": "save() writes md first"}]


# --- code-health ------------------------------------------------------------------


def test_health_json_has_stable_keys_and_uncited_hub(tmp_path: Path, tmp_cfg: Config) -> None:
    db = tmp_path / "cg" / ".codegraph" / "codegraph.db"
    _seed_graph(db)
    env = {**_env(tmp_cfg), "MEMO_CODEGRAPH_DB": str(db)}

    res = CliRunner().invoke(cli, ["code-health", "--json"], env=env)

    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert set(data) == {"drift", "dead_knowledge", "hubs_sin_memoria"}
    assert data["drift"]["status"] == "no-receipt"
    assert data["dead_knowledge"] == []
    hubs = data["hubs_sin_memoria"]
    assert [hub["name"] for hub in hubs] == ["save"]
    assert hubs[0]["callers"] == 1
    assert hubs[0]["file_path"] == "src/memo/store.py"


def test_health_dead_knowledge_and_cited_hub_excluded(tmp_path: Path, tmp_cfg: Config) -> None:
    db = tmp_path / "cg" / ".codegraph" / "codegraph.db"
    _seed_graph(db)
    dead_id, _ = _seed_memories(
        tmp_cfg,
        [
            ("helper handles retries", [_ref("src/memo/store.py", "helper", "memo.store.helper")]),
            ("save writes md first", [_ref("src/memo/store.py", "save", "memo.store.save")]),
        ],
    )
    env = {**_env(tmp_cfg), "MEMO_CODEGRAPH_DB": str(db)}

    res = CliRunner().invoke(cli, ["code-health", "--json"], env=env)

    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    # `helper` exists in the graph with 0 incoming calls edges → dead knowledge.
    assert [entry["id"] for entry in data["dead_knowledge"]] == [dead_id]
    assert data["dead_knowledge"][0]["symbols"] == ["helper"]
    # `save` (the only hub) is cited now → no hub gap.
    assert data["hubs_sin_memoria"] == []


def test_health_dead_knowledge_skips_foreign_repo_refs_by_uri(
    tmp_path: Path, tmp_cfg: Config
) -> None:
    db = tmp_path / "cg" / ".codegraph" / "codegraph.db"
    _seed_graph(db)
    # No repo_id field: the codegraph:// uri host is the repo claim (exactly
    # the refs `memo code-facts` mints). A foreign `helper` homonym must not
    # be judged against THIS graph, where the local `helper` has 0 callers.
    foreign_ref = {
        "uri": "codegraph://feedfacefeedface/function:helper",
        "kind": "function",
        "label": "helper",
        "qualified_name": "synapse.store.helper",
        "file_path": "src/synapse/store.py",
    }
    _seed_memories(tmp_cfg, [("foreign helper fact", [foreign_ref])])
    env = {**_env(tmp_cfg), "MEMO_CODEGRAPH_DB": str(db)}

    res = CliRunner().invoke(cli, ["code-health", "--json"], env=env)

    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["dead_knowledge"] == []


def test_health_summarizes_last_drift_receipt(tmp_path: Path, tmp_cfg: Config) -> None:
    db = tmp_path / "cg" / ".codegraph" / "codegraph.db"
    _seed_graph(db)
    receipt_dir = tmp_cfg.state_dir / "dream"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / "last.json").write_text(
        json.dumps(
            {
                "ts": 1.0,
                "code_drift": {
                    "status": "ok",
                    "scanned": 3,
                    "outdated": [{"id": "x"}],
                    "partial": [],
                    "repaired": [{"id": "y", "from": "a", "to": "b"}],
                },
            }
        ),
        encoding="utf-8",
    )
    env = {**_env(tmp_cfg), "MEMO_CODEGRAPH_DB": str(db)}

    res = CliRunner().invoke(cli, ["code-health", "--json"], env=env)

    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["drift"] == {
        "status": "ok",
        "scanned": 3,
        "outdated": 1,
        "partial": 0,
        "repaired": 1,
    }


def test_health_live_reverifies_every_ref(tmp_path: Path, tmp_cfg: Config) -> None:
    db = tmp_path / "cg" / ".codegraph" / "codegraph.db"
    _seed_graph(db)
    _seed_memories(
        tmp_cfg,
        [
            ("cites a live symbol", [_ref("src/memo/store.py", "save", "memo.store.save")]),
            ("cites a gone symbol", [_ref("src/memo/gone.py", "gone", "memo.gone.gone")]),
        ],
    )
    env = {**_env(tmp_cfg), "MEMO_CODEGRAPH_DB": str(db)}

    res = CliRunner().invoke(cli, ["code-health", "--live", "--json"], env=env)

    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["drift"]["live"] == {"vigente": 1, "desaparecido": 1, "no_verificable": 0}


def test_health_missing_db_notice_and_exit_zero(tmp_path: Path, tmp_cfg: Config) -> None:
    env = {**_env(tmp_cfg), "MEMO_CODEGRAPH_DB": str(tmp_path / "missing.db")}

    res = CliRunner().invoke(cli, ["code-health"], env=env)

    assert res.exit_code == 0, res.output
    assert "codegraph index no disponible" in res.output


def test_health_missing_db_json_keeps_stable_keys(tmp_path: Path, tmp_cfg: Config) -> None:
    env = {**_env(tmp_cfg), "MEMO_CODEGRAPH_DB": str(tmp_path / "missing.db")}

    res = CliRunner().invoke(cli, ["code-health", "--json"], env=env)

    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert set(data) == {"drift", "dead_knowledge", "hubs_sin_memoria"}
    assert data["dead_knowledge"] == []
    assert data["hubs_sin_memoria"] == []


# --- wiring -------------------------------------------------------------------------


def test_commands_registered_on_root_group(tmp_cfg: Config) -> None:
    res = CliRunner().invoke(cli, ["--help"], env=_env(tmp_cfg))

    assert res.exit_code == 0, res.output
    assert "code-nudge" in res.output
    assert "code-health" in res.output
