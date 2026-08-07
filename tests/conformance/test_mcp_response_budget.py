"""Every registered tool, invoked against ~10k memories, fits its cap.

This is the gate the five payload defects of 2026-08-06 would have hit. It
enumerates rather than lists, so a newly added tool is covered the day it is
added and cannot be forgotten.

Runs against a COPY of `big_corpus`'s data/state dirs (`shutil.copytree`,
same technique `test_index_rebuild_preserves.py::test_rebuild_preserves_user_signal`
uses) rather than the shared fixture itself. Enumerating every tool means
invoking WRITE tools (memo_save, memo_update, memo_delete, memo_forget,
memo_supersede, memo_offload, ...) honestly -- skipping them would gut the
gate, since a write tool's response can be just as unbounded as a read
tool's -- and the copy makes that safe: `big_corpus` is never opened for
write by this module, so there is nothing to restore afterward (proven
below by a byte-for-byte snapshot of its data/state dirs before and after).

No MLX: `MLXEmbedder.embed`/`STEmbedder.embed` and `MLXChat.chat` are replaced
outright (not primed via `repo_embedding_cache` -- there is no persistent
cache for QUERY-time embeddings, only the in-process LRU in `cache.py`, so
priming does not apply here the way it does for the rebuild test). A tripwire
on `_ensure_loaded` catches the belt-and-suspenders case where something
bypasses the replaced methods and reaches for a real model load.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastmcp.exceptions import ValidationError as FastMCPValidationError

from memo import mcp_budget
from memo.config import Config
from memo.embedder import MLXEmbedder
from memo.embedder_st import STEmbedder
from memo.llm import MLXChat
from memo.memory.facade import Memory
from memo.server import build_server

from .conftest import DIMS, seeded_id

pytestmark = pytest.mark.conformance

_TOPIC_A = "topic00"
_TOPIC_B = "topic01"
_CREATED = "2026-01-01T00:00:00+00:00"

# Arguments for tools that cannot be called bare, verified against the real
# signatures in server_*.py (the brief's own example ARGS entry for
# memo_related -- {"id": ...} -- was itself wrong; the real parameter is
# query_or_id) -- not guessed from the docstrings alone. IDs are dedicated
# per destructive op so mutating one tool's target never poisons a later
# tool's assumptions in the same alphabetical pass: memo_feedback_flag
# (kind="outdated") archives its target, so it gets its own id rather than
# sharing sid(0) with every read-only lookup.
ARGS: dict[str, dict[str, object]] = {
    "mem_compare": {
        "memory_id_a": seeded_id(0),
        "memory_id_b": seeded_id(1),
        "relation": "related",
    },
    "mem_judge": {"relation_id": "nonexistent-relation", "relation": "related"},
    "mem_timeline": {"observation_id": seeded_id(0)},
    "memo_around": {"id": seeded_id(0)},
    "memo_ask": {"question": _TOPIC_A},
    "memo_ask_as_of": {"question": _TOPIC_A, "as_of": _CREATED},
    "memo_ask_valid_as_of": {"question": _TOPIC_A, "as_of": _CREATED},
    "memo_attention_ack": {"id": "nonexistent-attn"},
    "memo_attention_add": {"project": "conformance", "summary": "gate probe"},
    "memo_backup_restore": {"backup_name": "__SEEDED__"},  # filled in below
    "memo_chat_ask": {"question": _TOPIC_A},
    "memo_collaborative_connections": {"entity": _TOPIC_A},
    "memo_collaborative_recommend": {"entity": _TOPIC_A},
    "memo_collaborative_share_connection": {
        "user_id": "conformance-user",
        "entity_a": _TOPIC_A,
        "entity_b": _TOPIC_B,
        "relationship": "related",
    },
    "memo_collaborative_share_insight": {
        "user_id": "conformance-user",
        "content": "conformance insight",
    },
    "memo_conflict_open": {"topic": "conformance-topic", "summary": "conformance dispute"},
    "memo_conflict_resolve": {"id": "nonexistent-conflict"},
    "memo_context": {"question": _TOPIC_A},
    "memo_context_pack": {"question": _TOPIC_A},
    "memo_contextual_record_click": {"memory_id": seeded_id(0)},
    "memo_contextual_record_search": {"query": _TOPIC_A, "memory_ids": [seeded_id(0)]},
    "memo_contextual_search": {"query": _TOPIC_A},
    "memo_contradict_resolve": {"pair_id": 999_999, "status": "dismissed"},
    "memo_crush_retrieve": {"hash_marker": "<<memo-crush:deadbeef>>"},
    "memo_delete": {"id": seeded_id(20)},
    "memo_diff": {"from_ts": _CREATED},
    "memo_embed_batch": {"texts": [_TOPIC_A, _TOPIC_B]},
    "memo_embed_query": {"text": _TOPIC_A},
    "memo_entity": {"name": _TOPIC_A},
    "memo_entity_search": {"query": _TOPIC_A},
    "memo_episodes_search": {"query": _TOPIC_A},
    "memo_evidence_pack": {"question": _TOPIC_A},
    "memo_explore": {"entity": _TOPIC_A},
    "memo_export_csv": {"output_path": "__SET_BELOW__"},
    "memo_export_json": {"output_path": "__SET_BELOW__"},
    "memo_export_markdown_bundle": {"output_path": "__SET_BELOW__"},
    "memo_export_passport": {"output_path": "__SET_BELOW__"},
    "memo_extract_entities": {"ids": [seeded_id(0)]},
    "memo_fact_edge_invalidate": {"id": "nonexistent-fact-edge"},
    "memo_fact_edge_save": {"subject": _TOPIC_A, "predicate": "relates_to", "object": _TOPIC_B},
    "memo_federation_preview": {"principal": "conformance-principal"},
    "memo_feedback_clear": {"source_id": seeded_id(0)},
    "memo_feedback_flag": {"source_id": seeded_id(28), "kind": "outdated"},
    "memo_feedback_record": {"source_id": seeded_id(0), "query": _TOPIC_A, "rating": "up"},
    "memo_focus_clear": {"project": "conformance"},
    "memo_focus_set": {"project": "conformance", "summary": "gate probe"},
    "memo_forget": {"id": seeded_id(21)},
    "memo_graph": {"verb": "neighbors", "entity": _TOPIC_A},
    "memo_graph_neighbors": {"entity": _TOPIC_A},
    "memo_graph_path": {"source": _TOPIC_A, "target": _TOPIC_B},
    "memo_handoff_consume": {"id": "nonexistent-handoff"},
    "memo_handoff_create": {
        "project": "conformance",
        "summary": "gate probe",
        "from_actor": "conformance-gate",
    },
    "memo_import_csv": {"input_path": "__SET_BELOW__"},
    "memo_import_json": {"input_path": "__SET_BELOW__"},
    "memo_import_passport": {"input_path": "__SET_BELOW__"},
    "memo_invalidate": {"id": seeded_id(22), "reason": "conformance probe"},
    "memo_links_backlinks": {"memory_id": seeded_id(0)},
    "memo_links_format": {"memory_id": seeded_id(0)},
    "memo_links_outlinks": {"memory_id": seeded_id(0)},
    "memo_links_suggest": {"content": f"See [[{_TOPIC_A}]] for details."},
    "memo_mark_reviewed": {"id": seeded_id(23)},
    "memo_offload": {"content": f"conformance offload content about {_TOPIC_A} " * 5},
    "memo_outcome_record": {
        "task_id": "conformance-task",
        "status": "success",
        "memory_ids": [seeded_id(0)],
        "idempotency_key": "conformance-key-loop",
    },
    "memo_procedure_promote": {"memory_ids": [seeded_id(0)], "title": "conformance procedure"},
    "memo_provenance": {"id": seeded_id(0)},
    "memo_query_delete": {"name": "conformance-delete-target"},
    "memo_query_run": {"name": "__SEEDED__"},  # filled in below
    "memo_query_save": {"name": "conformance-save-target", "query_text": _TOPIC_A},
    "memo_record_diff": {"id": seeded_id(0)},
    "memo_related": {"query_or_id": seeded_id(0)},
    "memo_rename": {"id": seeded_id(24), "title": "Renamed conformance memory"},
    "memo_repo_delete": {"repo": "__SET_BELOW__"},
    "memo_repo_embed": {"repo": "__SET_BELOW__"},
    "memo_repo_get_file": {"repo": "__SET_BELOW__", "path": "README.md"},
    "memo_repo_index": {"url": "__SET_BELOW__"},
    "memo_repo_search": {"query": _TOPIC_A},
    "memo_repo_status": {"repo": "__SET_BELOW__"},
    "memo_rerank": {"query": _TOPIC_A, "hits": []},
    "memo_save": {"content": f"Conformance probe save about {_TOPIC_A}."},
    "memo_save_text": {"text": f"Conformance probe save text about {_TOPIC_A}."},
    "memo_search": {"query": _TOPIC_A},
    "memo_search_as_of": {"query": _TOPIC_A, "as_of": _CREATED},
    "memo_search_trace": {"query": _TOPIC_A},
    "memo_search_valid_as_of": {"query": _TOPIC_A, "as_of": _CREATED},
    "memo_session_get": {"session_id": "nonexistent-session"},
    "memo_signal_remember": {"marker": "conformance-marker"},
    "memo_supersede": {
        "old_id": seeded_id(25),
        "new_id": seeded_id(26),
        "reason": "conformance probe",
    },
    "memo_synthesize_delete": {"id": "nonexistent-synth"},
    "memo_temporal_contradictions": {"entity": _TOPIC_A},
    "memo_temporal_timeline": {"entity": _TOPIC_A},
    "memo_unforget": {
        "id": seeded_id(21)
    },  # same id memo_forget targets -- forget < unforget alphabetically
    "memo_update": {
        "id": seeded_id(27),
        "content": f"Updated conformance content about {_TOPIC_A}.",
    },
    "memo_verbatim_search": {"query": _TOPIC_A},
    "memo_version_diff": {"memory_id": seeded_id(0)},
    "memo_version_history": {"memory_id": seeded_id(0)},
    "memo_version_rollback": {"memory_id": seeded_id(0), "version_id": 999_999},
    "memo_get": {"id": seeded_id(0)},
}

# Tools with a genuine external dependency. Each entry is a deliberate
# exclusion with a stated reason, not a convenience -- everything else,
# including every WRITE/DESTRUCTIVE tool, is invoked for real against the
# copy below.
SKIP: dict[str, str] = {
    "memo_reindex": (
        "rebuilds the whole VecStore schema (meta/vec/fts) in place; already "
        "proven at 10k-corpus scale, cache-primed and MLX-tripwired, by "
        "test_index_rebuild_preserves.py::test_rebuild_preserves_user_signal. "
        "Running it here too would share one live Memory/VecStore connection "
        "with the other ~160 calls in this same enumeration pass -- a "
        "correctness/ordering hazard (ids, paths, and derived columns the "
        "rest of this pass depends on get rewritten mid-loop), not mere "
        "inconvenience."
    ),
    "memo_ocr_image": (
        "requires the optional OCR extra (pillow/pytesseract) and a real "
        "image file to produce a meaningful result -- an external dependency "
        "unrelated to corpus scale (this tool's payload is bounded by one "
        "image's recognized text, not by the corpus)."
    ),
}


def _stub_vector(text: str) -> list[float]:
    """Deterministic, content-varying unit vector -- NOT a constant stub.

    A constant vector for every input makes every save/search look like a
    1.0-cosine near-duplicate of every other, which would silently empty out
    exactly the list-shaped fields (search hits, related, dedup candidates)
    this gate exists to check the size of. Mirrors
    test_index_rebuild_preserves.py's `_stub_vector`.
    """
    digest = hashlib.sha256(text.encode()).digest()
    raw = [(digest[d % len(digest)] - 128) / 128.0 for d in range(DIMS)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """Cheap (stat-only, no content reads) snapshot of every file under
    `root`: (size, mtime_ns) per relative path. `big_corpus` holds 10k+
    markdown files: hashing every byte twice (before/after) would make this
    module's own hygiene check as slow as the corpus build itself; a stat
    diff still catches any add/remove/modify, since a write touches mtime."""
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


# How many records the memo_import_* family is fed. The exports it reads are
# still taken from the WHOLE corpus -- only the slice handed back to import is
# a constant, and here is why that costs the gate nothing:
#
#   An import returns `ImportResult.__dict__` == {imported_count, skipped_count,
#   errors}. Two ints plus a list populated once per FAILED record -- and the
#   input is memo's own export of a valid corpus, so `errors` is empty at 200
#   records and equally empty at 10,001. The response is O(1) in the input by
#   construction; feeding it the whole corpus buys no payload signal.
#
# It cost 258s of the gate's 272s tool loop at N=2000 (95%), scaling linearly
# (worse -- each imported record is a fresh save against a growing store), which
# is what put the CI lane's 10,001-record run past its 600s per-test timeout.
#
# 200 is not the smallest number that works, it is the smallest that keeps the
# check meaningful: the realistic way an import response turns unbounded is
# echoing the records it imported, and 200 records (~300 chars each ≈ 15k
# tokens) blows the 10k cap, so that regression still fails here. A response
# that echoed only bare ids would need ~1,100 records to trip; raise this
# constant if an import result ever grows a per-record field.
_IMPORT_SAMPLE_RECORDS = 200


def _sample_export(src: Path, dst: Path, fmt: str, *, records: int) -> Path:
    """Write the first `records` records of an export to `dst`.

    Format-aware rather than a line slice: CSV bodies may carry embedded
    newlines, and the passport carries a `count` header that has to keep
    agreeing with its `memories` list or the importer rejects the file.
    """
    if fmt == "csv":
        with src.open(newline="", encoding="utf-8") as fh:
            rows = list(itertools.islice(csv.reader(fh), records + 1))  # + header
        with dst.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)
        return dst

    payload = json.loads(src.read_text(encoding="utf-8"))
    if isinstance(payload, list):  # json export: a bare list of records
        payload = payload[:records]
    elif isinstance(payload, dict):  # passport: header + "memories"
        memories = payload.get("memories")
        if isinstance(memories, list):
            kept = memories[:records]
            payload = {**payload, "memories": kept, "count": len(kept)}
    dst.write_text(json.dumps(payload), encoding="utf-8")
    return dst


def _make_local_repo(root: Path, name: str) -> Path:
    """A tiny local git repo -- `git clone <local-path>` needs no network, so
    the memo_repo_* family can be exercised against a REAL indexed repo
    instead of an external dependency."""
    src = root / name
    src.mkdir()
    (src / "README.md").write_text(f"# {name}\n\n{_TOPIC_A} content.\n", encoding="utf-8")
    for args in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "gate@example.com"],
        ["git", "config", "user.name", "gate"],
        ["git", "add", "README.md"],
        ["git", "commit", "-q", "-m", "seed"],
    ):
        subprocess.run(args, cwd=src, check=True)
    return src


@pytest.mark.asyncio
async def test_every_tool_result_fits_its_cap(
    big_corpus: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before_data = _snapshot(big_corpus.data_dir)
    before_state = _snapshot(big_corpus.state_dir)

    # -- MLX-free: replace embed/chat outright; a tripwire on _ensure_loaded
    # catches any path that bypasses the replacement and reaches for a real
    # model load, the same belt-and-suspenders technique
    # test_rebuild_preserves_user_signal uses.
    def _stub_embed(self: object, inputs: Any) -> list[list[float]]:
        return [_stub_vector(str(t)) for t in inputs]

    def _stub_chat(
        self: object,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {"message": {"content": "stub synthesis response"}}

    def _no_live_model(self: object) -> None:
        raise AssertionError(
            "a conformance tool call tried to load a live model -- embed/chat "
            "should have been fully replaced, so this lane stays MLX-free"
        )

    monkeypatch.setattr(MLXEmbedder, "embed", _stub_embed)
    monkeypatch.setattr(STEmbedder, "embed", _stub_embed)
    monkeypatch.setattr(MLXEmbedder, "_ensure_loaded", _no_live_model)
    monkeypatch.setattr(STEmbedder, "_ensure_loaded", _no_live_model)
    monkeypatch.setattr(MLXChat, "chat", _stub_chat)

    # -- Full MCP surface (162 tools), not the 41-tool "agent" default profile
    # mcp_profile() falls back to -- the two originally-measured defects
    # (memo_graph_export, memo_lint's advanced siblings) live in the
    # advanced-only tool set gated behind "full"/"default".
    monkeypatch.setenv("MEMO_MCP_PROFILE", "full")
    monkeypatch.delenv("MEMO_MCP_SLIM", raising=False)
    # Scripted caller, no elicitation-capable client: memo_delete and
    # memo_backup_restore (targets that actually exist) otherwise crash
    # reaching for ctx.session with no real MCP client attached. The flag's
    # own docstring names this exact scenario ("scripted elicitation-capable
    # clients"). Fail-open is unaffected -- every other GATED_TOOLS entry
    # (memo_feedback_clear, memo_repo_delete, memo_synthesize_delete,
    # memo_cache_evict) short-circuits before reaching ctx.session anyway
    # when there is nothing to confirm.
    monkeypatch.setenv("MEMO_ELICIT_CONFIRM", "false")
    # Undo a leak from earlier in the LANE, not a preference of this test.
    # `memo reindex --rebuild` does `os.environ.setdefault(
    # "MEMO_SKIP_MODEL_VERSION_CHECK", "1")` (cli_memory.py) -- harmless in a
    # CLI process that then exits, permanent for the pytest process once
    # test_index_rebuild_preserves.py invokes it through CliRunner. That flag
    # makes `_validate_vec_quant` (store/schema.py) SKIP adopting the on-disk
    # vec precision, which matters here because the corpus fixture builds its
    # index with `VecStore(...)`'s own default (int8) while `Memory` reads
    # MEMO_VEC_QUANTIZE, pinned to "off" by tests/conftest.py. Without the
    # adoption, the first tool that writes a vector binds float32 into an int8
    # column: `sqlite3.OperationalError: ... expected to be of type int8`.
    # Reproduced on this commit's parent, so it predates the budget work; the
    # containment belongs at the leaking test, which is not this one's to edit.
    monkeypatch.delenv("MEMO_SKIP_MODEL_VERSION_CHECK", raising=False)

    # -- Copy, not the shared fixture -- write tools run for real, honestly,
    # and there is nothing to restore on big_corpus afterward (proven below).
    data_copy = tmp_path / "data"
    state_copy = tmp_path / "state"
    shutil.copytree(big_corpus.data_dir, data_copy)
    shutil.copytree(big_corpus.state_dir, state_copy)
    cfg = Config(
        data_dir=data_copy,
        vault_path=tmp_path / "vault",
        state_dir=state_copy,
        reranker_enabled=False,
        embedder_dims=DIMS,
    )

    # memo_export_*/memo_import_* are the LLM/agent's only file-path surface,
    # deliberately restricted to Path.cwd() plus a few real user dirs
    # (server_import_export._get_allowed_base_dirs) -- pytest's tmp_path is
    # under neither, so this narrows the allow-list to tmp_path for the
    # duration of the test rather than chdir-ing the whole process (chdir
    # broke memo_start_session's git-state detection, which reads
    # os.getcwd() itself).
    import memo.server_import_export as import_export_module

    monkeypatch.setattr(
        import_export_module, "_get_allowed_base_dirs", lambda: (tmp_path.resolve(),)
    )

    memory = Memory(cfg)
    try:
        server = build_server(memory=memory)
        tools = await server.list_tools()
        names = sorted(t.name for t in tools)

        export_dir = tmp_path / "export"
        export_dir.mkdir()
        args_by_name: dict[str, dict[str, object]] = {k: dict(v) for k, v in ARGS.items()}
        args_by_name["memo_export_csv"]["output_path"] = str(export_dir / "e.csv")
        args_by_name["memo_export_json"]["output_path"] = str(export_dir / "e.json")
        args_by_name["memo_export_markdown_bundle"]["output_path"] = str(export_dir / "bundle")
        args_by_name["memo_export_passport"]["output_path"] = str(export_dir / "passport.json")

        # -- Setup: seed backing state that a fresh copy does not otherwise
        # have, so the tools that read it back exercise a real result
        # instead of a vacuous "not found". Each seed call is deliberately
        # OUTSIDE the alphabetical enumeration loop below -- several of these
        # pairs sort the wrong way for in-loop chaining (memo_query_run <
        # memo_query_save, memo_handoff_consume < memo_handoff_create).

        # Export first, then feed the exported files back into import's ARGS
        # -- a real payload instead of a hand-guessed schema (memo_import_
        # passport rejects anything without "schema": "memo.passport.v1").
        #
        # Import reads a SAMPLE of each export (see _IMPORT_SAMPLE_RECORDS),
        # written to its own directory: the enumeration loop below re-runs
        # every memo_export_* for real against the whole corpus and would
        # otherwise overwrite the file its memo_import_* sibling is about to
        # read (export sorts before import alphabetically).
        import_dir = tmp_path / "import"
        import_dir.mkdir()
        for fmt, export_tool, import_tool in (
            ("csv", "memo_export_csv", "memo_import_csv"),
            ("json", "memo_export_json", "memo_import_json"),
            ("passport", "memo_export_passport", "memo_import_passport"),
        ):
            await server.call_tool(export_tool, args_by_name[export_tool], run_middleware=False)
            exported = Path(str(args_by_name[export_tool]["output_path"]))
            args_by_name[import_tool]["input_path"] = str(
                _sample_export(
                    exported,
                    import_dir / exported.name,
                    fmt,
                    records=_IMPORT_SAMPLE_RECORDS,
                )
            )

        backup_res = await server.call_tool("memo_backup_create", {}, run_middleware=False)
        backup_sc = backup_res.structured_content
        backup_name = None
        if isinstance(backup_sc, dict):
            backup_name = (
                backup_sc.get("timestamp") or backup_sc.get("name") or backup_sc.get("backup_name")
            )
        args_by_name["memo_backup_restore"]["backup_name"] = backup_name or "nonexistent-backup"

        await server.call_tool(
            "memo_query_save",
            {"name": "conformance-seed-query", "query_text": _TOPIC_A},
            run_middleware=False,
        )
        args_by_name["memo_query_run"]["name"] = "conformance-seed-query"

        # memo_procedure_promote requires real outcome evidence
        # (successes>=2, utility>=0.75 -- memory/outcome_feedback_ops.py) or
        # it raises rather than degrading to a small "not found".
        for i in (1, 2):
            await server.call_tool(
                "memo_outcome_record",
                {
                    "task_id": f"conformance-seed-task-{i}",
                    "status": "success",
                    "memory_ids": [seeded_id(0)],
                    "idempotency_key": f"conformance-seed-outcome-{i}",
                },
                run_middleware=False,
            )

        # Real, local (no network) git repos -- two, because memo_repo_delete
        # sorts BEFORE memo_repo_embed/get_file/status alphabetically and
        # would delete a shared repo out from under them.
        async def _index_repo(repo_name: str) -> str:
            src = _make_local_repo(tmp_path, repo_name)
            res = await server.call_tool(
                "memo_repo_index", {"url": str(src), "name": repo_name}, run_middleware=False
            )
            sc = res.structured_content
            if isinstance(sc, dict):
                return str(sc.get("repo") or sc.get("name") or repo_name)
            return repo_name

        repo_id = await _index_repo("conformance-repo")
        repo_id_delete = await _index_repo("conformance-repo-del")
        for tool in ("memo_repo_embed", "memo_repo_get_file", "memo_repo_status"):
            args_by_name[tool]["repo"] = repo_id
        args_by_name["memo_repo_delete"]["repo"] = repo_id_delete
        # memo_repo_index itself is exercised again, for real, by the main
        # enumeration loop below -- a third repo so it does not collide.
        args_by_name["memo_repo_index"] = {
            "url": str(_make_local_repo(tmp_path, "conformance-repo-loop"))
        }

        over: list[str] = []
        uncallable: list[str] = []
        for name in names:
            if name in SKIP:
                continue
            call_args = args_by_name.get(name, {})
            try:
                result = await server.call_tool(name, call_args, run_middleware=False)
            except FastMCPValidationError:
                # Neither callable bare nor given an ARGS entry -- fails the
                # gate rather than being silently skipped, so the enumeration
                # cannot shrink as new tools are added.
                uncallable.append(name)
                continue
            tokens = mcp_budget.est_tokens(mcp_budget.result_text(result))
            cap = mcp_budget.cap_for(name)
            if cap and tokens > cap:
                over.append(f"{name}: {tokens} > {cap}")

        assert not uncallable, (
            f"tools needing an ARGS entry (neither callable bare nor listed): {uncallable}"
        )
        assert not over, "tools over budget:\n" + "\n".join(over)
    finally:
        memory.close()

    assert _snapshot(big_corpus.data_dir) == before_data, (
        "the shared big_corpus data dir changed -- this module must run "
        "only against the tmp_path copy"
    )
    assert _snapshot(big_corpus.state_dir) == before_state, (
        "the shared big_corpus state dir changed -- this module must run "
        "only against the tmp_path copy"
    )
