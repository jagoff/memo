"""Diagnostic + profile helpers for the memo CLI.

Extracted from cli.py (3a). Pure cfg->dict reporters shared by the
`doctor`/`stats` commands (still in cli.py) and the `profile` +
`backend-native` groups (cli_profile.py / cli_backend_native.py).
"""

from __future__ import annotations

import os
import re
import shlex
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memo.config import MODEL_PROFILES, Config


def _gc_report(cfg: Config, *, fix: bool) -> dict[str, list[str]]:
    """Run the orphan scan and deterministically release Memory resources."""
    from memo.memory import Memory

    with closing(Memory(cfg)) as mem:
        return mem.gc(fix=fix)


def _managed_sqlite_dbs(cfg: Config) -> list[tuple[str, Path]]:
    candidates = [
        ("memvec", cfg.db_path),
        ("history", cfg.history_db),
        ("graph", cfg.graph_db),
        ("crossref", cfg.crossref_db),
        ("contradictions", cfg.contradictions_db),
    ]
    # Under single_db the sidecar paths collapse onto db_path — list each
    # physical file once so health output isn't five identical rows.
    seen: set[str] = set()
    out: list[tuple[str, Path]] = []
    for label, path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append((label, path))
    return out


def _sqlite_table_exists(conn: Any, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _sqlite_table_count(conn: Any, table: str) -> int | None:
    if not _sqlite_table_exists(conn, table):
        return None
    row = conn.execute(f"SELECT count(*) FROM [{table}]").fetchone()  # noqa: S608
    return int(row[0]) if row else 0


def _sqlite_max_text(conn: Any, table: str, column: str) -> str:
    if not _sqlite_table_exists(conn, table):
        return ""
    row = conn.execute(f"SELECT max([{column}]) FROM [{table}]").fetchone()  # noqa: S608
    return str(row[0] or "") if row else ""


def _sqlite_vec_dims(conn: Any, table: str) -> int | None:
    if table not in {"vec", "repo_vec"}:
        raise ValueError(f"unknown vector table: {table}")
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if not row or not row[0]:
        return None
    match = re.search(r"embedding\s+FLOAT\[(\d+)\]", str(row[0]))
    return int(match.group(1)) if match else None


def _sqlite_db_health(label: str, path: Path, cfg: Config) -> dict[str, Any]:
    report: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "ok": True,
        "status": "missing",
    }
    if not path.exists():
        return report

    try:
        stat = path.stat()
        report.update(
            {
                "status": "checked",
                "size_bytes": int(stat.st_size),
                "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC)
                .isoformat()
                .replace("+00:00", "Z"),
            }
        )
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            conn.execute("PRAGMA query_only=ON")
            integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
            user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            table_count = int(
                conn.execute(
                    "SELECT count(*) FROM sqlite_master WHERE type IN ('table', 'view')"
                ).fetchone()[0]
            )
            report.update(
                {
                    "integrity_check": integrity,
                    "user_version": user_version,
                    "table_count": table_count,
                }
            )
            if label == "memvec":
                vec_dims = _sqlite_vec_dims(conn, "vec")
                repo_vec_dims = _sqlite_vec_dims(conn, "repo_vec")
                report.update(
                    {
                        "records": _sqlite_table_count(conn, "meta"),
                        "repo_sources": _sqlite_table_count(conn, "repo_sources"),
                        "repo_chunks": _sqlite_table_count(conn, "repo_chunks"),
                        "vec_dims": vec_dims,
                        "repo_vec_dims": repo_vec_dims,
                        "expected_dims": cfg.embedder_dims,
                        "latest_memory_update": _sqlite_max_text(conn, "meta", "updated"),
                        "latest_repo_index": _sqlite_max_text(conn, "repo_sources", "indexed_at"),
                    }
                )
                if vec_dims is not None and vec_dims != cfg.embedder_dims:
                    report["ok"] = False
                    report["status"] = "dimension_mismatch"
                if repo_vec_dims is not None and repo_vec_dims != cfg.embedder_dims:
                    report["ok"] = False
                    report["status"] = "dimension_mismatch"
            elif label == "history":
                report.update(
                    {
                        "events": _sqlite_table_count(conn, "events"),
                        "latest_event": _sqlite_max_text(conn, "events", "ts"),
                    }
                )
            elif label == "graph":
                report.update(
                    {
                        "entities": _sqlite_table_count(conn, "entities"),
                        "links": _sqlite_table_count(conn, "entity_memory"),
                        "latest_seen": _sqlite_max_text(conn, "entities", "last_seen"),
                    }
                )
            if integrity.lower() != "ok":
                report["ok"] = False
                report["status"] = "integrity_failed"
        finally:
            conn.close()
    except Exception as exc:
        report.update({"ok": False, "status": "error", "error": str(exc)})
    return report


def _db_health_report(cfg: Config) -> list[dict[str, Any]]:
    return [_sqlite_db_health(label, path, cfg) for label, path in _managed_sqlite_dbs(cfg)]


_PROFILE_FIELDS = (
    "llm_model",
    "llm_revision",
    "helper_model",
    "helper_revision",
    "embedder_model",
    "embedder_revision",
    "embedder_dims",
    "reranker_enabled",
    "reranker_model",
    "reranker_revision",
)

_PROFILE_ENV_KEYS = (
    "MEMO_MODEL_PROFILE",
    "MEMO_LLM_MODEL",
    "MEMO_LLM_REVISION",
    "MEMO_HELPER_MODEL",
    "MEMO_HELPER_REVISION",
    "MEMO_EMBEDDER_MODEL",
    "MEMO_EMBEDDER_REVISION",
    "MEMO_EMBEDDER_DIMS",
    "MEMO_RERANKER_ENABLED",
    "MEMO_RERANKER_MODEL",
    "MEMO_RERANKER_REVISION",
    "MEMO_RERANK_INPUT_K",
    "MEMO_RERANK_FUSION_ALPHA",
)


def _profile_active_config(cfg: Config) -> dict[str, Any]:
    return {field: getattr(cfg, field) for field in _PROFILE_FIELDS}


def _profile_expected_config(cfg: Config) -> dict[str, Any]:
    raw = MODEL_PROFILES.get(cfg.model_profile, {})
    return {field: raw[field] for field in _PROFILE_FIELDS if field in raw}


def _profile_overrides(cfg: Config) -> list[dict[str, Any]]:
    active = _profile_active_config(cfg)
    expected = _profile_expected_config(cfg)
    overrides: list[dict[str, Any]] = []
    for field, expected_value in expected.items():
        actual_value = active.get(field)
        if actual_value != expected_value:
            overrides.append(
                {
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )
    return overrides


def _model_cache_report(cfg: Config) -> list[dict[str, Any]]:
    from memo.embedder_select import resolve_backend
    from memo.model_pins import hf_hub_cache_dir

    hf_cache = hf_hub_cache_dir()
    if resolve_backend(cfg) == "mlx":
        roles = [
            ("embedder", cfg.embedder_model, cfg.embedder_revision),
            ("llm", cfg.llm_model, cfg.llm_revision),
            ("helper", cfg.helper_model, cfg.helper_revision),
        ]
        if cfg.reranker_enabled:
            roles.append(("reranker", cfg.reranker_model, cfg.reranker_revision))
    else:
        # CPU backend: only the sentence-transformers embedder model is loaded;
        # the MLX llm/helper/reranker never run here.
        roles = [("embedder", cfg.st_embedder_model, cfg.st_embedder_revision)]
    seen: set[tuple[str, str, str | None]] = set()
    out: list[dict[str, Any]] = []
    for role, model, revision in roles:
        key = (role, model, revision)
        if key in seen:
            continue
        seen.add(key)
        cache_dir = hf_cache / f"models--{model.replace('/', '--')}"
        out.append(
            {
                "role": role,
                "model": model,
                "revision": revision,
                "cached": cache_dir.is_dir(),
                "cache_path": str(cache_dir),
            }
        )
    return out


def _typed_embedder_profile(cfg: Config) -> dict[str, Any]:
    """Return Memo's native embedding-space contract."""
    from memo.embedder_select import active_embedder_identity

    return {
        "schema": "memo.embedder_profile.v1",
        "model_id": active_embedder_identity(cfg),
        "dims": int(cfg.embedder_dims),
        "normalization": "l2",
        "max_seq_len": None,
        "quantization": None,
        "provider": "memo",
    }


def _profile_status_report(
    cfg: Config,
    *,
    include_db: bool = True,
    include_env: bool = True,
) -> dict[str, Any]:
    db_report = _db_health_report(cfg) if include_db else []
    memvec = next((item for item in db_report if item.get("label") == "memvec"), {})
    db_dims = {
        "vec_dims": memvec.get("vec_dims"),
        "repo_vec_dims": memvec.get("repo_vec_dims"),
        "expected_dims": cfg.embedder_dims,
        "status": memvec.get("status") or ("not_checked" if not include_db else "missing"),
    }
    dimension_mismatch = bool(
        include_db
        and memvec
        and memvec.get("exists")
        and (
            (memvec.get("vec_dims") is not None and memvec.get("vec_dims") != cfg.embedder_dims)
            or (
                memvec.get("repo_vec_dims") is not None
                and memvec.get("repo_vec_dims") != cfg.embedder_dims
            )
        )
    )
    env = {
        key: os.environ[key]
        for key in _PROFILE_ENV_KEYS
        if include_env and os.environ.get(key) not in (None, "")
    }
    status = "dimension_mismatch" if dimension_mismatch else "ok"
    report: dict[str, Any] = {
        "schema": "memo.profile_status.v1",
        "ok": not dimension_mismatch,
        "status": status,
        "profile": cfg.model_profile,
        "known_profiles": sorted(MODEL_PROFILES),
        "active": _profile_active_config(cfg),
        "expected": _profile_expected_config(cfg),
        "overrides": _profile_overrides(cfg),
        "environment": env,
        "models": _model_cache_report(cfg),
        "paths": {
            "data_dir": str(cfg.data_dir),
            "state_dir": str(cfg.state_dir),
            "db_path": str(cfg.db_path),
        },
        "db": db_dims,
    }
    typed = _typed_embedder_profile(cfg)
    if typed is not None:
        report["typed_profile"] = typed
    return report


def _profile_repair_plan(cfg: Config, *, include_db: bool = True) -> dict[str, Any]:
    status = _profile_status_report(cfg, include_db=include_db)
    actions: list[dict[str, Any]] = []
    raw_db = status.get("db")
    db: dict[str, Any] = raw_db if isinstance(raw_db, dict) else {}
    vec_dims = db.get("vec_dims")
    repo_vec_dims = db.get("repo_vec_dims")
    expected_dims = db.get("expected_dims")
    if vec_dims is not None and vec_dims != expected_dims:
        actions.append(
            {
                "severity": "high",
                "kind": "memory_index_rebuild",
                "reason": f"memvec vec table is FLOAT[{vec_dims}] but active config expects FLOAT[{expected_dims}]",
                "commands": [
                    "memo backup --out memo-pre-profile-repair.zip",
                    f"rm {shlex.quote(str(cfg.db_path))}",
                    "memo reindex",
                ],
                "destructive": True,
                "review_required": True,
            }
        )
    if repo_vec_dims is not None and repo_vec_dims != expected_dims:
        actions.append(
            {
                "severity": "high",
                "kind": "repo_index_rebuild",
                "reason": (
                    f"memvec repo_vec table is FLOAT[{repo_vec_dims}] but active config "
                    f"expects FLOAT[{expected_dims}]"
                ),
                "commands": [
                    "memo backup --out memo-pre-profile-repair.zip",
                    f"rm {shlex.quote(str(cfg.db_path))}",
                    "memo reindex",
                    "memo repo index <repo> --force",
                ],
                "destructive": True,
                "review_required": True,
            }
        )
    if db.get("status") == "missing":
        actions.append(
            {
                "severity": "medium",
                "kind": "memory_index_create",
                "reason": "memvec.db is missing; semantic search will need a rebuild",
                "commands": ["memo reindex"],
                "destructive": False,
                "review_required": False,
            }
        )
    if status.get("overrides"):
        actions.append(
            {
                "severity": "info",
                "kind": "profile_override_review",
                "reason": "active MEMO_* values override the named model profile",
                "commands": [
                    "memo profile status --json",
                    "memo install-slash --client all",
                ],
                "destructive": False,
                "review_required": False,
            }
        )
    if not actions:
        actions.append(
            {
                "severity": "info",
                "kind": "no_repair_required",
                "reason": "active model profile, configured dims, and checked DB dims are aligned",
                "commands": [],
                "destructive": False,
                "review_required": False,
            }
        )
    return {
        "schema": "memo.profile_repair_plan.v1",
        "ok": status.get("ok", True),
        "status": status.get("status", "ok"),
        "profile_status": status,
        "actions": actions,
    }


def _json_import_check(label: str, check: Callable[[], None]) -> dict[str, Any]:
    try:
        check()
    except Exception as exc:
        return {"label": label, "ok": False, "error": str(exc)}
    return {"label": label, "ok": True, "error": ""}


def freshness_report() -> dict[str, str]:
    """Resolve installed version + package dir + ``MEMO_DEV_REPO`` and run the
    install-freshness check. The single wiring shared by both ``memo doctor``
    output paths (human and ``--json``), so they can never drift apart."""
    from memo import __version__ as _installed_version
    from memo.flags import flag_str
    from memo.runtime.freshness import check_install_freshness, installed_package_dir

    _dev_repo = flag_str("MEMO_DEV_REPO")
    return check_install_freshness(
        installed_version=_installed_version,
        installed_pkg_dir=installed_package_dir(),
        repo_root=Path(_dev_repo).expanduser() if _dev_repo else None,
    )


def _doctor_report(
    cfg: Config,
    *,
    check_db: bool,
    strict_runtime: bool,
    do_gc: bool,
    fix: bool,
) -> dict[str, Any]:
    _freshness = freshness_report()

    from memo.runtime.mcp_config import scan_mcp_configs

    _mcp_config_issues = scan_mcp_configs()

    def check_sqlite_vec() -> None:
        # Probe that the extension loads AND supports the PARTITION KEY /
        # metadata-column kNN filtering the store now relies on (alpha vec0
        # API) — a too-old sqlite-vec would load but fail these, so a plain
        # `import + load` check would pass while real queries break.
        from memo.sqlite_compat import import_sqlite_vec

        sqlite_vec = import_sqlite_vec()
        serialize_float32 = sqlite_vec.serialize_float32

        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute(
                "CREATE VIRTUAL TABLE probe USING vec0("
                "id TEXT PRIMARY KEY, part TEXT PARTITION KEY, "
                "emb FLOAT[2] distance_metric=cosine, kind TEXT)"
            )
            blob = serialize_float32([1.0, 0.0])
            conn.execute(
                "INSERT INTO probe(id, part, emb, kind) VALUES (?,?,?,?)", ("a", "p", blob, "n")
            )
            conn.execute(
                "SELECT id FROM probe WHERE emb MATCH ? AND k=1 AND part=? AND kind!=?",
                (blob, "p", "x"),
            ).fetchall()
            conn.execute("DROP TABLE probe")
        finally:
            conn.close()

    def check_mlx() -> None:
        from memo.mlx_gpu import suppress_swig_deprecation_warnings

        suppress_swig_deprecation_warnings()
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401

    def check_sentence_transformers() -> None:
        import sentence_transformers  # noqa: F401

    def check_fts5() -> None:
        # FTS5 is the BM25 backbone. If sqlite was built without it, hybrid
        # search degrades silently to vec-only. Probe with a throwaway table.

        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE probe USING fts5(x)")
            conn.execute("DROP TABLE probe")
        finally:
            conn.close()

    from memo.cli_runtime import _runtime_install_report

    runtime = _runtime_install_report()
    data_dir = {"path": str(cfg.data_dir), "ok": cfg.data_dir.is_dir()}
    vault_path = {
        "path": str(cfg.vault_path) if cfg.vault_path else "",
        "ok": True if cfg.vault_path is None else cfg.vault_path.is_dir(),
        "set": cfg.vault_path is not None,
    }
    # Backend-aware: on the CPU (sentence-transformers) backend MLX is expected
    # to be absent, so probing it must not fail an otherwise-healthy Linux install.
    from memo.embedder_select import resolve_backend

    imports = [
        _json_import_check("sqlite_vec", check_sqlite_vec),
        _json_import_check("sqlite_fts5", check_fts5),
    ]
    if resolve_backend(cfg) == "mlx":
        imports.append(_json_import_check("mlx", check_mlx))
    else:
        imports.append(_json_import_check("sentence_transformers", check_sentence_transformers))
    daemon = _recall_daemon_health(cfg)
    db_report = _db_health_report(cfg) if check_db else []
    trust_report: dict[str, Any] | None = None
    if check_db:
        from memo.trust_preflight import trust_preflight

        trust_report = trust_preflight(cfg)
    gc_report: dict[str, Any] | None = None
    if do_gc:
        gc_report = _gc_report(cfg, fix=fix)
    ok = (
        data_dir["ok"]
        and vault_path["ok"]
        and all(item["ok"] for item in imports)
        and (not strict_runtime or not runtime["warnings"])
        and all(item.get("ok", True) for item in db_report)
        and (trust_report is None or bool(trust_report["ok"]))
    )
    return {
        "schema": "memo.doctor.v1",
        "ok": ok,
        "runtime": runtime,
        "storage": {"data_dir": data_dir, "vault_path": vault_path},
        "profile": _profile_status_report(cfg, include_db=check_db),
        "imports": imports,
        "models": _model_cache_report(cfg),
        "db": db_report,
        "trust": trust_report,
        "gc": gc_report,
        "recall_daemon": daemon,
        "freshness": _freshness,
        "mcp_config_issues": _mcp_config_issues,
    }


def _recall_daemon_health(cfg: Config) -> dict[str, Any]:
    """Probe the recall daemon: PID alive + socket responding to ping.

    Never raises. Returns a dict the doctor surfaces under `recall_daemon`.
    """
    from memo.recall_server import (
        _is_pid_alive,
        _pid_file,
        _read_pid,
        _socket_path,
    )

    state_dir = cfg.state_dir
    sock = _socket_path(state_dir)
    pid_path = _pid_file(state_dir)
    pid = _read_pid(state_dir)
    sock_exists = sock.exists()

    if pid is None and not sock_exists:
        return {
            "running": False,
            "pid": None,
            "socket": str(sock),
            "pid_file": str(pid_path),
            "socket_exists": False,
            "pid_alive": False,
            "ping_ok": False,
            "note": "not started",
            "error": "",
        }

    alive = pid is not None and _is_pid_alive(pid)
    ping_ok = False
    ping_err = ""
    if sock_exists:
        try:
            from memo import embedder_client

            resp = embedder_client.ping(state_dir=state_dir)
            ping_ok = isinstance(resp, dict) and resp.get("ok") is not False
        except Exception as exc:
            ping_err = f"{type(exc).__name__}: {exc}"

    return {
        "running": bool(alive and ping_ok),
        "pid": pid,
        "pid_alive": alive,
        "socket": str(sock),
        "socket_exists": sock_exists,
        "ping_ok": ping_ok,
        "error": ping_err,
    }
