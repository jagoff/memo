"""Cold-start importer parsers — Codex rollouts, opencode SQLite,
ChatGPT / Claude.ai export JSON. Fixtures mirror the real on-disk formats
(verified against ~/.codex/sessions and ~/.local/share/opencode/opencode.db,
2026-07).

Also covers two robustness properties:
- per-record isolation: a malformed record must not abort the whole import
- source provenance: mine_exchange_stream stamps extra["source"] on saved memories
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from memo.history_importers import (
    iter_chatgpt_exchanges,
    iter_claude_export_exchanges,
    iter_codex_exchanges,
    iter_opencode_exchanges,
)


def test_iter_codex_exchanges_pairs_and_skips_injected_blocks(tmp_path: Path):
    f = tmp_path / "rollout-2026-05-20T18-33-57-abc.jsonl"
    lines = [
        {"timestamp": "t0", "type": "session_meta", "payload": {"id": "x"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "# AGENTS.md instructions for /repo\n<INSTRUCTIONS>...",
                    }
                ],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "<environment_context>cwd=/repo</environment_context>",
                    }
                ],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "arma el roadmap de features"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Voy a revisar la memoria primero"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Roadmap: 1) vec layer"}],
            },
        },
    ]
    f.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")

    pairs = list(iter_codex_exchanges(f))

    assert pairs == [
        ("arma el roadmap de features", "Voy a revisar la memoria primero\n\nRoadmap: 1) vec layer")
    ]


def _make_opencode_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE message (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
            data TEXT NOT NULL
        );
        CREATE TABLE part (
            id TEXT PRIMARY KEY, message_id TEXT NOT NULL, session_id TEXT NOT NULL,
            time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
            data TEXT NOT NULL
        );
        """
    )
    conn.executemany(
        "INSERT INTO message VALUES (?,?,?,?,?)",
        [
            ("m1", "s1", 1, 1, json.dumps({"role": "user"})),
            ("m2", "s1", 2, 2, json.dumps({"role": "assistant"})),
            ("m3", "s2", 3, 3, json.dumps({"role": "user"})),
            ("m4", "s2", 4, 4, json.dumps({"role": "assistant"})),
        ],
    )
    conn.executemany(
        "INSERT INTO part VALUES (?,?,?,?,?,?)",
        [
            (
                "p1",
                "m1",
                "s1",
                1,
                1,
                json.dumps({"type": "text", "text": "arregla el bug de sync"}),
            ),
            ("p2", "m2", "s1", 2, 2, json.dumps({"type": "text", "text": "el bug era el flock"})),
            ("p2b", "m2", "s1", 2, 2, json.dumps({"type": "reasoning", "text": "hidden thinking"})),
            ("p3", "m3", "s2", 3, 3, json.dumps({"type": "text", "text": "otro tema"})),
            ("p4", "m4", "s2", 4, 4, json.dumps({"type": "text", "text": "respuesta dos"})),
        ],
    )
    conn.commit()
    conn.close()


def test_iter_opencode_exchanges_joins_parts_per_session(tmp_path: Path):
    db = tmp_path / "opencode.db"
    _make_opencode_db(db)

    pairs = list(iter_opencode_exchanges(db))

    assert pairs == [
        ("arregla el bug de sync", "el bug era el flock"),
        ("otro tema", "respuesta dos"),
    ]
    assert all("hidden thinking" not in a for _, a in pairs)


def test_iter_opencode_exchanges_missing_db_yields_nothing(tmp_path: Path):
    assert list(iter_opencode_exchanges(tmp_path / "nope.db")) == []


def test_iter_chatgpt_exchanges_orders_mapping_by_create_time(tmp_path: Path):
    export = tmp_path / "conversations.json"
    convo = {
        "title": "envs",
        "mapping": {
            "n0": {"message": None},
            "n2": {
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["usa uv, un venv por repo"]},
                    "create_time": 101.0,
                }
            },
            "n1": {
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["como manejo venvs?"]},
                    "create_time": 100.0,
                }
            },
            "n3": {
                "message": {
                    "author": {"role": "system"},
                    "content": {"content_type": "text", "parts": ["sys"]},
                    "create_time": 99.0,
                }
            },
        },
    }
    export.write_text(json.dumps([convo]), encoding="utf-8")

    pairs = list(iter_chatgpt_exchanges(export))

    assert pairs == [("como manejo venvs?", "usa uv, un venv por repo")]


def test_iter_claude_export_handles_text_and_block_formats(tmp_path: Path):
    export = tmp_path / "conversations.json"
    data = [
        {
            "uuid": "c1",
            "name": "memo debug",
            "chat_messages": [
                {"sender": "human", "text": "por que falla el recall hook?"},
                {"sender": "assistant", "text": "el daemon no estaba corriendo"},
            ],
        },
        {
            "uuid": "c2",
            "name": "blocks",
            "chat_messages": [
                {"sender": "human", "content": [{"type": "text", "text": "segunda charla"}]},
                {
                    "sender": "assistant",
                    "content": [{"type": "text", "text": "respuesta en bloques"}],
                },
            ],
        },
    ]
    export.write_text(json.dumps(data), encoding="utf-8")

    pairs = list(iter_claude_export_exchanges(export))

    assert pairs == [
        ("por que falla el recall hook?", "el daemon no estaba corriendo"),
        ("segunda charla", "respuesta en bloques"),
    ]


# ---------------------------------------------------------------------------
# Fix 1 — per-record robustness
# ---------------------------------------------------------------------------


def test_iter_opencode_malformed_msg_data_skipped(tmp_path: Path):
    """A msg_data row that decodes to a non-dict (e.g. JSON integer 42) must
    NOT abort the import.  The original except (ValueError, TypeError) does NOT
    catch the resulting AttributeError from `42.get("role")`, so before the fix
    this test raises and fails; after the fix the valid exchange is returned."""
    db = tmp_path / "opencode.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE message (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
            data TEXT NOT NULL
        );
        CREATE TABLE part (
            id TEXT PRIMARY KEY, message_id TEXT NOT NULL, session_id TEXT NOT NULL,
            time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL,
            data TEXT NOT NULL
        );
        """
    )
    conn.executemany(
        "INSERT INTO message VALUES (?,?,?,?,?)",
        [
            ("m1", "s1", 1, 1, json.dumps({"role": "user"})),
            # malformed: valid JSON but not a dict — json.loads("42") == 42,
            # then 42.get("role") raises AttributeError (not caught pre-fix)
            ("m_bad", "s1", 2, 2, "42"),
            ("m2", "s1", 3, 3, json.dumps({"role": "assistant"})),
        ],
    )
    conn.executemany(
        "INSERT INTO part VALUES (?,?,?,?,?,?)",
        [
            ("p1", "m1", "s1", 1, 1, json.dumps({"type": "text", "text": "arregla el bug"})),
            ("p_bad", "m_bad", "s1", 2, 2, json.dumps({"type": "text", "text": "no importa"})),
            ("p2", "m2", "s1", 3, 3, json.dumps({"type": "text", "text": "ya lo arregle"})),
        ],
    )
    conn.commit()
    conn.close()

    # Must not raise; the valid exchange must survive.
    pairs = list(iter_opencode_exchanges(db))
    assert pairs == [("arregla el bug", "ya lo arregle")]


# ---------------------------------------------------------------------------
# Fix 2 — source provenance
# ---------------------------------------------------------------------------


def _stub_embed_4dim(self, inputs):  # type: ignore[override]
    """Deterministic 4-dim stub (same shape as conftest.mem_with_stub)."""
    out = []
    for s in inputs:
        h = sum(ord(c) for c in (s or "")) % 4
        v = [0.0] * 4
        v[h] = 1.0
        out.append(v)
    return out


def test_mine_exchange_stream_stamps_source_provenance(tmp_path: Path, monkeypatch):
    """mine_exchange_stream stamps extra['source'] = 'imported:<name>' on every
    saved memory when source_name is set and no extra_fn is provided.
    Before Fix 2, extra is None and rec.extra has no 'source' key."""
    from memo.config import Config
    from memo.memory import Memory
    from memo.transcript_miner import mine_exchange_stream

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed_4dim)
    monkeypatch.setattr("memo.embedder.MLXEmbedder.__init__", lambda self, **kw: None)

    # Bypass the LLM: return one candidate directly.
    monkeypatch.setattr(
        "memo.transcript_miner.extract_insights",
        lambda *a, **kw: [
            {
                "title": "fixed bug with broad except in opencode importer",
                "type": "bug",
                "body": (
                    "The import was aborting on malformed records because AttributeError "
                    "was not caught. Fixed by using broad except Exception so that one "
                    "bad record skips and valid records are still processed and saved."
                ),
                "tags": ["import", "bug"],
            }
        ],
    )
    # No duplicates in an empty store — make it explicit.
    monkeypatch.setattr("memo.transcript_miner.is_near_duplicate", lambda *a, **kw: False)

    data_dir = tmp_path / "data"
    vault = tmp_path / "vault"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    (vault / "Obsidian" / "AI" / "memory").mkdir(parents=True)
    state_dir.mkdir()

    cfg = Config(
        data_dir=data_dir,
        vault_path=vault,
        state_dir=state_dir,
        embedder_dims=4,
        reranker_enabled=False,
    )
    mem = Memory(cfg)
    chat = mem._ensure_chat()

    # The assistant text must pass _passes_prefilter (>=200 chars + trigger keyword).
    user_text = "how do we handle malformed records in the opencode importer?"
    assist_text = (
        "The fix is to wrap the per-record body in a broad except Exception handler "
        "so a single malformed record does not abort the entire import run. "
        "The bug was AttributeError from calling .get() on a non-dict JSON value. "
        "After the fix all valid records are processed and malformed ones are silently skipped."
    )

    result = mine_exchange_stream(
        mem,
        chat,
        cfg,
        iter([(user_text, assist_text)]),
        turn_hashes=set(),
        source_name="opencode.db",
    )
    mem.close()

    assert result["saved"], "expected at least one saved memory"

    # Reopen to read back the persisted extra field.
    mem2 = Memory(cfg)
    rec = mem2.get(result["saved"][0])
    mem2.close()

    assert rec is not None, "saved memory must be retrievable"
    assert rec.extra.get("source") == "imported:opencode.db"
