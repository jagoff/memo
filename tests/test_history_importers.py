"""Cold-start importer parsers — Codex rollouts, opencode SQLite,
ChatGPT / Claude.ai export JSON. Fixtures mirror the real on-disk formats
(verified against ~/.codex/sessions and ~/.local/share/opencode/opencode.db,
2026-07)."""

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
        {"type": "response_item", "payload": {"type": "message", "role": "user",
         "content": [{"type": "input_text",
                      "text": "# AGENTS.md instructions for /repo\n<INSTRUCTIONS>..."}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
         "content": [{"type": "input_text",
                      "text": "<environment_context>cwd=/repo</environment_context>"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "arma el roadmap de features"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "Voy a revisar la memoria primero"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "Roadmap: 1) vec layer"}]}},
    ]
    f.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")

    pairs = list(iter_codex_exchanges(f))

    assert pairs == [
        ("arma el roadmap de features",
         "Voy a revisar la memoria primero\n\nRoadmap: 1) vec layer")
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
            ("p1", "m1", "s1", 1, 1,
             json.dumps({"type": "text", "text": "arregla el bug de sync"})),
            ("p2", "m2", "s1", 2, 2,
             json.dumps({"type": "text", "text": "el bug era el flock"})),
            ("p2b", "m2", "s1", 2, 2,
             json.dumps({"type": "reasoning", "text": "hidden thinking"})),
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
            "n2": {"message": {"author": {"role": "assistant"},
                   "content": {"content_type": "text",
                               "parts": ["usa uv, un venv por repo"]},
                   "create_time": 101.0}},
            "n1": {"message": {"author": {"role": "user"},
                   "content": {"content_type": "text", "parts": ["como manejo venvs?"]},
                   "create_time": 100.0}},
            "n3": {"message": {"author": {"role": "system"},
                   "content": {"content_type": "text", "parts": ["sys"]},
                   "create_time": 99.0}},
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
                {"sender": "human",
                 "content": [{"type": "text", "text": "segunda charla"}]},
                {"sender": "assistant",
                 "content": [{"type": "text", "text": "respuesta en bloques"}]},
            ],
        },
    ]
    export.write_text(json.dumps(data), encoding="utf-8")

    pairs = list(iter_claude_export_exchanges(export))

    assert pairs == [
        ("por que falla el recall hook?", "el daemon no estaba corriendo"),
        ("segunda charla", "respuesta en bloques"),
    ]
