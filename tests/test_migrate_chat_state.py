import json
from pathlib import Path

from scripts.migrate_synapse_chat_state import migrate_feedback


def _write(path: Path, lines: list[dict | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for line in lines:
            fh.write((line if isinstance(line, str) else json.dumps(line)) + "\n")


def test_migrates_and_remaps_schema(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write(
        src / "source_votes.jsonl",
        [
            {
                "created_at": "t",
                "question_key": "k",
                "query": "q",
                "source_id": "s",
                "rating": "up",
                "query_embedding": [0.1],
                "schema": "synapse.source_vote.v1",
                "extra_synapse_field": 1,
            },
            "corrupt line",
        ],
    )
    _write(
        src / "events.jsonl",
        [
            {
                "feedback_id": "f1",
                "created_at": "t",
                "chat_session_id": "s",
                "turn_id": "1",
                "query": "q",
                "answer": "a",
                "source_ids": ["x"],
                "rating": "up",
                "schema": "synapse.chat_feedback.v1",
                "trace_id": "ignored",
            },
        ],
    )
    stats = migrate_feedback(src, dst)
    assert stats == {"events": 1, "votes": 1, "skipped": 1}
    vote = json.loads((dst / "source_votes.jsonl").read_text().splitlines()[0])
    assert vote["schema"] == "memo.chat.source_vote.v1"
    assert "extra_synapse_field" not in vote
    # idempotente: segunda corrida no duplica
    stats2 = migrate_feedback(src, dst)
    assert stats2["events"] == 0 and stats2["votes"] == 0
