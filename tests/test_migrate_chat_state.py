import json
from pathlib import Path

from memo.chat.feedback import (  # type: ignore[import-untyped]
    ChatFeedback,
    FeedbackStore,
    SourceVoteStore,
)
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


def test_round_trip_through_stores(tmp_path: Path) -> None:
    """Validate migrated data can be loaded through real store classes."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write(
        src / "source_votes.jsonl",
        [
            {
                "created_at": "2026-07-31T10:00:00Z",
                "question_key": "abc123",
                "query": "test query",
                "source_id": "src1",
                "rating": "up",
                "query_embedding": [0.1, 0.2, 0.3],
                "schema": "synapse.source_vote.v1",
            },
            {
                "created_at": "2026-07-31T11:00:00Z",
                "question_key": "def456",
                "query": "another query",
                "source_id": "src2",
                "rating": "down",
                # No query_embedding to test default_factory
                "schema": "synapse.source_vote.v1",
            },
        ],
    )
    _write(
        src / "events.jsonl",
        [
            {
                "feedback_id": "fb001",
                "created_at": "2026-07-31T10:00:00Z",
                "chat_session_id": "sess1",
                "turn_id": "1",
                "query": "what is X",
                "answer": "X is Y",
                "source_ids": ["s1", "s2"],
                "rating": "up",
                "schema": "synapse.chat_feedback.v1",
            },
        ],
    )

    # Migrate
    stats = migrate_feedback(src, dst)
    assert stats["events"] == 1
    assert stats["votes"] == 2

    # Load through real stores
    votes = SourceVoteStore(dst).load()
    events = FeedbackStore(dst).load()

    # Validate votes
    assert len(votes) == 2
    assert votes[0].rating == "up"
    assert votes[0].schema == "memo.chat.source_vote.v1"
    assert votes[0].query_embedding == [0.1, 0.2, 0.3]
    assert votes[1].rating == "down"
    assert votes[1].query_embedding == []  # default_factory produced empty list

    # Validate events
    assert len(events) == 1
    assert events[0].feedback_id == "fb001"
    assert events[0].rating == "up"
    assert events[0].schema == "memo.chat.feedback.v1"
    assert isinstance(events[0], ChatFeedback)
