"""One-off: migrate synapse chat feedback signals from the final backup into memo state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_DEFAULT_BACKUP = (
    Path.home()
    / ".memo-daemon-backups"
    / "20260730T213401-synapse-final"
    / "dot-synapse"
    / "state"
    / "feedback"
)
_VOTE_FIELDS = {"created_at", "question_key", "query", "source_id", "rating", "query_embedding"}
_EVENT_FIELDS = {
    "feedback_id",
    "created_at",
    "chat_session_id",
    "turn_id",
    "query",
    "answer",
    "source_ids",
    "rating",
    "correction_text",
}


def _migrate_file(src: Path, dst: Path, fields: set[str], schema: str) -> tuple[int, int]:
    if not src.exists():
        return 0, 0
    existing: set[str] = set()
    if dst.exists():
        existing = set(dst.read_text(encoding="utf-8").splitlines())
    migrated, skipped = 0, 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a", encoding="utf-8") as out:
        for line in src.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(record, dict):
                skipped += 1
                continue
            clean = {k: v for k, v in record.items() if k in fields}
            clean["schema"] = schema
            serialized = json.dumps(clean, ensure_ascii=False)
            if serialized in existing:
                continue
            out.write(serialized + "\n")
            existing.add(serialized)
            migrated += 1
    return migrated, skipped


def migrate_feedback(src_dir: Path, dst_dir: Path) -> dict[str, int]:
    votes, vote_skipped = _migrate_file(
        src_dir / "source_votes.jsonl",
        dst_dir / "source_votes.jsonl",
        _VOTE_FIELDS,
        "memo.chat.source_vote.v1",
    )
    events, event_skipped = _migrate_file(
        src_dir / "events.jsonl",
        dst_dir / "events.jsonl",
        _EVENT_FIELDS,
        "memo.chat.feedback.v1",
    )
    return {"events": events, "votes": votes, "skipped": vote_skipped + event_skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, default=_DEFAULT_BACKUP)
    parser.add_argument("--state", type=Path, default=Path.home() / ".local" / "share" / "memo")
    args = parser.parse_args()
    stats = migrate_feedback(args.backup, args.state / "chat" / "feedback")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
