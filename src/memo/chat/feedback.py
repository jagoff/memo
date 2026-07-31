"""Chat feedback: append-only vote stores + retrieval boosts (exact + semantic)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from memo.chat.dedup import SCORE_FIELDS, score_of

_MIN_FACTOR, _MAX_FACTOR = 1.0, 5.0


def question_key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SourceVote:
    created_at: str
    question_key: str
    query: str
    source_id: str
    rating: str  # "up" | "down"
    query_embedding: list[float] = field(default_factory=list)
    schema: str = "memo.chat.source_vote.v1"


@dataclass(frozen=True)
class ChatFeedback:
    feedback_id: str
    created_at: str
    chat_session_id: str
    turn_id: str
    query: str
    answer: str
    source_ids: list[str]
    rating: str
    correction_text: str = ""
    schema: str = "memo.chat.feedback.v1"


class _JsonlStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_dicts(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
        return out


class SourceVoteStore(_JsonlStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root / "source_votes.jsonl")

    def record(self, vote: SourceVote) -> None:
        self._append(asdict(vote))

    def load(self) -> list[SourceVote]:
        fields = {f for f in SourceVote.__dataclass_fields__}
        out: list[SourceVote] = []
        for d in self._load_dicts():
            try:
                out.append(SourceVote(**{k: v for k, v in d.items() if k in fields}))
            except TypeError:
                continue
        return out

    def latest_by_pair(self) -> dict[tuple[str, str], SourceVote]:
        latest: dict[tuple[str, str], SourceVote] = {}
        for vote in self.load():
            latest[(vote.question_key, vote.source_id)] = vote
        return latest


class FeedbackStore(_JsonlStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root / "events.jsonl")

    def append(self, fb: ChatFeedback) -> None:
        self._append(asdict(fb))

    def load(self) -> list[ChatFeedback]:
        fields = {f for f in ChatFeedback.__dataclass_fields__}
        out: list[ChatFeedback] = []
        for d in self._load_dicts():
            try:
                out.append(ChatFeedback(**{k: v for k, v in d.items() if k in fields}))
            except TypeError:
                continue
        return out


def filter_negative_sources(
    sources: list[dict[str, Any]],
    latest: dict[tuple[str, str], SourceVote],
    qkey: str,
) -> list[dict[str, Any]]:
    out = []
    for s in sources:
        vote = latest.get((qkey, str(s.get("id"))))
        if vote is not None and vote.rating == "down":
            continue
        out.append(s)
    return out


def _boost_field(s: dict[str, Any]) -> str:
    for name in SCORE_FIELDS:
        if isinstance(s.get(name), (int, float)):
            return name
    return "score"


def boost_positive_sources(
    sources: list[dict[str, Any]],
    latest: dict[tuple[str, str], SourceVote],
    qkey: str,
    *,
    factor: float,
) -> list[dict[str, Any]]:
    factor = min(max(factor, _MIN_FACTOR), _MAX_FACTOR)
    out = []
    for s in sources:
        vote = latest.get((qkey, str(s.get("id"))))
        if vote is not None and vote.rating == "up":
            s = dict(s)
            name = _boost_field(s)
            s[name] = float(s.get(name) or 0.0) * factor
            s["source_vote_boost"] = factor
        out.append(s)
    out.sort(key=score_of, reverse=True)
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def boost_semantic(
    sources: list[dict[str, Any]],
    query_vec: list[float],
    votes: list[SourceVote],
    *,
    threshold: float,
    factor: float,
) -> list[dict[str, Any]]:
    factor = min(max(factor, _MIN_FACTOR), _MAX_FACTOR)
    up_by_source: dict[str, list[list[float]]] = {}
    for v in votes:
        if v.rating == "up" and v.query_embedding:
            up_by_source.setdefault(v.source_id, []).append(v.query_embedding)
    out = []
    for s in sources:
        if "source_vote_boost" in s:
            out.append(s)
            continue
        embeddings = up_by_source.get(str(s.get("id")), [])
        if any(_cosine(query_vec, e) >= threshold for e in embeddings):
            s = dict(s)
            name = _boost_field(s)
            s[name] = float(s.get(name) or 0.0) * factor
            s["source_vote_boost"] = factor
        out.append(s)
    out.sort(key=score_of, reverse=True)
    return out
