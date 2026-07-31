from pathlib import Path

from memo.chat import feedback
from memo.chat.dedup import SCORE_FIELDS
from memo.chat.feedback import (
    ChatFeedback,
    FeedbackStore,
    SourceVote,
    SourceVoteStore,
    boost_positive_sources,
    boost_semantic,
    filter_negative_sources,
    question_key,
)


def test_boost_field_precedence_matches_dedup_score_fields() -> None:
    # _boost_field must stay derived from dedup.SCORE_FIELDS — a second,
    # independently-maintained field-precedence tuple drifting out of sync
    # would boost the wrong field silently.
    for field in SCORE_FIELDS:
        assert feedback._boost_field({field: 1.0}) == field


def _vote(qk: str, sid: str, rating: str, emb: list[float] | None = None) -> SourceVote:
    return SourceVote(
        created_at="2026-07-30T00:00:00",
        question_key=qk,
        query="q",
        source_id=sid,
        rating=rating,
        query_embedding=emb or [],
    )


def _src(sid: str, score: float) -> dict:
    return {
        "source": "memory",
        "id": sid,
        "title": sid,
        "score": score,
        "normalized_score": score,
        "snippet": "x",
    }


def test_question_key_stable() -> None:
    assert question_key("  Hola Mundo ") == question_key("hola mundo")
    assert len(question_key("x")) == 16


def test_store_roundtrip_and_latest_wins(tmp_path: Path) -> None:
    store = SourceVoteStore(tmp_path)
    store.record(_vote("k1", "s1", "up"))
    store.record(_vote("k1", "s1", "down"))
    (tmp_path / "source_votes.jsonl").open("a").write("not json\n")
    latest = SourceVoteStore(tmp_path).latest_by_pair()
    assert latest[("k1", "s1")].rating == "down"


def test_filter_negative_and_boost_positive() -> None:
    latest = {("k", "bad"): _vote("k", "bad", "down"), ("k", "good"): _vote("k", "good", "up")}
    sources = [_src("bad", 0.9), _src("good", 0.4), _src("meh", 0.5)]
    kept = filter_negative_sources(sources, latest, "k")
    assert [s["id"] for s in kept] == ["good", "meh"]
    boosted = boost_positive_sources(kept, latest, "k", factor=1.5)
    assert boosted[0]["id"] == "good"  # 0.4*1.5=0.6 > 0.5
    assert boosted[0]["source_vote_boost"] == 1.5


def test_boost_factor_clamped() -> None:
    latest = {("k", "a"): _vote("k", "a", "up")}
    out = boost_positive_sources([_src("a", 1.0)], latest, "k", factor=99.0)
    assert out[0]["source_vote_boost"] == 5.0


def test_semantic_boost_by_cosine() -> None:
    votes = [_vote("otra", "a", "up", emb=[1.0, 0.0]), _vote("otra", "b", "down", emb=[1.0, 0.0])]
    out = boost_semantic(
        [_src("a", 0.4), _src("b", 0.4)], [1.0, 0.0], votes, threshold=0.75, factor=1.5
    )
    by_id = {s["id"]: s for s in out}
    assert by_id["a"].get("source_vote_boost") == 1.5  # up-vote similar
    assert "source_vote_boost" not in by_id["b"]  # down votes no generalizan
    far = boost_semantic([_src("a", 0.4)], [0.0, 1.0], votes, threshold=0.75, factor=1.5)
    assert "source_vote_boost" not in far[0]  # coseno 0 < 0.75


def test_source_vote_skip_shape_corrupt(tmp_path: Path) -> None:
    store = SourceVoteStore(tmp_path)
    store.record(_vote("k1", "s1", "up"))
    # Append shape-corrupt line (valid JSON but missing required field)
    (tmp_path / "source_votes.jsonl").open("a").write(
        '{"created_at":"2026-07-30T00:00:00","question_key":"k1","query":"q","source_id":"s2"}\n'
    )
    store.record(_vote("k1", "s3", "down"))
    # Load should skip the shape-corrupt line and return only 2 valid votes
    votes = SourceVoteStore(tmp_path).load()
    assert len(votes) == 2
    assert votes[0].source_id == "s1"
    assert votes[1].source_id == "s3"


def test_feedback_store_roundtrip_and_corrupt(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path)
    fb1 = ChatFeedback(
        feedback_id="f1",
        created_at="2026-07-30T00:00:00",
        chat_session_id="session1",
        turn_id="turn1",
        query="q1",
        answer="a1",
        source_ids=["s1"],
        rating="good",
    )
    fb2 = ChatFeedback(
        feedback_id="f2",
        created_at="2026-07-30T00:00:00",
        chat_session_id="session2",
        turn_id="turn2",
        query="q2",
        answer="a2",
        source_ids=["s2"],
        rating="bad",
    )
    store.append(fb1)
    store.append(fb2)
    # Append corrupt JSON line
    (tmp_path / "events.jsonl").open("a").write("not json\n")
    # Append shape-corrupt line (valid JSON but missing required field)
    (tmp_path / "events.jsonl").open("a").write(
        '{"feedback_id":"f3","created_at":"2026-07-30T00:00:00","chat_session_id":"s"}\n'
    )
    # Load should skip corrupt lines and return only 2 valid events
    feedbacks = FeedbackStore(tmp_path).load()
    assert len(feedbacks) == 2
    assert feedbacks[0].feedback_id == "f1"
    assert feedbacks[0].schema == "memo.chat.feedback.v1"
    assert feedbacks[1].feedback_id == "f2"
    assert feedbacks[1].schema == "memo.chat.feedback.v1"
