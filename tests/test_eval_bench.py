"""Public-benchmark adapter — parsers, isolation, ingestion, scoring, judge.

All tests are network-free and MLX-free: fetch is exercised via an injected
fetcher, ingestion via a stubbed 4-dim embedder, grading via fake judges.
"""

from __future__ import annotations

import hashlib  # noqa: F401  # used by sibling append-tasks (I4/I7)
import json  # noqa: F401  # used by sibling append-tasks (I3/I4/I6/I7)
from pathlib import Path  # noqa: F401  # used by sibling append-tasks (I3/I7)
from types import SimpleNamespace  # noqa: F401  # used by sibling append-task (I6)

import pytest

from memo import eval_bench
from memo.config import Config  # noqa: F401  # used by sibling append-tasks (I3/I4)
from memo.errors import MemoError

# --- tiny dataset fixtures ---------------------------------------------------

LOCOMO_SAMPLE = {
    "sample_id": "conv-1",
    "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "I adopted a dog named Rex."},
            {"speaker": "Melanie", "dia_id": "D1:2", "text": "Congrats on the new dog!"},
        ],
        "session_2_date_time": "10:00 am on 25 May, 2023",
        "session_2": [
            {"speaker": "Caroline", "dia_id": "D2:1", "text": "Rex learned to fetch the ball."},
        ],
    },
    "qa": [
        {
            "question": "What is the name of Caroline's dog?",
            "answer": "Rex",
            "evidence": ["D1:1"],
            "category": 4,
        },
        {
            "question": "What color is Caroline's cat?",
            "adversarial_answer": "No information available",
            "evidence": [],
            "category": 5,
        },
    ],
}

LME_INSTANCE = {
    "question_id": "q-001",
    "question_type": "knowledge-update",
    "question": "Where does the user work now?",
    "answer": "Acme",
    "question_date": "2023/05/30 (Tue) 10:00",
    "haystack_session_ids": ["noise_sess", "answer_sess"],
    "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 09:00"],
    "haystack_sessions": [
        [
            {"role": "user", "content": "I like turtles."},
            {"role": "assistant", "content": "Noted."},
        ],
        [
            {"role": "user", "content": "Quick update on my job."},
            {"role": "user", "content": "I started working at Acme.", "has_answer": True},
        ],
    ],
    "answer_session_ids": ["answer_sess"],
}

LME_ABS_INSTANCE = {
    **LME_INSTANCE,
    "question_id": "q-002_abs",
    "question_type": "single-session-user",
    "haystack_session_ids": ["s1"],
    "haystack_dates": ["2023/05/20 (Sat) 02:21"],
    "haystack_sessions": [[{"role": "user", "content": "hello there"}]],
    "answer_session_ids": [],
}

# --- date parsing --------------------------------------------------------------


def test_parse_bench_date_formats():
    assert eval_bench._parse_bench_date("1:56 pm on 8 May, 2023") == "2023-05-08T13:56:00"
    assert eval_bench._parse_bench_date("2023/05/20 (Sat) 02:21") == "2023-05-20T02:21:00"
    assert eval_bench._parse_bench_date("total garbage") is None
    assert eval_bench._parse_bench_date(None) is None

# --- parsers -------------------------------------------------------------------


def test_parse_locomo_normalizes_turns_and_qa():
    samples = eval_bench.parse_locomo([LOCOMO_SAMPLE])
    assert len(samples) == 1
    s = samples[0]
    assert s.sample_id == "conv-1"
    assert [t.turn_id for t in s.turns] == ["D1:1", "D1:2", "D2:1"]
    assert s.turns[0].session_id == "session_1"
    assert s.turns[0].role == "Caroline"
    assert s.turns[2].date == "2023-05-25T10:00:00"
    single, adv = s.qa
    assert single.category == "single_hop"
    assert single.evidence_turn_ids == ("D1:1",)
    assert single.answer == "Rex"
    assert not single.abstention
    assert adv.abstention
    assert adv.category == "adversarial"


def test_parse_longmemeval_categories_evidence_abstention():
    samples = eval_bench.parse_longmemeval([LME_INSTANCE, LME_ABS_INSTANCE])
    s = samples[0]
    assert s.sample_id == "q-001"
    assert len(s.qa) == 1
    qa = s.qa[0]
    assert qa.category == "knowledge_update"
    assert qa.evidence_session_ids == ("answer_sess",)
    assert qa.evidence_turn_ids == ("answer_sess:1",)  # the has_answer turn
    assert not qa.abstention
    assert {t.session_id for t in s.turns} == {"noise_sess", "answer_sess"}
    assert s.turns[0].date == "2023-05-20T02:21:00"
    assert samples[1].qa[0].abstention  # question_id ends with _abs


def test_parse_dataset_dispatch():
    assert eval_bench.parse_dataset("locomo", [LOCOMO_SAMPLE])[0].sample_id == "conv-1"
    assert eval_bench.parse_dataset("longmemeval_oracle", [LME_INSTANCE])[0].sample_id == "q-001"
    with pytest.raises(MemoError):
        eval_bench.parse_dataset("nope", [])
