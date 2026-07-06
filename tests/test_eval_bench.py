"""Public-benchmark adapter — parsers, isolation, ingestion, scoring, judge.

All tests are network-free and MLX-free: fetch is exercised via an injected
fetcher, ingestion via a stubbed 4-dim embedder, grading via fake judges.
"""

from __future__ import annotations

import hashlib  # used by sibling append-tasks (I4/I7)
import json  # used by sibling append-tasks (I3/I4/I6/I7)
from pathlib import Path  # noqa: F401  # used by sibling append-tasks (I3/I7)
from types import SimpleNamespace  # noqa: F401  # used by sibling append-task (I6)

import pytest

from memo import eval_bench
from memo.config import Config  # used by sibling append-tasks (I3/I4)
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


# --- fetch/cache -----------------------------------------------------------------


def test_fetch_dataset_caches_and_reuses(tmp_path):
    calls: list[str] = []

    def fake_fetch(url: str) -> bytes:
        calls.append(url)
        return b'{"ok": true}'

    p1 = eval_bench.fetch_dataset("locomo", tmp_path, fetcher=fake_fetch)
    p2 = eval_bench.fetch_dataset("locomo", tmp_path, fetcher=fake_fetch)
    assert p1 == p2 == tmp_path / "locomo.json"
    assert calls == [eval_bench.DATASET_URLS["locomo"]]  # 2nd call served from cache


def test_fetch_dataset_rejects_non_json_payload(tmp_path):
    with pytest.raises(json.JSONDecodeError):
        eval_bench.fetch_dataset("locomo", tmp_path, fetcher=lambda url: b"<html>err</html>")
    assert not (tmp_path / "locomo.json").exists()  # nothing cached on failure


def test_fetch_dataset_unknown_name_and_url_override(tmp_path):
    with pytest.raises(MemoError):
        eval_bench.fetch_dataset("nope", tmp_path, fetcher=lambda url: b"{}")
    seen: list[str] = []
    eval_bench.fetch_dataset(
        "custom", tmp_path, url="https://example.com/d.json",
        fetcher=lambda url: (seen.append(url), b"[]")[1],
    )
    assert seen == ["https://example.com/d.json"]

# --- isolated store config --------------------------------------------------------


def test_bench_store_config_is_isolated(tmp_path, tmp_cfg):
    root = tmp_path / "bench-root"
    bcfg = eval_bench.bench_store_config(root, tmp_cfg)
    assert str(bcfg.data_dir).startswith(str(root))
    assert str(bcfg.state_dir).startswith(str(root))
    assert bcfg.data_dir != tmp_cfg.data_dir
    assert bcfg.state_dir != tmp_cfg.state_dir
    # model settings are inherited from the live config
    assert bcfg.embedder_dims == tmp_cfg.embedder_dims
    assert bcfg.embedder_model == tmp_cfg.embedder_model
    assert bcfg.llm_model == tmp_cfg.llm_model
    assert bcfg.reranker_enabled == tmp_cfg.reranker_enabled
    assert bcfg.data_dir.is_dir() and bcfg.state_dir.is_dir()


# --- ingestion ---------------------------------------------------------------------


@pytest.fixture
def bench_mem(tmp_path, monkeypatch):
    """Isolated bench store with a deterministic 4-dim stub embedder.

    Dims are pinned to the stub's output (embedder_dims=4 on BOTH configs),
    per the MLX-invariants house rule for stubbed embedders."""

    def _stub_embed(self, inputs):
        out = []
        for s in inputs:
            h = hashlib.sha256((s or "").encode("utf-8")).digest()
            v = [((h[j] / 255.0) * 2.0) - 1.0 for j in range(4)]
            n = sum(x * x for x in v) ** 0.5
            out.append([x / n for x in v])
        return out

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    live_data = tmp_path / "live-data"
    live_state = tmp_path / "live-state"
    live_data.mkdir()
    live_state.mkdir()
    live = Config(
        data_dir=live_data, state_dir=live_state, embedder_dims=4, reranker_enabled=False
    )
    root = tmp_path / "bench" / "locomo" / "conv-1"
    bcfg = eval_bench.bench_store_config(root, live)
    from memo.memory import Memory

    mem = Memory(bcfg)
    yield mem, root, live
    mem.close()


def test_ingest_sample_isolated_backdated_idempotent(bench_mem):
    mem, root, live = bench_mem
    sample = eval_bench.parse_locomo([LOCOMO_SAMPLE])[0]
    res = eval_bench.ingest_sample(mem, sample, root)

    assert set(res.turn_to_memory) == {"D1:1", "D1:2", "D2:1"}
    assert res.turn_to_session["D2:1"] == "session_2"
    # provenance + back-dating on the saved record
    rec = mem.get(res.turn_to_memory["D1:1"])
    assert rec is not None
    assert rec.extra["bench_turn_id"] == "D1:1"
    assert rec.extra["bench_session_id"] == "session_1"
    assert rec.created.startswith("2023-05-08")
    # ISOLATION: memories landed under the bench root, live data_dir untouched
    assert (root / "manifest.json").exists()
    assert list(live.data_dir.rglob("*.md")) == []
    md_before = len(list((root / "data").rglob("*.md")))
    assert md_before == 3
    # IDEMPOTENT: second call reuses the manifest, writes nothing new
    res2 = eval_bench.ingest_sample(mem, sample, root)
    assert res2.turn_to_memory == res.turn_to_memory
    assert len(list((root / "data").rglob("*.md"))) == md_before


def test_load_manifest_missing_or_corrupt(tmp_path):
    assert eval_bench.load_manifest(tmp_path) is None
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    assert eval_bench.load_manifest(tmp_path) is None
