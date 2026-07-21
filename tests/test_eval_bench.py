"""Public-benchmark adapter — parsers, isolation, ingestion, scoring, judge.

All tests are network-free and MLX-free: fetch is exercised via an injected
fetcher, ingestion via a stubbed 4-dim embedder, grading via fake judges.
"""

from __future__ import annotations

import hashlib  # used by sibling append-tasks (I4/I7)
import json  # used by sibling append-tasks (I3/I4/I6/I7)
from pathlib import Path  # noqa: F401  # used by sibling append-tasks (I3/I7)
from types import SimpleNamespace  # used by sibling append-task (I6)

import pytest

from memo import eval_bench
from memo.config import Config  # used by sibling append-tasks (I3/I4)
from memo.errors import MemoError
from memo.eval_recall import Row

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
        "custom",
        tmp_path,
        url="https://example.com/d.json",
        fetcher=lambda url: (seen.append(url), b"[]")[1],
    )
    assert seen == ["https://example.com/d.json"]


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/d.json",
        "file:///tmp/dataset.json",
        "https:///missing-host.json",
        "https://user:secret@example.com/d.json",
    ],
)
def test_fetch_dataset_rejects_non_https_or_credentialed_urls(tmp_path, url):
    with pytest.raises(MemoError, match="HTTPS"):
        eval_bench.fetch_dataset("custom", tmp_path, url=url, fetcher=lambda _url: b"[]")


def test_dataset_redirect_handler_rejects_https_downgrade():
    handler = eval_bench._HTTPSRedirectHandler()

    with pytest.raises(MemoError, match="HTTPS"):
        handler.redirect_request(
            object(), None, 302, "Found", {}, "http://example.com/dataset.json"
        )


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
    live = Config(data_dir=live_data, state_dir=live_state, embedder_dims=4, reranker_enabled=False)
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


# --- retrieval scoring ----------------------------------------------------------------


def _qa(**kw):
    base = dict(
        qa_id="q1",
        question="what dog?",
        answer="Rex",
        category="single_hop",
        abstention=False,
        evidence_session_ids=(),
        evidence_turn_ids=(),
    )
    base.update(kw)
    return eval_bench.BenchQA(**base)


def test_expected_memory_ids_turn_first_session_fallback():
    ingest = eval_bench.IngestResult(
        turn_to_memory={"D1:1": "a" * 32, "D1:2": "b" * 32},
        turn_to_session={"D1:1": "session_1", "D1:2": "session_1"},
    )
    assert eval_bench.expected_memory_ids(_qa(evidence_turn_ids=("D1:1",)), ingest) == ["a" * 32]
    assert eval_bench.expected_memory_ids(
        _qa(evidence_session_ids=("session_1",)), ingest
    ) == sorted(["a" * 32, "b" * 32])
    assert eval_bench.expected_memory_ids(_qa(), ingest) == []


def test_category_label_sets_split_and_abstention_unscored():
    sample = eval_bench.parse_locomo([LOCOMO_SAMPLE])[0]
    ingest = eval_bench.IngestResult(
        turn_to_memory={"D1:1": "a" * 32, "D1:2": "b" * 32, "D2:1": "c" * 32},
        turn_to_session={"D1:1": "session_1", "D1:2": "session_1", "D2:1": "session_2"},
    )
    by_cat = eval_bench.category_label_sets(sample, ingest)
    assert set(by_cat) == {"single_hop", "adversarial"}
    single = by_cat["single_hop"].prompts[0]
    assert single.relevant and single.expect_ids == ["a" * 32]
    adv = by_cat["adversarial"].prompts[0]
    assert not adv.relevant and adv.expect_ids == []  # abstention never scores retrieval


def test_score_retrieval_end_to_end(bench_mem):
    mem, root, _live = bench_mem
    sample = eval_bench.parse_locomo([LOCOMO_SAMPLE])[0]
    ingest = eval_bench.ingest_sample(mem, sample, root)
    rows = eval_bench.score_retrieval(mem, sample, ingest, k=3)
    assert set(rows) == {"single_hop", "adversarial"}
    row, n = rows["single_hop"]
    assert n == 1
    assert isinstance(row, Row)
    # stub embeddings are deterministic-but-arbitrary: assert ranges, not values
    assert 0.0 <= row.recall_at_k <= 1.0
    assert 0.0 <= row.ndcg_at_k <= 1.0
    assert 0.0 <= row.mrr <= 1.0
    _, n_adv = rows["adversarial"]
    assert n_adv == 0


def test_aggregate_retrieval_weighted_mean():
    r1 = Row(config="bench", precision_at_k=0.2, recall_at_k=1.0, ndcg_at_k=1.0, mrr=1.0)
    r2 = Row(config="bench", precision_at_k=0.0, recall_at_k=0.0, ndcg_at_k=0.0, mrr=0.0)
    agg = eval_bench.aggregate_retrieval(
        [{"single_hop": (r1, 1)}, {"single_hop": (r2, 3), "adversarial": (r2, 0)}]
    )
    assert agg["single_hop"]["recall_at_k"] == 0.25  # (1.0*1 + 0.0*3) / 4
    assert agg["single_hop"]["n_questions"] == 4
    assert "adversarial" not in agg  # zero scored prompts contribute nothing


# --- judge + QA grading ------------------------------------------------------------


def test_parse_verdict():
    assert eval_bench._parse_verdict("Yes")
    assert eval_bench._parse_verdict("  yes, it entails the gold answer")
    assert not eval_bench._parse_verdict("No")
    assert not eval_bench._parse_verdict("")


class _RexJudge:
    name = "fake"

    def grade(self, *, question, gold, answer, abstention):
        if abstention:
            return "couldn't find" in answer.lower()
        return gold.lower() in answer.lower()


def test_grade_sample_qa_and_category_rollup():
    sample = eval_bench.parse_locomo([LOCOMO_SAMPLE])[0]
    mem = SimpleNamespace(
        ask=lambda q, k=5: {
            "answer": "The dog is Rex." if "dog" in q else "I couldn't find that in memory."
        }
    )
    results = eval_bench.grade_sample_qa(mem, sample, _RexJudge(), k=3)
    assert [r.correct for r in results] == [True, True]
    acc = eval_bench.qa_accuracy_by_category(results)
    assert acc["single_hop"] == {"accuracy": 1.0, "n_questions": 1}
    assert acc["adversarial"] == {"accuracy": 1.0, "n_questions": 1}


def test_grade_sample_qa_max_qa_cap():
    sample = eval_bench.parse_locomo([LOCOMO_SAMPLE])[0]
    mem = SimpleNamespace(ask=lambda q, k=5: {"answer": "Rex"})
    assert len(eval_bench.grade_sample_qa(mem, sample, _RexJudge(), max_qa=1)) == 1


def test_judge_from_flags_default_is_local_mlx(tmp_cfg, monkeypatch):
    monkeypatch.delenv("MEMO_BENCH_JUDGE", raising=False)
    j = eval_bench.judge_from_flags(tmp_cfg)
    assert isinstance(j, eval_bench.MLXJudge)
    assert j._model == tmp_cfg.llm_model  # empty MEMO_BENCH_JUDGE_MODEL falls back


def test_judge_from_flags_api_env_gated(tmp_cfg, monkeypatch):
    monkeypatch.setenv("MEMO_BENCH_JUDGE", "api")
    with pytest.raises(MemoError):
        eval_bench.judge_from_flags(tmp_cfg)  # url + model missing
    monkeypatch.setenv("MEMO_BENCH_JUDGE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("MEMO_BENCH_JUDGE_MODEL", "judge-1")
    monkeypatch.setenv("MEMO_BENCH_JUDGE_API_KEY_ENV", "TEST_BENCH_KEY")
    monkeypatch.delenv("TEST_BENCH_KEY", raising=False)
    with pytest.raises(MemoError):
        eval_bench.judge_from_flags(tmp_cfg)  # key env var empty
    monkeypatch.setenv("TEST_BENCH_KEY", "sk-test")
    j = eval_bench.judge_from_flags(tmp_cfg)
    assert isinstance(j, eval_bench.APIJudge)


@pytest.mark.parametrize(
    "url",
    [
        "http://api.example.com/v1",
        "ftp://api.example.com/v1",
        "http://192.168.1.10/v1",
        "https:///missing-host",
    ],
)
def test_judge_from_flags_rejects_insecure_remote_urls(tmp_cfg, monkeypatch, url):
    monkeypatch.setenv("MEMO_BENCH_JUDGE", "api")
    monkeypatch.setenv("MEMO_BENCH_JUDGE_URL", url)
    monkeypatch.setenv("MEMO_BENCH_JUDGE_MODEL", "judge-1")
    monkeypatch.setenv("MEMO_BENCH_JUDGE_API_KEY_ENV", "TEST_BENCH_KEY")
    monkeypatch.setenv("TEST_BENCH_KEY", "sk-test")

    with pytest.raises(MemoError, match=r"HTTPS|loopback"):
        eval_bench.judge_from_flags(tmp_cfg)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000/v1", "http://[::1]:8000/v1"])
def test_judge_from_flags_allows_http_loopback(tmp_cfg, monkeypatch, url):
    monkeypatch.setenv("MEMO_BENCH_JUDGE", "api")
    monkeypatch.setenv("MEMO_BENCH_JUDGE_URL", url)
    monkeypatch.setenv("MEMO_BENCH_JUDGE_MODEL", "judge-1")
    monkeypatch.setenv("MEMO_BENCH_JUDGE_API_KEY_ENV", "TEST_BENCH_KEY")
    monkeypatch.setenv("TEST_BENCH_KEY", "sk-test")

    assert isinstance(eval_bench.judge_from_flags(tmp_cfg), eval_bench.APIJudge)


def test_api_judge_posts_openai_shape(monkeypatch):
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"choices": [{"message": {"content": "yes"}}]}'

    class _Opener:
        def open(self, req, timeout=0):
            captured["url"] = req.full_url
            captured["auth"] = req.get_header("Authorization")
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _Resp()

    def fake_build_opener(*handlers):
        captured["handlers"] = handlers
        return _Opener()

    import urllib.request

    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)
    j = eval_bench.APIJudge("https://api.example.com/v1", "judge-1", "sk-test")
    assert j.grade(question="q", gold="Rex", answer="the dog is Rex", abstention=False)
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["model"] == "judge-1"
    assert captured["body"]["temperature"] == 0
    assert any(
        isinstance(handler, eval_bench._NoRedirectHandler) for handler in captured["handlers"]
    )


def test_api_judge_redirect_handler_rejects_credential_forwarding():
    handler = eval_bench._NoRedirectHandler()

    redirected = handler.redirect_request(
        object(), None, 302, "Found", {}, "http://attacker.example/steal"
    )

    assert redirected is None


# --- contradiction scan (bench faithfulness for knowledge-update) ----------------------


def test_scan_bench_contradictions_delegates_and_returns_counters():
    calls = {}

    class _Scanner:
        def scan_corpus(self, **kw):
            calls.update(kw)
            return SimpleNamespace(pairs_examined=7, pairs_inserted=3)

    mem = SimpleNamespace(contradict_scanner=_Scanner())
    examined, inserted = eval_bench.scan_bench_contradictions(mem)

    assert (examined, inserted) == (7, 3)
    # Bench pairs may share a date across sessions → never filter by day gap.
    assert calls["min_days_apart"] == 0
    assert calls["persist"] is True


# --- receipt + report ------------------------------------------------------------------


def _receipt(dataset="locomo", **kw):
    base = {
        "schema": eval_bench.RECEIPT_SCHEMA,
        "ts": "2026-07-03T12:00:00",
        "dataset": dataset,
        "k": 5,
        "n_samples": 1,
        "judge": "mlx",
        "llm_model": "m",
        "embedder_model": "e",
        "retrieval": {
            "single_hop": {
                "recall_at_k": 0.8,
                "ndcg_at_k": 0.7,
                "mrr": 0.6,
                "precision_at_k": 0.3,
                "n_questions": 10,
            }
        },
        "qa": {"single_hop": {"accuracy": 0.5, "n_questions": 10}},
    }
    base.update(kw)
    return base


def test_write_and_load_receipts_roundtrip(tmp_path):
    p = eval_bench.write_receipt(tmp_path, _receipt())
    assert p.parent == tmp_path / "bench" / "runs"
    loaded = eval_bench.load_receipts(tmp_path, last=5)
    assert len(loaded) == 1
    assert loaded[0]["dataset"] == "locomo"
    assert loaded[0]["_file"] == p.name
    # dataset filter + schema filter
    (p.parent / "junk.json").write_text('{"schema": "other"}', encoding="utf-8")
    assert eval_bench.load_receipts(tmp_path, last=5, dataset="longmemeval_s") == []


def test_render_report_stable_keys_across_runs(tmp_path):
    r1 = {**_receipt(), "_file": "locomo-a.json"}
    r2 = {**_receipt(), "_file": "locomo-b.json"}
    r2["retrieval"]["single_hop"] = dict(r2["retrieval"]["single_hop"], recall_at_k=0.9)
    md = eval_bench.render_report([r2, r1])  # newest-first in, newest last column out
    assert "| retrieval/single_hop/recall_at_k | 0.8 | 0.9 |" in md
    assert "| qa/single_hop/accuracy | 0.5 | 0.5 |" in md
    assert eval_bench.render_report([]).startswith("# memo eval bench")


def test_safe_dir_name():
    assert eval_bench._safe_dir_name("conv-1") == "conv-1"
    assert eval_bench._safe_dir_name("a/b c") == "a_b_c"


# --- judge granularity guard ---------------------------------------------------


def test_grade_sample_qa_calls_judge_once_per_qa():
    """Guard: memo judges ONE answer per LLM-judge call. Batching multiple
    candidates into one judge call masks real differences (honest-agent-memory
    finding). This test fails if a future refactor batches grading."""
    from memo import eval_bench

    class _CountingJudge:
        name = "counting"

        def __init__(self) -> None:
            self.calls = 0

        def grade(self, *, question, gold, answer, abstention):
            self.calls += 1
            return True

    class _FakeMem:
        def ask(self, question, k=5):
            return {"answer": "some answer"}

    sample = eval_bench.BenchSample(
        sample_id="s1",
        turns=(),
        qa=(
            eval_bench.BenchQA(
                qa_id="q1",
                category="single_hop",
                question="a?",
                answer="a",
                abstention=False,
                evidence_session_ids=(),
                evidence_turn_ids=(),
            ),
            eval_bench.BenchQA(
                qa_id="q2",
                category="single_hop",
                question="b?",
                answer="b",
                abstention=False,
                evidence_session_ids=(),
                evidence_turn_ids=(),
            ),
            eval_bench.BenchQA(
                qa_id="q3",
                category="single_hop",
                question="c?",
                answer="c",
                abstention=False,
                evidence_session_ids=(),
                evidence_turn_ids=(),
            ),
        ),
    )
    judge = _CountingJudge()
    results = eval_bench.grade_sample_qa(_FakeMem(), sample, judge, k=5)
    assert judge.calls == 3
    assert len(results) == 3


# --- oracle regime ---------------------------------------------------------------


def test_grade_sample_qa_oracle_regime_bypasses_ask(monkeypatch):
    """oracle regime answers from raw sample.turns via the LLM, never calling mem.ask."""
    from memo import eval_bench

    captured = {"chat_calls": 0, "ask_calls": 0}

    class _FakeChat:
        def chat(self, model, messages, options=None):
            captured["chat_calls"] += 1
            # echo a deterministic answer derived from the turns text
            return {"message": {"content": "PARIS"}}

    monkeypatch.setattr("memo.llm.MLXChat", lambda *a, **k: _FakeChat())

    class _FakeMem:
        llm_model = "fake-model"

        def ask(self, question, k=5):
            captured["ask_calls"] += 1
            return {"answer": "should-not-be-used"}

    class _Judge:
        name = "j"

        def grade(self, *, question, gold, answer, abstention):
            return answer.strip().upper() == gold.strip().upper()

    Turn = eval_bench.BenchTurn
    sample = eval_bench.BenchSample(
        sample_id="s1",
        turns=(
            Turn(
                turn_id="t1",
                session_id="session_1",
                role="user",
                text="I live in Paris.",
                date=None,
            ),
        ),
        qa=(
            eval_bench.BenchQA(
                qa_id="q1",
                category="single_hop",
                question="Where do I live?",
                answer="Paris",
                abstention=False,
                evidence_session_ids=(),
                evidence_turn_ids=(),
            ),
        ),
    )
    results = eval_bench.grade_sample_qa(_FakeMem(), sample, _Judge(), k=5, regime="oracle")
    assert captured["ask_calls"] == 0
    assert captured["chat_calls"] == 1
    assert results[0].correct is True
