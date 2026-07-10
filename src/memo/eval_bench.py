"""Public long-memory benchmark adapter (LoCoMo / LongMemEval) — pure logic.

`memo eval bench` (cli_eval_bench.py) ingests a public benchmark into an
ISOLATED store (per-sample data_dir/state_dir under <state_dir>/bench/,
never the live corpus), scores retrieval through the SAME
eval_recall.run_config → rank_hits path the recall eval uses, and grades
end-to-end `memo ask` answers per question category with a pluggable judge.

Offline batch only — nothing here runs in the UserPromptSubmit recall hook.
No new dependencies: stdlib urllib for the one-shot dataset download,
strptime (not dateparser) for the two datasets' date formats.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from memo import eval_bench_taxonomy as taxonomy
from memo.config import Config
from memo.errors import MemoError
from memo.eval_recall import Cfg, LabelSet, Prompt, Row, run_config

# --- Normalized schema --------------------------------------------------------


@dataclass(frozen=True)
class BenchTurn:
    turn_id: str  # dataset-native id: LoCoMo dia_id ("D1:3") or "<session_id>:<idx>"
    session_id: str
    role: str  # speaker name (LoCoMo) or user/assistant (LongMemEval)
    text: str
    date: str | None  # ISO8601 or None (falls back to save-time NOW)


@dataclass(frozen=True)
class BenchQA:
    qa_id: str
    question: str
    answer: str
    category: str  # normalized: single_hop / multi_hop / temporal_reasoning / knowledge_update / ...
    abstention: bool  # correct behavior is to decline (adversarial / _abs)
    evidence_session_ids: tuple[str, ...]
    evidence_turn_ids: tuple[str, ...]


@dataclass(frozen=True)
class BenchSample:
    """One ingestion scope: a LoCoMo conversation or a LongMemEval haystack."""

    sample_id: str
    turns: tuple[BenchTurn, ...]
    qa: tuple[BenchQA, ...]


# --- Date parsing (offline, no dateparser dep) ---------------------------------

# LoCoMo: "1:56 pm on 8 May, 2023" · LongMemEval: "2023/05/20 (Sat) 02:21"
_DATE_FORMATS = ("%I:%M %p on %d %B, %Y", "%Y/%m/%d (%a) %H:%M", "%Y-%m-%d")


def _parse_bench_date(raw: str | None) -> str | None:
    if not raw:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt).isoformat()
        except ValueError:
            continue
    return None


# --- Parsers --------------------------------------------------------------------

# Category ints per the LoCoMo paper / locomo10.json; abstention is ALSO
# detected structurally (adversarial_answer key), so a numbering drift in the
# dataset only mislabels category names, never correctness semantics.
_LOCOMO_CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal_reasoning",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}
_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


def _locomo_session_keys(conv: dict[str, Any]) -> list[str]:
    keyed = []
    for key, val in conv.items():
        m = _SESSION_KEY_RE.match(key)
        if m and isinstance(val, list):
            keyed.append((int(m.group(1)), key))
    return [k for _, k in sorted(keyed)]


def parse_locomo(raw: Any) -> list[BenchSample]:
    """Normalize locomo10.json (a list of conversation samples)."""
    if isinstance(raw, dict):
        raw = [raw]
    samples: list[BenchSample] = []
    for s in raw or []:
        conv = s.get("conversation") or {}
        sample_id = str(s.get("sample_id") or f"sample_{len(samples)}")
        turns: list[BenchTurn] = []
        for key in _locomo_session_keys(conv):
            sid = key
            date = _parse_bench_date(conv.get(f"{key}_date_time"))
            for i, d in enumerate(conv[key]):
                text = str(d.get("text") or "").strip()
                if not text:
                    continue  # image-only turns carry no text
                turns.append(
                    BenchTurn(
                        turn_id=str(d.get("dia_id") or f"{sid}:{i}"),
                        session_id=sid,
                        role=str(d.get("speaker") or ""),
                        text=text,
                        date=date,
                    )
                )
        qa_items: list[BenchQA] = []
        for j, q in enumerate(s.get("qa") or []):
            try:
                cat_num = int(q.get("category") or 0)
            except (TypeError, ValueError):
                cat_num = 0
            abstention = "adversarial_answer" in q or cat_num == 5
            gold = str((q.get("adversarial_answer") if abstention else q.get("answer")) or "")
            qa_items.append(
                BenchQA(
                    qa_id=f"{sample_id}:{j}",
                    question=str(q.get("question") or ""),
                    answer=gold,
                    category=_LOCOMO_CATEGORY_NAMES.get(cat_num, f"category_{cat_num}"),
                    abstention=abstention,
                    evidence_session_ids=(),
                    evidence_turn_ids=tuple(str(e) for e in (q.get("evidence") or [])),
                )
            )
        samples.append(BenchSample(sample_id=sample_id, turns=tuple(turns), qa=tuple(qa_items)))
    return samples


def parse_longmemeval(raw: Any) -> list[BenchSample]:
    """Normalize a LongMemEval JSON (a list of per-question haystack instances)."""
    samples: list[BenchSample] = []
    for inst in raw or []:
        qid = str(inst.get("question_id") or f"q_{len(samples)}")
        sids = [str(x) for x in (inst.get("haystack_session_ids") or [])]
        dates = [str(x) for x in (inst.get("haystack_dates") or [])]
        turns: list[BenchTurn] = []
        evidence_turn_ids: list[str] = []
        for si, session in enumerate(inst.get("haystack_sessions") or []):
            sid = sids[si] if si < len(sids) else f"session_{si}"
            date = _parse_bench_date(dates[si]) if si < len(dates) else None
            for ti, t in enumerate(session or []):
                text = str(t.get("content") or "").strip()
                if not text:
                    continue
                tid = f"{sid}:{ti}"
                turns.append(
                    BenchTurn(
                        turn_id=tid,
                        session_id=sid,
                        role=str(t.get("role") or ""),
                        text=text,
                        date=date,
                    )
                )
                if t.get("has_answer"):
                    evidence_turn_ids.append(tid)
        qa = BenchQA(
            qa_id=qid,
            question=str(inst.get("question") or ""),
            answer=str(inst.get("answer") or ""),
            category=str(inst.get("question_type") or "unknown").replace("-", "_"),
            abstention=qid.endswith("_abs"),
            evidence_session_ids=tuple(str(x) for x in (inst.get("answer_session_ids") or [])),
            evidence_turn_ids=tuple(evidence_turn_ids),
        )
        samples.append(BenchSample(sample_id=qid, turns=tuple(turns), qa=(qa,)))
    return samples


def parse_dataset(name: str, raw: Any) -> list[BenchSample]:
    if name == "locomo":
        return parse_locomo(raw)
    if name.startswith("longmemeval"):
        return parse_longmemeval(raw)
    raise MemoError(f"unknown bench dataset {name!r}")


# --- Dataset download / cache ----------------------------------------------------

DATASET_URLS: dict[str, str] = {
    # LoCoMo-10 (Maharana et al. 2024) — public JSON in the snap-research repo.
    "locomo": "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    # LongMemEval (Wu et al. 2024) — JSON files on the HF dataset mirror.
    # oracle = evidence-only haystacks (small, fast); _s = ~115k-token haystacks.
    "longmemeval_oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle.json",
    "longmemeval_s": "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s.json",
}

Fetcher = Callable[[str], bytes]


def _http_fetch(url: str) -> bytes:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "memo-eval-bench"})  # noqa: S310 — https URL from fixed table or explicit --url
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — https URL from fixed table or explicit --url
        return resp.read()


def fetch_dataset(
    name: str,
    dest_dir: Path,
    *,
    url: str | None = None,
    fetcher: Fetcher = _http_fetch,
) -> Path:
    """Download-and-cache a benchmark JSON; returns the cached path.

    An existing non-empty cache file is reused (delete it to re-download).
    The payload is validated as JSON before caching so an HTML error page
    never poisons the cache."""
    resolved = url or DATASET_URLS.get(name)
    if not resolved:
        raise MemoError(
            f"unknown bench dataset {name!r}; known: {', '.join(sorted(DATASET_URLS))} "
            "(or pass --url / --file)"
        )
    dest = dest_dir / f"{name}.json"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    data = fetcher(resolved)
    json.loads(data.decode("utf-8"))  # fail fast on a non-JSON payload
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return dest


# --- Isolated bench store ----------------------------------------------------------


def bench_store_config(root: Path, live: Any) -> Config:
    """Build the ISOLATED Config for a bench store under `root`.

    data_dir/state_dir point INSIDE `root` (explicit kwargs — the tmp_cfg
    isolation pattern, never Config.from_env()), so ingestion physically
    cannot touch the live corpus. Only model settings are copied from the
    live config so bench vectors match the live embedder."""
    data = root / "data"
    state = root / "state"
    data.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    return Config(
        data_dir=data,
        state_dir=state,
        llm_model=live.llm_model,
        embedder_model=live.embedder_model,
        embedder_dims=live.embedder_dims,
        reranker_enabled=live.reranker_enabled,
    )


# --- Ingestion into the isolated store ----------------------------------------------

MANIFEST_NAME = "manifest.json"


@dataclass
class IngestResult:
    turn_to_memory: dict[str, str]  # turn_id -> full memory id
    turn_to_session: dict[str, str]  # turn_id -> session_id


def load_manifest(root: Path) -> IngestResult | None:
    p = root / MANIFEST_NAME
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return IngestResult(
            turn_to_memory={str(k): str(v) for k, v in raw["turn_to_memory"].items()},
            turn_to_session={str(k): str(v) for k, v in raw["turn_to_session"].items()},
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return None


def ingest_sample(
    mem: Any,
    sample: BenchSample,
    root: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> IngestResult:
    """Save every conversation turn as a durable memory in the bench store.

    Idempotent: a manifest covering all of the sample's turns short-circuits
    (re-runs skip re-embedding). Turns carry bench provenance in `extra`
    (bench_turn_id / bench_session_id) — the evidence join key for scoring —
    and are back-dated via save(created=...) so temporal-reasoning questions
    see original event time, not ingest time."""
    cached = load_manifest(root)
    if cached is not None and len(cached.turn_to_memory) == len(sample.turns):
        return cached
    turn_to_memory: dict[str, str] = {}
    turn_to_session: dict[str, str] = {}
    for i, turn in enumerate(sample.turns, start=1):
        if progress is not None:
            progress(i, len(sample.turns))
        rec = mem.save(
            content=f"{turn.role}: {turn.text}" if turn.role else turn.text,
            title=f"{sample.sample_id} {turn.turn_id}",
            type_="note",  # durable tier: reference tier is excluded from the recall pool
            tags=[f"bench:{sample.sample_id}"],
            extra={"bench_turn_id": turn.turn_id, "bench_session_id": turn.session_id},
            auto_project=False,
            created=turn.date,
        )
        turn_to_memory[turn.turn_id] = rec.id
        turn_to_session[turn.turn_id] = turn.session_id
    result = IngestResult(turn_to_memory, turn_to_session)
    (root / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "sample_id": sample.sample_id,
                "turn_to_memory": turn_to_memory,
                "turn_to_session": turn_to_session,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


# --- Retrieval scoring (shared rank_hits path via eval_recall.run_config) ------------


def bench_retrieval_cfg() -> Cfg:
    """One config mirroring the live hook's mode + similarity floor."""
    from memo.flags import flag_float, flag_str

    return Cfg(
        "bench",
        mode=str(flag_str("MEMO_RECALL_MODE") or "vec"),
        floor=float(flag_float("MEMO_RECALL_MIN_SIM") or 0.0),
        exclude_archived=False,
    )


def expected_memory_ids(qa: BenchQA, ingest: IngestResult) -> list[str]:
    """Evidence → memory ids: turn-level ids first, session-level fallback."""
    ids = {ingest.turn_to_memory[t] for t in qa.evidence_turn_ids if t in ingest.turn_to_memory}
    if not ids and qa.evidence_session_ids:
        wanted = set(qa.evidence_session_ids)
        ids = {
            mid
            for tid, mid in ingest.turn_to_memory.items()
            if ingest.turn_to_session.get(tid) in wanted
        }
    return sorted(ids)


def category_label_sets(sample: BenchSample, ingest: IngestResult) -> dict[str, LabelSet]:
    """One LabelSet per question category. Abstention questions carry no
    expect_ids (relevant=False) — they are graded end-to-end in Task 6, not
    as retrieval."""
    by_cat: dict[str, list[Prompt]] = {}
    for qa in sample.qa:
        ids = [] if qa.abstention else expected_memory_ids(qa, ingest)
        by_cat.setdefault(qa.category, []).append(
            Prompt(text=qa.question, relevant=bool(ids), expect_ids=ids)
        )
    return {cat: LabelSet(prompts=ps) for cat, ps in by_cat.items()}


def score_retrieval(
    mem: Any, sample: BenchSample, ingest: IngestResult, *, k: int
) -> dict[str, tuple[Row, int]]:
    """Per-category Row via eval_recall.run_config — the SAME dedup +
    rank_hits + reference-tier-exclusion path `memo eval recall` measures.
    Returns {category: (Row, n_scored_prompts)}."""
    cfg = bench_retrieval_cfg()
    out: dict[str, tuple[Row, int]] = {}
    for cat, labels in category_label_sets(sample, ingest).items():
        n = sum(1 for p in labels.prompts if p.expect_ids)
        out[cat] = (run_config(mem, cfg, k, labels), n)
    return out


def aggregate_retrieval(
    per_sample: list[dict[str, tuple[Row, int]]],
) -> dict[str, dict[str, float | int]]:
    """Weighted-mean (by scored-prompt count) retrieval metrics per category."""
    acc: dict[str, dict[str, float]] = {}
    for rows in per_sample:
        for cat, (row, n) in rows.items():
            if n <= 0:
                continue
            slot = acc.setdefault(
                cat,
                {"recall_at_k": 0.0, "ndcg_at_k": 0.0, "mrr": 0.0, "precision_at_k": 0.0, "n": 0.0},
            )
            slot["recall_at_k"] += row.recall_at_k * n
            slot["ndcg_at_k"] += row.ndcg_at_k * n
            slot["mrr"] += row.mrr * n
            slot["precision_at_k"] += row.precision_at_k * n
            slot["n"] += n
    out: dict[str, dict[str, float | int]] = {}
    for cat, slot in acc.items():
        total = slot.pop("n")
        out[cat] = {m: round(v / total, 3) for m, v in slot.items()}
        out[cat]["n_questions"] = int(total)
    return out


# --- Pluggable QA judge ---------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are grading a memory assistant's answer against a gold answer. "
    "Reply with exactly one word: yes or no."
)


def _judge_user_prompt(question: str, gold: str, answer: str, abstention: bool) -> str:
    if abstention:
        return (
            "The correct behavior for this question is to ABSTAIN — the "
            "information is not present in memory.\n"
            f"Question: {question}\n"
            f"Model answer: {answer}\n"
            "Does the model answer decline to answer or state that the "
            "information is unavailable? Reply yes or no."
        )
    return (
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Model answer: {answer}\n"
        "Does the model answer contain or entail the gold answer? Reply yes or no."
    )


def _parse_verdict(text: str) -> bool:
    return (text or "").strip().lower().startswith("yes")


class Judge(Protocol):
    name: str

    def grade(self, *, question: str, gold: str, answer: str, abstention: bool) -> bool: ...


class MLXJudge:
    """Local judge on memo's MLXChat (default). MLX loads lazily on first grade."""

    name = "mlx"

    def __init__(self, model: str) -> None:
        self._model = model
        self._chat: Any = None

    def grade(self, *, question: str, gold: str, answer: str, abstention: bool) -> bool:
        if self._chat is None:
            from memo.llm import MLXChat  # deferred — never load MLX at import time

            self._chat = MLXChat()
        out = self._chat.chat(
            self._model,
            [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": _judge_user_prompt(question, gold, answer, abstention),
                },
            ],
            options={"temperature": 0.0, "num_predict": 8},
        )
        return _parse_verdict(out.get("message", {}).get("content", ""))


class APIJudge:
    """OpenAI-compatible chat-completions judge (env-gated via MEMO_BENCH_JUDGE=api)."""

    name = "api"

    def __init__(self, url: str, model: str, api_key: str) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._api_key = api_key

    def grade(self, *, question: str, gold: str, answer: str, abstention: bool) -> bool:
        import urllib.request

        payload = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": 8,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": _judge_user_prompt(question, gold, answer, abstention),
                },
            ],
        }
        req = urllib.request.Request(  # noqa: S310 — user-configured https endpoint
            f"{self._url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — user-configured https endpoint
            data = json.load(resp)
        return _parse_verdict(data["choices"][0]["message"]["content"])


def judge_from_flags(live: Any) -> Judge:
    """Resolve the judge from MEMO_BENCH_* flags. Local MLX by default; the
    api judge requires URL + model and reads its key from the env var named
    by MEMO_BENCH_JUDGE_API_KEY_ENV (secrets stay out of MEMO_* flags)."""
    from memo.flags import flag_str

    kind = (flag_str("MEMO_BENCH_JUDGE") or "mlx").strip().lower()
    model = (flag_str("MEMO_BENCH_JUDGE_MODEL") or "").strip()
    if kind == "mlx":
        return MLXJudge(model or live.llm_model)
    if kind == "api":
        url = (flag_str("MEMO_BENCH_JUDGE_URL") or "").strip()
        if not url or not model:
            raise MemoError(
                "MEMO_BENCH_JUDGE=api requires MEMO_BENCH_JUDGE_URL and MEMO_BENCH_JUDGE_MODEL"
            )
        key_env = (flag_str("MEMO_BENCH_JUDGE_API_KEY_ENV") or "OPENAI_API_KEY").strip()
        api_key = os.environ.get(key_env, "")
        if not api_key:
            raise MemoError(
                f"api judge: env var {key_env} is empty "
                "(set it, or point MEMO_BENCH_JUDGE_API_KEY_ENV at the right variable)"
            )
        return APIJudge(url, model, api_key)
    raise MemoError(f"unknown MEMO_BENCH_JUDGE {kind!r} (expected: mlx | api)")


# --- End-to-end QA grading ---------------------------------------------------------


@dataclass
class QAResult:
    qa_id: str
    category: str
    abstention: bool
    correct: bool
    answer_head: str  # first 400 chars, for the receipt


def grade_sample_qa(
    mem: Any,
    sample: BenchSample,
    judge: Judge,
    *,
    k: int = 5,
    max_qa: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[QAResult]:
    """`memo ask` each QA over the bench store and judge the answer."""
    items = list(sample.qa)[:max_qa] if max_qa else list(sample.qa)
    results: list[QAResult] = []
    for i, qa in enumerate(items, start=1):
        if progress is not None:
            progress(i, len(items))
        res = mem.ask(qa.question, k=k)
        answer = str(res.get("answer") or "")
        correct = judge.grade(
            question=qa.question, gold=qa.answer, answer=answer, abstention=qa.abstention
        )
        results.append(QAResult(qa.qa_id, qa.category, qa.abstention, correct, answer[:400]))
    return results


def qa_accuracy_by_category(results: list[QAResult]) -> dict[str, dict[str, float | int]]:
    by_cat: dict[str, list[QAResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    return {
        cat: {
            "accuracy": round(sum(1 for r in rs if r.correct) / len(rs), 3),
            "n_questions": len(rs),
        }
        for cat, rs in by_cat.items()
    }


# --- Capability taxonomy rollup + abstention (auxiliary view) -------------------------
#
# Memoria's benchmark reports a unified 6-bucket ability taxonomy on top of the
# raw per-category numbers (docs/memory-ability-taxonomy.md). We do the same:
# the raw category numbers stay primary; these give a cross-dataset,
# cross-run-comparable view and pull abstention out as its own first-class
# metric instead of leaving it mixed into a topic category.

_RETRIEVAL_METRICS = ("recall_at_k", "ndcg_at_k", "mrr", "precision_at_k")


def capability_retrieval(
    by_category: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    """Roll aggregate_retrieval() output up into capability buckets."""
    return taxonomy.rollup_weighted(by_category, _RETRIEVAL_METRICS)


def capability_qa(results: list[QAResult]) -> dict[str, dict[str, float | int]]:
    """Per-capability-bucket QA accuracy. Abstention questions route to the
    abstention bucket (bucket_for honors the flag), not their topic category."""
    by_bucket: dict[str, list[QAResult]] = {}
    for r in results:
        by_bucket.setdefault(taxonomy.bucket_for(r.category, r.abstention), []).append(r)
    return {
        bucket: {
            "accuracy": round(sum(1 for r in rs if r.correct) / len(rs), 3),
            "n_questions": len(rs),
        }
        for bucket, rs in by_bucket.items()
    }


def abstention_summary(results: list[QAResult]) -> dict[str, float | int]:
    """First-class abstention / hallucination metric across all categories.

    For an abstention question the correct behavior is to decline; the judge
    grades `correct=True` when the answer declines. So over the abstention
    subset: `abstention_accuracy` = correctly-declined rate, and
    `hallucination_rate` = answered-anyway rate (1 − accuracy) — the share of
    no-evidence questions memo answered instead of abstaining."""
    abst = [r for r in results if r.abstention]
    n = len(abst)
    if n == 0:
        return {
            "n_questions": 0,
            "correct_abstentions": 0,
            "hallucinations": 0,
            "abstention_accuracy": 0.0,
            "hallucination_rate": 0.0,
        }
    correct = sum(1 for r in abst if r.correct)
    return {
        "n_questions": n,
        "correct_abstentions": correct,
        "hallucinations": n - correct,
        "abstention_accuracy": round(correct / n, 3),
        "hallucination_rate": round((n - correct) / n, 3),
    }


# --- Results receipt + markdown report --------------------------------------------------

RECEIPT_SCHEMA = "memo.eval_bench.receipt.v1"


def _safe_dir_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "sample"


def runs_dir(state_dir: Path) -> Path:
    return state_dir / "bench" / "runs"


def write_receipt(state_dir: Path, receipt: dict[str, Any]) -> Path:
    out = runs_dir(state_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = out / f"{receipt.get('dataset', 'bench')}-{ts}.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_receipts(
    state_dir: Path, *, last: int = 3, dataset: str | None = None
) -> list[dict[str, Any]]:
    """Most-recent-first receipts (schema-checked, optionally dataset-filtered)."""
    out: list[dict[str, Any]] = []
    d = runs_dir(state_dir)
    if not d.exists():
        return out
    for p in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(r, dict) or r.get("schema") != RECEIPT_SCHEMA:
            continue
        if dataset and r.get("dataset") != dataset:
            continue
        r["_file"] = p.name
        out.append(r)
        if len(out) >= last:
            break
    return out


def render_report(receipts: list[dict[str, Any]]) -> str:
    """Markdown comparison — one column per run, newest last; stable
    `retrieval/<category>/<metric>` and `qa/<category>/accuracy` row keys so
    runs stay comparable as categories come and go."""
    if not receipts:
        return "# memo eval bench\n\n_No runs found._\n"
    cols = list(reversed(receipts))  # oldest → newest, newest last column
    lines = ["# memo eval bench — run comparison", ""]
    lines.append("| metric | " + " | ".join(str(c.get("_file", c.get("ts", "?"))) for c in cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))

    def row(label: str, values: list[str]) -> str:
        return f"| {label} | " + " | ".join(values) + " |"

    lines.append(row("dataset", [str(c.get("dataset", "?")) for c in cols]))
    lines.append(row("k", [str(c.get("k", "?")) for c in cols]))
    lines.append(row("judge", [str(c.get("judge") or "—") for c in cols]))
    ret_cats = sorted({cat for c in cols for cat in (c.get("retrieval") or {})})
    for cat in ret_cats:
        for metric in ("recall_at_k", "ndcg_at_k", "mrr", "precision_at_k", "n_questions"):
            lines.append(
                row(
                    f"retrieval/{cat}/{metric}",
                    [
                        str((c.get("retrieval") or {}).get(cat, {}).get(metric, "—"))
                        for c in cols
                    ],
                )
            )
    qa_cats = sorted({cat for c in cols for cat in (c.get("qa") or {})})
    for cat in qa_cats:
        lines.append(
            row(
                f"qa/{cat}/accuracy",
                [str((c.get("qa") or {}).get(cat, {}).get("accuracy", "—")) for c in cols],
            )
        )

    # Auxiliary capability-taxonomy rollup (Memoria-style 6-bucket view).
    cap_ret_buckets = sorted({b for c in cols for b in (c.get("capability_retrieval") or {})})
    for bucket in cap_ret_buckets:
        for metric in ("recall_at_k", "ndcg_at_k", "mrr", "precision_at_k", "n_questions"):
            lines.append(
                row(
                    f"capability/{bucket}/{metric}",
                    [
                        str((c.get("capability_retrieval") or {}).get(bucket, {}).get(metric, "—"))
                        for c in cols
                    ],
                )
            )
    cap_qa_buckets = sorted({b for c in cols for b in (c.get("capability_qa") or {})})
    for bucket in cap_qa_buckets:
        lines.append(
            row(
                f"capability_qa/{bucket}/accuracy",
                [str((c.get("capability_qa") or {}).get(bucket, {}).get("accuracy", "—")) for c in cols],
            )
        )

    # Abstention as a first-class cross-cut (declined-when-no-evidence).
    if any(c.get("abstention") for c in cols):
        for metric in ("abstention_accuracy", "hallucination_rate", "n_questions"):
            lines.append(
                row(
                    f"abstention/{metric}",
                    [str((c.get("abstention") or {}).get(metric, "—")) for c in cols],
                )
            )
    return "\n".join(lines) + "\n"
