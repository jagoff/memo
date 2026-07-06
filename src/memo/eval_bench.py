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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
