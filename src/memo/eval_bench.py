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

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from memo.errors import MemoError

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
