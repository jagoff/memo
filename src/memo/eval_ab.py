"""`memo eval ab` — blind-judge A/B: recall context ON vs OFF, pure logic.

The instrument docs/BENCHMARK.md lacks: the same labeled prompts (the
answerable subset of a `memo.eval_recall.labels.v1` set) are answered twice by
the local LLM — WITH memo's recall-pipeline top-K as context and WITHOUT any
context — and a blind LLM judge scores each pair on a 0-1 rubric (correctness /
groundedness / specificity). The judge never sees labels, memory ids, or which
answer used memo; pair order is randomized deterministically from a fixed seed.
A tie band absorbs judge noise. We also report the context-token cost of the
ON condition, so the output is "how much better than no memo, at what price".

Blindness is enforced twice:

* **Symmetric responder prompts.** Both conditions share the SAME system
  prompt, which forbids mentioning what sources/context/memory the answer did
  or did not have; the ON condition receives the context under the neutral
  `Background notes:` label, and a recall miss makes the ON user turn
  IDENTICAL to the OFF turn — there is no condition-naming text an answer
  could echo.
* **Leak scrub before judging.** If either answer still names its
  (non-)sources (`detect_leak`, tell-tale phrase list), the pair never
  reaches the judge: it is marked `leaked=True`, forced to a tie, and counted
  separately in the report (`leaked_pairs`).

The ON retrieval is the recall pipeline itself, not raw search:
`recall_search_fn` reproduces the eval_recall recall-faithful path — the
over-fetched candidate pool with the hook's tier exclusions, ranked by the
shared `rank_hits` under the live `knobs_from_flags` resolution, plus the
hook's flag-gated post-rank injection filters — so the A/B is comparable to
what production recall injects.

No CLI/IO concerns here (the CLI lives in `cli_eval.py`) and no MLX imports —
the chat and search callables are injected (`recall_search_fn` defers its
recall imports), so tests stay stub-only and the module imports clean off
Apple Silicon.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memo.eval_recall import LabelSet, Prompt
from memo.eval_tokens import count_tokens

# --- Versioned prompts (responder and judge are separate, frozen constants) ---

PROMPTS_VERSION = "memo.eval_ab.prompts.v2"

# ONE responder system prompt for BOTH conditions — symmetric by construction.
# The only asymmetry between ON and OFF is the neutral `Background notes:`
# block in the ON user turn; the instruction never to mention sources/context/
# memory is identical, so a compliant answer carries no condition marker the
# judge could see.
RESPONDER_SYSTEM = (
    "You are a personal assistant. Answer the user's question directly, in "
    "the question's language. Be concise and specific. If you do not know the "
    "user-specific answer, say plainly that you don't have that information. "
    "Never mention what sources, context, notes, or memory you did or did not "
    "have."
)

# Neutral context label for the ON user turn (never "memory"/"recall" words).
CONTEXT_LABEL = "Background notes:"

JUDGE_SYSTEM = (
    "You are a blind evaluator comparing two assistant answers to the same "
    "question. You do not know how either answer was produced. Score EACH "
    "answer on three criteria, each 0.0-1.0:\n"
    "- correctness: coherent, plausible, actually responsive to the question\n"
    "- groundedness: commits to concrete user-specific information instead of "
    "hedging, generic filler, or apparent fabrication\n"
    "- specificity: density of names, values, and actionable detail\n"
    "Reply with JSON only, exactly this shape:\n"
    '{"a": {"correctness": 0.0, "groundedness": 0.0, "specificity": 0.0}, '
    '"b": {"correctness": 0.0, "groundedness": 0.0, "specificity": 0.0}}'
)

SUBSCORES = ("correctness", "groundedness", "specificity")

AB_SCHEMA = "memo.eval_ab.v1"

_DEFAULT_TIE_BAND = 0.05
_PER_HIT_BODY_CHARS = 700

# (system, user) -> assistant content. The CLI wires this to MLXChat; tests
# pass a stub. Judge and responder share the transport, never the prompts.
ChatFn = Callable[[str, str], str]
SearchFn = Callable[[str], list[Any]]
ProgressFn = Callable[[int, int], None]


# --- Label selection ----------------------------------------------------------


def answerable_prompts(labels: LabelSet) -> list[Prompt]:
    """The prompts with a known answer (same rule as precision@K scoring)."""
    return [p for p in labels.prompts if p.relevant or p.expect_ids]


# --- ON-condition context -----------------------------------------------------


def build_context(hits: Sequence[Any], *, per_hit_chars: int = _PER_HIT_BODY_CHARS) -> str:
    """Render recall hits as responder context: title + body head, NO ids.

    Ids are deliberately excluded so an answer can never echo one into the
    judge's view — the judge must stay blind to which answer had memo.
    """
    lines: list[str] = []
    for i, h in enumerate(hits, start=1):
        title = str(getattr(h, "title", "") or "").strip()
        body = str(getattr(h, "body", "") or "").strip().replace("\n", " ")
        lines.append(f"{i}. {title}\n   {body[:per_hit_chars]}")
    return "\n".join(lines)


def responder_user_on(question: str, context: str) -> str:
    """ON-condition user turn. With context: neutral label + notes + question.

    On a recall MISS the turn is the bare question — byte-identical to the OFF
    turn — so there is no "(no memories recalled)" tell an answer could echo
    into the judge's view."""
    if not context:
        return question
    return f"{CONTEXT_LABEL}\n{context}\n\nQuestion: {question}"


# --- Recall-faithful ON retrieval ---------------------------------------------


def recall_search_fn(mem: Any, *, k: int) -> SearchFn:
    """Build the ON-condition SearchFn: the recall pipeline, not raw search.

    Reproduces the recall-faithful path of `eval_recall._run_config_inner`
    under the LIVE flag/overlay resolution (no grid pins): the over-fetched
    candidate pool with the hook's reference-tier + `_uncertain` exclusions and
    the reranker disabled (the shared `rank_hits` below IS the ranking, same as
    the eval gate), the flag-gated recency band, `rank_hits` with
    `knobs_from_flags(top_k=k)` (hybrid true-cosine gate included), then the
    hook's flag-gated post-rank injection filters (skip-below/gap trim,
    unmatched-term gate, pre-top-K paraphrase collapse). Returns the top-K —
    what production recall would inject for the prompt.

    Recall imports are deferred so importing this module stays MLX-free.
    """
    from memo.flags import flag_bool, flag_float, flag_int
    from memo.recall_logic import (
        apply_injection_filters,
        apply_recency_band,
        collapse_near_dups,
        fetch_recency_band,
        knobs_from_flags,
        make_vec_cosine,
        rank_hits,
        uncertain_exclusion,
        unmatched_term_gate,
    )
    from memo.tiers import REFERENCE_TYPES

    knobs = knobs_from_flags(top_k=k)
    exclude_types = set(REFERENCE_TYPES) if flag_bool("MEMO_RECALL_EXCLUDE_REFERENCE") else None
    exclude_tags = uncertain_exclusion()

    def _search(text: str) -> list[Any]:
        # `memo eval ab` is not a user-visible retrieval: without this, every
        # search hit writes an access-log row (search_ops.py's
        # `_stage_record_usage`), inflating `access_count` on whichever
        # memories the eval surfaces — the same signal `memo usefulness` /
        # `dead_weight()` read to decide what's noise. See eval_recall.py's
        # `_search_for_eval` for the same fix on the sibling gate.
        hits = list(
            mem.search(
                text,
                limit=k * 4,
                mode=knobs.mode,
                disable_reranker=True,
                exclude_types=exclude_types,
                exclude_tags=exclude_tags,
                _track_usage=False,
            )
        )
        band_days = flag_int("MEMO_RECALL_RECENCY_BAND_DAYS") or 0
        if band_days > 0:
            hits = apply_recency_band(
                hits,
                fetch_recency_band(
                    mem, days=band_days, exclude_types=exclude_types, floor=knobs.min_sim
                ),
            )
        vc = make_vec_cosine(mem, text) if knobs.mode == "hybrid" else None
        ranked = rank_hits(hits, knobs, vec_cosine=vc)
        ranked = apply_injection_filters(ranked)
        if flag_bool("MEMO_RECALL_UNMATCHED_TERM_GATE") and unmatched_term_gate(text, ranked):
            ranked = []
        if flag_bool("MEMO_RECALL_DEDUP_COLLAPSE") and len(ranked) > 1:
            ranked = collapse_near_dups(
                ranked, threshold=flag_float("MEMO_RECALL_INTRA_DEDUP_THRESHOLD") or 0.8
            )
        return ranked[:k]

    return _search


# --- Leak scrub (defense in depth for judge blindness) ------------------------

# Tell-tale phrases that de-anonymize a condition if the judge reads them in an
# answer: the old asymmetric-prompt vocabulary, the "no memories" miss marker,
# and the current neutral context label. Substring match, case-insensitive.
LEAK_PHRASES = (
    "memory context",
    "saved memory",
    "no memories",
    "recalled",
    "background notes",
)


def detect_leak(answer: str) -> bool:
    """True when an answer names its (non-)sources — text that would break the
    judge's blindness. Checked BEFORE judging; a leaked pair never reaches the
    judge (see `run_pair`)."""
    low = (answer or "").lower()
    return any(phrase in low for phrase in LEAK_PHRASES)


# --- Blind judge --------------------------------------------------------------


def on_goes_first(seed: int, index: int, prompt_text: str) -> bool:
    """Deterministic per-pair order: seeded hash, no `random` module."""
    digest = hashlib.sha256(f"{seed}:{index}:{prompt_text}".encode()).digest()
    return digest[0] % 2 == 0


def judge_user_prompt(question: str, answer_a: str, answer_b: str) -> str:
    """The judge sees ONLY the question and the two answers — no labels, no
    ids, no memory context, no condition names."""
    return f"Question:\n{question}\n\nAnswer A:\n{answer_a}\n\nAnswer B:\n{answer_b}"


def _clamp01(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, v))


def parse_judge_scores(raw: str) -> tuple[dict[str, float], dict[str, float], bool]:
    """Parse the judge JSON into ({a subscores}, {b subscores}, parse_error).

    Tolerates prose around the JSON (first `{` to last `}`). On any failure
    both sides get all-zero subscores and parse_error=True — a broken judge
    call must never silently count as a win for either condition.
    """
    zero = dict.fromkeys(SUBSCORES, 0.0)
    text = raw or ""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return dict(zero), dict(zero), True
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return dict(zero), dict(zero), True
    if not isinstance(data, dict):
        return dict(zero), dict(zero), True
    out: list[dict[str, float]] = []
    for key in ("a", "b"):
        side = data.get(key)
        if not isinstance(side, dict):
            return dict(zero), dict(zero), True
        out.append({s: _clamp01(side.get(s)) for s in SUBSCORES})
    return out[0], out[1], False


def mean_score(subscores: dict[str, float]) -> float:
    return round(sum(subscores.get(s, 0.0) for s in SUBSCORES) / len(SUBSCORES), 4)


def decide_winner(score_on: float, score_off: float, *, tie_band: float) -> str:
    """`on` / `off` / `tie` — the band absorbs judge noise on near-equal pairs."""
    if abs(score_on - score_off) <= tie_band:
        return "tie"
    return "on" if score_on > score_off else "off"


# --- Run ----------------------------------------------------------------------


@dataclass
class PairResult:
    prompt: str
    order: str  # "on_first" | "off_first" — which condition was Answer A
    answer_on: str
    answer_off: str
    n_context_hits: int
    context_tokens_on: int
    context_tokens_off: int  # always 0; kept explicit for the report
    scores_on: dict[str, float] = field(default_factory=dict)
    scores_off: dict[str, float] = field(default_factory=dict)
    mean_on: float = 0.0
    mean_off: float = 0.0
    winner: str = "tie"
    judge_raw: str = ""
    judge_parse_error: bool = False
    leaked: bool = False  # an answer named its (non-)sources; judge skipped


def run_pair(
    prompt: Prompt,
    index: int,
    *,
    search: SearchFn,
    chat: ChatFn,
    k: int,
    seed: int,
    tie_band: float,
) -> PairResult:
    hits = list(search(prompt.text))[:k]
    context = build_context(hits)

    answer_on = chat(RESPONDER_SYSTEM, responder_user_on(prompt.text, context))
    answer_off = chat(RESPONDER_SYSTEM, prompt.text)

    on_first = on_goes_first(seed, index, prompt.text)
    if detect_leak(answer_on) or detect_leak(answer_off):
        # Blindness is broken: the judge must never see this pair. Forced tie,
        # counted separately in the report (summarize's `leaked_pairs`).
        zero = dict.fromkeys(SUBSCORES, 0.0)
        return PairResult(
            prompt=prompt.text,
            order="on_first" if on_first else "off_first",
            answer_on=answer_on,
            answer_off=answer_off,
            n_context_hits=len(hits),
            context_tokens_on=count_tokens(context),
            context_tokens_off=0,
            scores_on=dict(zero),
            scores_off=dict(zero),
            winner="tie",
            leaked=True,
        )

    a, b = (answer_on, answer_off) if on_first else (answer_off, answer_on)
    judge_raw = chat(JUDGE_SYSTEM, judge_user_prompt(prompt.text, a, b))
    scores_a, scores_b, parse_error = parse_judge_scores(judge_raw)
    scores_on, scores_off = (scores_a, scores_b) if on_first else (scores_b, scores_a)

    m_on, m_off = mean_score(scores_on), mean_score(scores_off)
    winner = "tie" if parse_error else decide_winner(m_on, m_off, tie_band=tie_band)
    return PairResult(
        prompt=prompt.text,
        order="on_first" if on_first else "off_first",
        answer_on=answer_on,
        answer_off=answer_off,
        n_context_hits=len(hits),
        context_tokens_on=count_tokens(context),
        context_tokens_off=0,
        scores_on=scores_on,
        scores_off=scores_off,
        mean_on=m_on,
        mean_off=m_off,
        winner=winner,
        judge_raw=judge_raw,
        judge_parse_error=parse_error,
    )


def run_ab(
    prompts: Sequence[Prompt],
    *,
    search: SearchFn,
    chat: ChatFn,
    k: int = 5,
    seed: int = 42,
    tie_band: float = _DEFAULT_TIE_BAND,
    progress: ProgressFn | None = None,
) -> list[PairResult]:
    """Answer + blind-judge every prompt. Up to 3 chat calls per prompt (ON
    answer, OFF answer, judge — the judge is skipped for leaked pairs) —
    offline batch only, never the recall hook."""
    results: list[PairResult] = []
    for index, prompt in enumerate(prompts):
        if progress is not None:
            progress(index + 1, len(prompts))
        results.append(
            run_pair(prompt, index, search=search, chat=chat, k=k, seed=seed, tie_band=tie_band)
        )
    return results


# --- Aggregation & audit trail ------------------------------------------------


def summarize(results: Sequence[PairResult]) -> dict[str, Any]:
    n = len(results)
    wins = sum(1 for r in results if r.winner == "on")
    ties = sum(1 for r in results if r.winner == "tie")
    losses = sum(1 for r in results if r.winner == "off")
    sub_deltas = {
        s: round(sum(r.scores_on.get(s, 0.0) - r.scores_off.get(s, 0.0) for r in results) / n, 4)
        if n
        else 0.0
        for s in SUBSCORES
    }
    return {
        "prompts": n,
        "wins_on": wins,
        "ties": ties,
        "losses_on": losses,
        "win_rate_on": round(wins / n, 4) if n else 0.0,
        "mean_delta": round(sum(r.mean_on - r.mean_off for r in results) / n, 4) if n else 0.0,
        "sub_deltas": sub_deltas,
        "context_tokens_on": sum(r.context_tokens_on for r in results),
        "context_tokens_off": sum(r.context_tokens_off for r in results),
        "judge_parse_errors": sum(1 for r in results if r.judge_parse_error),
        "leaked_pairs": sum(1 for r in results if r.leaked),
    }


def write_detail(state_dir: Path, payload: dict[str, Any]) -> Path:
    """Persist the raw run (pairs + answers + judge output) for auditing."""
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = state_dir / "eval" / f"ab_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
