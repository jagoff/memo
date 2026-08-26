"""`memo eval tokens` — token-economy measurement gate.

Mirrors the `memo eval recall --gate` retrieval-regression discipline for the
token economy. Measures, per lever, whether it saves tokens WITHOUT degrading
recall, over a committed corpus. Two planes memo controls directly (no LLM):

  P1 recall-output block size — quality guard = precision over ids that survive
     into the injected block.
  P2 capture size — quality guard = a labeled must-keep row surviving the crush.

Design note: we measure the transformation FUNCTION at unit level; a lever need
not be wired into the live pipeline to be measured. The data decides wiring.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CHARS_PER_TOKEN = 4  # keep in lockstep with token_meter._CHARS_PER_TOKEN

MIN_SAVING_FRAC = 0.05  # a lever must cut >=5% tokens on its plane to count
QUALITY_EPS = 0.0  # precision must not drop; must-keep row must survive


def count_tokens(text: str) -> int:
    """Token count: tiktoken (cl100k_base) when available, chars/4 fallback."""
    if not text:
        return 0
    try:
        from memo.token_meter import count_tokens_accurate

        return count_tokens_accurate(text)
    except Exception:
        return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def surviving_ids(block_text: str, candidate_ids: list[str]) -> set[str]:
    """Which candidate memory ids survived into a rendered recall block.

    The renderer prints each hit as `**[<id[:8]>] title**`, so an id counts as
    surviving iff its 8-char short prefix appears anywhere in the block.
    """
    return {cid for cid in candidate_ids if cid[:8] in block_text}


@dataclass
class LeverRow:
    lever: str
    plane: str  # "recall_output" | "capture"
    tokens_off: int
    tokens_on: int
    quality_off: float
    quality_on: float

    @property
    def saved_frac(self) -> float:
        return (self.tokens_off - self.tokens_on) / self.tokens_off if self.tokens_off else 0.0

    @property
    def quality_delta(self) -> float:
        return self.quality_on - self.quality_off

    @property
    def passed(self) -> bool:
        return self.saved_frac >= MIN_SAVING_FRAC and self.quality_delta >= -QUALITY_EPS


@dataclass
class P1Sample:
    tokens_off: int
    tokens_on: int
    prec_off: float
    prec_on: float


def _precision(block: str, expect_ids: list[str]) -> float:
    if not expect_ids:
        return 1.0
    return len(surviving_ids(block, expect_ids)) / len(expect_ids)


def measure_recall_sample(block_off: str, block_on: str, expect_ids: list[str]) -> P1Sample:
    """One prompt: token + precision measurement of the OFF vs ON recall block."""
    return P1Sample(
        tokens_off=count_tokens(block_off),
        tokens_on=count_tokens(block_on),
        prec_off=_precision(block_off, expect_ids),
        prec_on=_precision(block_on, expect_ids),
    )


def aggregate_recall(lever: str, samples: list[P1Sample]) -> LeverRow:
    """Fold per-prompt P1 samples into one LeverRow (sum tokens, mean precision)."""
    n = len(samples) or 1
    return LeverRow(
        lever=lever,
        plane="recall_output",
        tokens_off=sum(s.tokens_off for s in samples),
        tokens_on=sum(s.tokens_on for s in samples),
        quality_off=sum(s.prec_off for s in samples) / n,
        quality_on=sum(s.prec_on for s in samples) / n,
    )


@contextmanager
def env_pins(overrides: dict[str, str]) -> Iterator[None]:
    """Temporarily set MEMO_* env vars, restoring prior state on exit."""
    prior: dict[str, str | None] = {k: os.environ.get(k) for k in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for k, v in prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def render_block(
    hits: list[Any], env: dict[str, str], *, body_chars: int, token_budget: int
) -> str:
    """Render the recall block for `hits` under a lever's env pins.

    Applies L4 verbosity steering when MEMO_RECALL_VERBOSITY_LEVEL > 0, so the
    lever's true effect on the injected block size is measured.
    """
    from memo.cli_recall_hook import maybe_inject_verbosity_steering
    from memo.flags_recall import flag_recall_verbosity_level
    from memo.recall_logic import render_recall_context

    with env_pins(env):
        block = render_recall_context(
            hits, [], turn=2, body_chars=body_chars, token_budget=token_budget
        )
        level = flag_recall_verbosity_level()
        if level > 0:
            block = maybe_inject_verbosity_steering(block, level)
        return block


# The P1 levers to measure. Env-expressible knobs that transform the injected
# recall block. Baseline (OFF) is the empty env; each lever is its ON delta.
RECALL_LEVERS: list[dict[str, Any]] = [
    {"name": "recall_format_compact", "env": {"MEMO_RECALL_FORMAT": "compact"}},
    {"name": "verbosity_steer_L2", "env": {"MEMO_RECALL_VERBOSITY_LEVEL": "2"}},
    {"name": "body_chars_200", "env": {"MEMO_RECALL_BODY_CHARS": "200"}},
    {"name": "top_k_2", "env": {"MEMO_RECALL_TOP_K": "2"}},
    {"name": "intra_dedup_on", "env": {"MEMO_RECALL_INTRA_DEDUP": "1"}},
]


CAPTURE_CORPUS_SCHEMA = "memo.token_corpus.v1"


@dataclass
class CaptureCase:
    name: str
    rows: list[Any]
    must_keep_index: int


@dataclass
class P2Sample:
    tokens_off: int
    tokens_on: int
    survived: bool
    rows_survived: int  # new: count of original rows that survived
    rows_total: int  # new: total original rows


def load_capture_corpus(path: Path) -> list[CaptureCase]:
    """Load the committed P2 capture corpus (schema memo.token_corpus.v1)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise ValueError(f"capture corpus {path} must be an object with a `cases` list")
    cases: list[CaptureCase] = []
    for c in raw["cases"]:
        cases.append(
            CaptureCase(
                name=str(c["name"]),
                rows=list(c["rows"]),
                must_keep_index=int(c["must_keep_index"]),
            )
        )
    return cases


def _count_rows_survived(original_rows: list[Any], crushed_content: str) -> int:
    """Count how many of the original rows appear in the crushed output."""
    try:
        arr = json.loads(crushed_content)
    except (json.JSONDecodeError, TypeError):
        return 0
    if not isinstance(arr, list):
        return 0
    return sum(1 for row in original_rows if row in arr)


def measure_crush_case(
    case: CaptureCase, crush_fn: Callable[[str], tuple[str, str | None]]
) -> P2Sample:
    """One capture case: token delta + multi-row survival quality."""
    original = json.dumps(case.rows, ensure_ascii=False)
    crushed, _hash = crush_fn(original)
    survived = _row_survived(case.rows[case.must_keep_index], crushed)
    rows_survived = _count_rows_survived(case.rows, crushed)
    return P2Sample(
        tokens_off=count_tokens(original),
        tokens_on=count_tokens(crushed),
        survived=survived,
        rows_survived=rows_survived,
        rows_total=len(case.rows),
    )


def _row_survived(row: Any, crushed_content: str) -> bool:
    try:
        arr = json.loads(crushed_content)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(arr, list) and row in arr


def aggregate_capture(lever: str, samples: list[P2Sample]) -> LeverRow:
    """Fold per-case P2 samples into one LeverRow. Quality = mean row survival fraction."""
    n = len(samples) or 1
    total_rows = sum(s.rows_total for s in samples) or 1
    survived_rows = sum(s.rows_survived for s in samples)
    return LeverRow(
        lever=lever,
        plane="capture",
        tokens_off=sum(s.tokens_off for s in samples),
        tokens_on=sum(s.tokens_on for s in samples),
        quality_off=1.0,
        quality_on=survived_rows / total_rows,
    )


def gate_metrics(rows: list[LeverRow]) -> dict[str, dict[str, float | bool]]:
    """Per-lever snapshot the gate tracks: saved_frac, quality_delta, passed."""
    return {
        r.lever: {
            "saved_frac": round(r.saved_frac, 4),
            "quality_delta": round(r.quality_delta, 4),
            "passed": r.passed,
        }
        for r in rows
    }


@dataclass
class GateResult:
    passed: bool
    message: str
    regressions: list[str]


def check_gate(rows: list[LeverRow], baseline: dict, *, tol: float = 1e-9) -> GateResult:
    """FAIL if any lever that was PASSING at baseline regressed (now fails,
    disappeared, or its token saving shrank beyond `tol`)."""
    cur = gate_metrics(rows)
    regressions: list[str] = []
    for lever, base in baseline.items():
        if not isinstance(base, dict) or not base.get("passed"):
            continue  # only guard previously-passing levers
        c = cur.get(lever)
        if c is None:
            regressions.append(f"{lever}: missing from current run")
        elif not c["passed"]:
            regressions.append(f"{lever}: was passing, now FAIL")
        elif float(c["saved_frac"]) < float(base["saved_frac"]) - tol:
            regressions.append(
                f"{lever}: saving dropped {float(base['saved_frac']):.3f}"
                f"→{float(c['saved_frac']):.3f}"
            )
    passed = not regressions
    message = "PASS — no token/quality regression" if passed else "FAIL — " + "; ".join(regressions)
    return GateResult(passed, message, regressions)


def run_all(
    *,
    prompts: list[Any],
    search: Callable[[str], list[Any]],
    corpus: list[CaptureCase],
    crush_fn: Callable[[str], tuple[str, str | None]],
    k: int = 5,
    body_chars: int = 200,
    token_budget: int = 400,
) -> list[LeverRow]:
    """Measure every P1 recall lever and the P2 crusher over the given corpus.

    `search`/`crush_fn` are injected so tests run without MLX; the CLI wires the
    real live-index search and the real `maybe_crush_json_capture`.
    """
    rows: list[LeverRow] = []

    # --- P1: recall-output levers ---------------------------------------
    hits_by_prompt = {p.text: (search(p.text) or [])[:k] for p in prompts}
    for lever in RECALL_LEVERS:
        samples: list[P1Sample] = []
        for p in prompts:
            hits = hits_by_prompt[p.text]
            block_off = render_block(hits, {}, body_chars=body_chars, token_budget=token_budget)
            block_on = render_block(
                hits, lever["env"], body_chars=body_chars, token_budget=token_budget
            )
            samples.append(measure_recall_sample(block_off, block_on, p.expect_ids))
        rows.append(aggregate_recall(lever["name"], samples))

    # --- P2: capture crusher --------------------------------------------
    p2 = [measure_crush_case(c, crush_fn) for c in corpus]
    if p2:
        rows.append(aggregate_capture("crusher_L1", p2))
    return rows
