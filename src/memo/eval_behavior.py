"""Behavior eval — does a recalled memory actually steer the answer?

`memo eval recall` measures *retrieval*: did the right memory surface. Nothing
measures the step after it. A memory can surface in top-1 and still be ignored,
re-litigated, or answered around — and every gate stays green, because the
retrieval metrics were satisfied the moment it appeared in the block.

This harness closes that loop:

1. Seed an isolated store with the scenario's memories (a real store, real
   embeddings, real index — nothing is mocked).
2. Run the **real** ``memo recall-hook`` as a subprocess against that store, the
   same way Claude Code runs it, and take its ``additionalContext`` verbatim.
   No ranking is reimplemented here, so the harness cannot drift from the hook.
3. Feed ``[injected block + prompt]`` to a model and judge the answer against
   the scenario's gates.

**What this measures, precisely.** The model in step 3 is memo's own local LLM,
not Claude. So a failure does not prove "Claude would ignore this memory" — it
proves the *injected block was not strong enough to steer a competent model*.
That is squarely memo's responsibility: block formatting, labelling, ordering,
truncation and token budget are all memo-side. Read a red gate as "the payload
under-steers", never as a claim about a specific agent.

Scenario corpus: ``eval/behavior_scenarios.json``
(schema ``memo.eval_behavior.scenario.v1``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

SCHEMA_VERSION = "memo.eval_behavior.scenario.v1"

RECALL_GATE_KINDS = frozenset({"must_recall", "must_not_recall"})
ANSWER_GATE_KINDS = frozenset(
    {"answer_must_contain_any", "answer_must_not_contain_any", "semantic"}
)
GATE_KINDS = RECALL_GATE_KINDS | ANSWER_GATE_KINDS

# The judge and the answerer both default to the *helper* model, never the 30B
# generation model: a scenario run already holds the embedder resident, and
# stacking a 30B answerer plus a 30B judge on top of it is the exact residency
# mix that OOM'd this machine before.
DEFAULT_ANSWER_MODEL_ENV = "MEMO_HELPER_MODEL"

_JUDGE_SYSTEM = (
    "You judge whether a claim about an answer is true. "
    "Reply with exactly one word: YES or NO. No explanation."
)


class Answerer(Protocol):
    def __call__(self, context: str, prompt: str) -> str: ...


class Judge(Protocol):
    def __call__(self, answer: str, statement: str) -> bool: ...


@dataclass(frozen=True)
class SeedMemory:
    title: str
    content: str
    type: str = "note"
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Gate:
    kind: str
    seed_index: int | None = None
    patterns: tuple[str, ...] = ()
    statement: str = ""

    @property
    def is_answer_gate(self) -> bool:
        return self.kind in ANSWER_GATE_KINDS


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    why: str
    prompt: str
    seed_memories: tuple[SeedMemory, ...]
    gates: tuple[Gate, ...]

    @property
    def answer_gates(self) -> tuple[Gate, ...]:
        return tuple(g for g in self.gates if g.is_answer_gate)


@dataclass
class GateResult:
    gate: Gate
    passed: bool
    detail: str = ""


@dataclass
class ScenarioResult:
    scenario_id: str
    recall_block: str
    answer: str
    gates: list[GateResult] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and all(g.passed for g in self.gates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "passed": self.passed,
            "error": self.error,
            "gates": [
                {"kind": g.gate.kind, "passed": g.passed, "detail": g.detail} for g in self.gates
            ],
        }


# --- loading -----------------------------------------------------------------


def _gate_from_dict(raw: dict) -> Gate:
    kind = str(raw.get("kind") or "")
    if kind not in GATE_KINDS:
        raise ValueError(f"unknown gate kind {kind!r}; expected one of {sorted(GATE_KINDS)}")
    seed_index = raw.get("seed_index")
    return Gate(
        kind=kind,
        seed_index=int(seed_index) if seed_index is not None else None,
        patterns=tuple(str(p) for p in (raw.get("patterns") or ())),
        statement=str(raw.get("statement") or ""),
    )


def _scenario_from_dict(raw: dict, source: str) -> Scenario:
    scenario_id = str(raw.get("scenario_id") or "")
    if not scenario_id:
        raise ValueError(f"{source}: a scenario is missing scenario_id")
    seeds = tuple(
        SeedMemory(
            title=str(s.get("title") or ""),
            content=str(s.get("content") or ""),
            type=str(s.get("type") or "note"),
            tags=tuple(str(t) for t in (s.get("tags") or ())),
        )
        for s in (raw.get("seed_memories") or [])
    )
    if not seeds:
        raise ValueError(f"{source}: scenario {scenario_id} seeds no memories")
    gates = tuple(_gate_from_dict(g) for g in (raw.get("gates") or []))
    if not any(g.is_answer_gate for g in gates):
        # Without an answer-layer gate the scenario only re-tests retrieval,
        # which `memo eval recall` already covers against a far larger corpus.
        raise ValueError(
            f"{source}: scenario {scenario_id} has no answer-layer gate "
            f"({sorted(ANSWER_GATE_KINDS)}) — that is retrieval, not behavior"
        )
    for gate in gates:
        if gate.kind in RECALL_GATE_KINDS:
            if gate.seed_index is None or not 0 <= gate.seed_index < len(seeds):
                raise ValueError(
                    f"{source}: scenario {scenario_id} has a {gate.kind} gate whose "
                    f"seed_index {gate.seed_index!r} is not a seeded memory"
                )
        elif gate.kind == "semantic":
            if not gate.statement:
                raise ValueError(f"{source}: scenario {scenario_id} has an empty semantic gate")
        elif not gate.patterns:
            raise ValueError(
                f"{source}: scenario {scenario_id} has a {gate.kind} gate with no patterns"
            )
    return Scenario(
        scenario_id=scenario_id,
        title=str(raw.get("title") or scenario_id),
        why=str(raw.get("why") or ""),
        prompt=str(raw.get("prompt") or ""),
        seed_memories=seeds,
        gates=gates,
    )


def load_scenarios(path: Path) -> list[Scenario]:
    """Parse a scenario corpus. Raises ValueError on a malformed one."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read scenarios {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("scenarios"), list):
        raise ValueError(f"{path} must be an object with a `scenarios` list")
    scenarios = [_scenario_from_dict(s, str(path)) for s in raw["scenarios"]]
    if not scenarios:
        raise ValueError(f"{path} has no scenarios")
    return scenarios


# --- the seeded store + the real hook ----------------------------------------


def seed_store(scenario: Scenario, workdir: Path) -> list[str]:
    """Write the scenario's memories into a fresh store under `workdir`.

    Returns the saved ids, positionally aligned with `scenario.seed_memories`.
    """
    from memo.config import Config
    from memo.memory import Memory

    data_dir = workdir / "data"
    state_dir = workdir / "state"
    data_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    prior = {k: os.environ.get(k) for k in ("MEMO_DATA_DIR", "MEMO_STATE_DIR")}
    os.environ["MEMO_DATA_DIR"] = str(data_dir)
    os.environ["MEMO_STATE_DIR"] = str(state_dir)
    try:
        mem = Memory(Config.from_env())
        try:
            return [
                mem.save(
                    content=seed.content,
                    title=seed.title,
                    type_=seed.type,
                    tags=list(seed.tags),
                ).id
                for seed in scenario.seed_memories
            ]
        finally:
            mem.close()
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_recall_hook(prompt: str, workdir: Path, *, timeout: float = 60.0) -> str:
    """Run the REAL `memo recall-hook` against the seeded store.

    Deliberately a subprocess of this interpreter's `memo.cli`, not an
    in-process call and not a reimplementation: the hook's own ranking,
    injection filters, token budget and formatting are exactly what we want to
    measure, and a subprocess is also how Claude Code invokes it.
    """
    # The hook downgrades to BM25 unless `state_dir/.prewarm_ts` is fresh — its
    # subprocess-side guard against paying a cold MLX load inside the 5s budget.
    # A per-scenario temp state_dir never has that signal, so without this stamp
    # the eval would only ever measure the cold-start downgrade: a degraded mode
    # a real session does not run in (SessionStart's `memo prewarm` stamps it).
    # Combined with the warm-daemon socket below, the signal is accurate rather
    # than a cheat — the embedder really is warm.
    state_dir = workdir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / ".prewarm_ts").write_text(str(time.time()), encoding="utf-8")

    env = {
        **os.environ,
        "MEMO_DATA_DIR": str(workdir / "data"),
        "MEMO_STATE_DIR": str(state_dir),
        "MEMO_NONINTERACTIVE": "1",
        # Serve embeds off the warm recall daemon when it is up, exactly as a
        # live session does; falls back to an in-process load when it is not.
        "MEMO_EMBEDDER_VIA_DAEMON": "1",
        # Never let a socket hiccup silently degrade the measurement. With
        # REQUIRE_DAEMON on (the ambient setting on this machine) an
        # unreachable daemon makes embed_query fail and the hook falls back to
        # BM25 — so the eval would report a vec-mode result it never measured.
        # A live daemon can still read "unreachable" purely from a client
        # timeout too short for the 4B profile, hence the explicit timeout.
        "MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON": "0",
        # Well above the 30s query default — a fixed harness decision, not a
        # knob: an eval is not latency-bound, and every MEMO_* knob has to be a
        # registered FlagSpec rather than an inline os.environ read.
        "MEMO_EMBEDDER_CLIENT_TIMEOUT": "120",
        # The eval is about the payload, not about a session's adaptive state.
        "MEMO_RECALL_DISABLE": "0",
    }
    payload = json.dumps({"prompt": prompt, "session_id": "eval-behavior"})
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "memo.cli", "recall-hook"],
            input=payload,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"recall-hook timed out after {timeout}s") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"recall-hook exited {proc.returncode}: {proc.stderr.strip()[:400]}")
    try:
        out = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"recall-hook emitted non-JSON: {proc.stdout[:200]!r}") from exc
    specific = out.get("hookSpecificOutput") or {}
    return str(specific.get("additionalContext") or "")


# --- default MLX-backed answerer + judge --------------------------------------


def _model_name() -> str:
    return os.environ.get(DEFAULT_ANSWER_MODEL_ENV) or "mlx-community/Qwen3-4B-4bit"


def default_answerer(context: str, prompt: str) -> str:
    """Answer `prompt` with `context` injected exactly as the hook emitted it."""
    from memo.llm import MLXChat

    messages = [
        {
            "role": "system",
            "content": (
                "You are a coding assistant. The context below was injected by the "
                "user's memory system and is authoritative: prefer it over your own "
                "assumptions, and contradict it only explicitly.\n\n" + context
            ),
        },
        {"role": "user", "content": prompt},
    ]
    # Generous: a truncated answer fails `answer_must_contain_any` for any fact
    # that would have appeared past the cut, which reads as "the payload did not
    # steer" when it only means "the answer was cut off".
    reply = MLXChat().chat(_model_name(), messages, {"temperature": 0.0, "max_tokens": 1200})
    return str((reply.get("message") or {}).get("content") or "")


def default_judge(answer: str, statement: str) -> bool:
    from memo.llm import MLXChat

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {
            "role": "user",
            "content": f"ANSWER:\n{answer}\n\nCLAIM: {statement}\n\nIs the claim true?",
        },
    ]
    reply = MLXChat().chat(_model_name(), messages, {"temperature": 0.0, "max_tokens": 8})
    verdict = str((reply.get("message") or {}).get("content") or "").strip().upper()
    return verdict.startswith("YES")


# --- gate evaluation ----------------------------------------------------------


def _eval_recall_gate(gate: Gate, block: str, seed_ids: list[str]) -> GateResult:
    assert gate.seed_index is not None  # validated at load
    memory_id = seed_ids[gate.seed_index]
    # The block renders short ids; match on a prefix so a full-id render and a
    # truncated one both count.
    present = memory_id[:8] in block
    if gate.kind == "must_recall":
        return GateResult(
            gate, present, "" if present else f"{memory_id[:8]} absent from the block"
        )
    return GateResult(
        gate, not present, f"{memory_id[:8]} present but must not be" if present else ""
    )


def _eval_answer_gate(gate: Gate, answer: str, judge: Judge) -> GateResult:
    folded = answer.casefold()
    if gate.kind == "answer_must_contain_any":
        hit = next((p for p in gate.patterns if p.casefold() in folded), None)
        return GateResult(
            gate, hit is not None, "" if hit else f"none of {list(gate.patterns)} appear"
        )
    if gate.kind == "answer_must_not_contain_any":
        hit = next((p for p in gate.patterns if p.casefold() in folded), None)
        return GateResult(gate, hit is None, f"forbidden {hit!r} appears" if hit else "")
    verdict = judge(answer, gate.statement)
    return GateResult(gate, verdict, "" if verdict else f"judge rejected: {gate.statement}")


def run_scenario(
    scenario: Scenario,
    *,
    answerer: Answerer | None = None,
    judge: Judge | None = None,
    recall_only: bool = False,
    workdir: Path | None = None,
) -> ScenarioResult:
    """Seed, recall through the real hook, answer, and score every gate."""
    result = ScenarioResult(scenario_id=scenario.scenario_id, recall_block="", answer="")
    owned: tempfile.TemporaryDirectory | None = None
    if workdir is None:
        owned = tempfile.TemporaryDirectory(prefix="memo-eval-behavior-")
        workdir = Path(owned.name)
    try:
        seed_ids = seed_store(scenario, workdir)
        result.recall_block = run_recall_hook(scenario.prompt, workdir)

        for gate in scenario.gates:
            if gate.kind in RECALL_GATE_KINDS:
                result.gates.append(_eval_recall_gate(gate, result.recall_block, seed_ids))

        if recall_only:
            return result

        result.answer = (answerer or default_answerer)(result.recall_block, scenario.prompt)
        for gate in scenario.answer_gates:
            result.gates.append(_eval_answer_gate(gate, result.answer, judge or default_judge))
        return result
    except (RuntimeError, OSError, ValueError) as exc:
        # Surfaced, never swallowed: an errored scenario is not a passing one.
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        if owned is not None:
            owned.cleanup()


def run_scenarios(
    scenarios: list[Scenario],
    *,
    answerer: Answerer | None = None,
    judge: Judge | None = None,
    recall_only: bool = False,
) -> list[ScenarioResult]:
    return [
        run_scenario(s, answerer=answerer, judge=judge, recall_only=recall_only) for s in scenarios
    ]
