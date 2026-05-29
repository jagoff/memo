"""Recall-quality evaluation — precision@K / noise@K over the live corpus.

Pure logic, no CLI/IO concerns (the CLI lives in `cli_eval.py`, the legacy
script in `scripts/recall_eval.py` is a thin shim over this module). For each
labeled prompt we run `Memory.search` under several configs, apply the same
post-filters the recall-hook uses (similarity floor, optional archive
exclusion), take the top-K, and classify each hit as relevant / noise. We
report precision@K (over the prompts that have a known answer), noise@K (over
all prompts), and p50 search latency per config — so the default recall
mode/floor is chosen from data, not by eye.

Relevance can be judged two ways per prompt:
  * `expect_ids` — ground truth: a hit is relevant iff its id matches (prefix
    match ≥ 8 hex chars, so short ids in label files work).
  * otherwise a term heuristic: the hit's title/tags/path/body-head must
    contain one of `relevant_terms` and must not be `noise`.

Label sets load from JSON (`load_labels`); `DEFAULT_LABELS` is a built-in
example tuned to the author's stack corpus — replace it with your own via
`memo eval recall --labels mylabels.json` for meaningful numbers.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LABELS_SCHEMA = "memo.eval_recall.labels.v1"

_ID_RE = re.compile(r"[0-9a-f]{8,64}")


@dataclass
class Prompt:
    text: str
    relevant: bool = False  # contributes to precision@K when True
    expect_ids: list[str] = field(default_factory=list)


@dataclass
class LabelSet:
    prompts: list[Prompt]
    relevant_terms: set[str] = field(default_factory=set)
    noise_tags: set[str] = field(default_factory=set)
    noise_path_fragments: tuple[str, ...] = ()
    session_context: str = ""

    def fingerprint(self) -> str:
        """Stable hash of the label content, for cache invalidation."""
        import hashlib

        payload = json.dumps(
            {
                "prompts": [(p.text, p.relevant, sorted(p.expect_ids)) for p in self.prompts],
                "relevant_terms": sorted(self.relevant_terms),
                "noise_tags": sorted(self.noise_tags),
                "noise_path_fragments": list(self.noise_path_fragments),
                "session_context": self.session_context,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --- Built-in example label set (author's stack corpus) ----------------------

DEFAULT_LABELS = LabelSet(
    prompts=[
        Prompt("cómo está la arquitectura del stack synapse memflow memo", relevant=True),
        Prompt("qué falta para retomar Tier 2 del stack de consciencia", relevant=True),
        Prompt("dónde está el registro de puertos y daemons del stack", relevant=True),
        Prompt("cómo funciona el recall de memo y por qué trae ruido", relevant=True),
        Prompt("qué decidí sobre el gate de aliases memory_* en memflow", relevant=True),
        Prompt("qué reunión tengo agendada para mañana", relevant=False),
        Prompt("receta para hacer una tarta de manzana", relevant=False),
    ],
    relevant_terms={
        "synapse", "memflow", "memo", "3-pilares", "stack", "tier-2", "tier 2",
        "arquitectura", "recall", "provenance", "consciousness", "consciencia",
        "onboarding", "daemon", "mcp",
    },
    noise_tags={
        "04-archive", "old", "moka", "foda", "swot", "aws", "aws-tagging",
        "hr", "1a1", "companies",
    },
    noise_path_fragments=("inactive/", "/04-archive/", "04-archive/", "/old/", "/companies/"),
    session_context=(
        "trabajando en el stack de consciencia: synapse, memflow, memo; "
        "git, recall, provenance, tier 2"
    ),
)


def load_labels(path: Path) -> LabelSet:
    """Parse a label-set JSON file. Raises ValueError on a malformed file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read label set {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("prompts"), list):
        raise ValueError(f"label set {path} must be an object with a `prompts` list")
    prompts: list[Prompt] = []
    for p in raw["prompts"]:
        if isinstance(p, str):
            prompts.append(Prompt(text=p))
        elif isinstance(p, dict) and p.get("text"):
            prompts.append(Prompt(
                text=str(p["text"]),
                relevant=bool(p.get("relevant", False)),
                expect_ids=[str(x) for x in (p.get("expect_ids") or [])],
            ))
    if not prompts:
        raise ValueError(f"label set {path} has no usable prompts")
    return LabelSet(
        prompts=prompts,
        relevant_terms={str(t).lower() for t in (raw.get("relevant_terms") or [])},
        noise_tags={str(t).lower() for t in (raw.get("noise_tags") or [])},
        noise_path_fragments=tuple(str(f) for f in (raw.get("noise_path_fragments") or [])),
        session_context=str(raw.get("session_context") or ""),
    )


# --- Configs to compare ------------------------------------------------------


@dataclass
class Cfg:
    name: str
    mode: str
    floor: float
    exclude_archived: bool
    context: bool = False


def default_configs() -> list[Cfg]:
    return [
        Cfg("A vec/0.60/keep", "vec", 0.60, exclude_archived=False),
        Cfg("B vec/0.72/excl", "vec", 0.72, exclude_archived=True),
        Cfg("C hyb/0.40/excl", "hybrid", 0.40, exclude_archived=True),
        Cfg("D hyb/0.40/ctx", "hybrid", 0.40, exclude_archived=True, context=True),
    ]


# --- Classification ----------------------------------------------------------


def _is_noise(rec: Any, labels: LabelSet) -> bool:
    tags = {str(t).lower() for t in (getattr(rec, "tags", None) or [])}
    if tags & labels.noise_tags:
        return True
    p = (getattr(rec, "path", None) or "").lower()
    return any(frag in p for frag in labels.noise_path_fragments)


def _id_matches(hit_id: str, expect_ids: list[str]) -> bool:
    a = (hit_id or "").strip().lower()
    for raw in expect_ids:
        e = (raw or "").strip().lower()
        if a == e:
            return True
        shorter = min(len(a), len(e))
        if shorter >= 8 and (a.startswith(e) or e.startswith(a)):
            return True
    return False


def _is_relevant(rec: Any, prompt: Prompt, labels: LabelSet) -> bool:
    if prompt.expect_ids:
        return _id_matches(getattr(rec, "id", ""), prompt.expect_ids)
    if _is_noise(rec, labels):
        return False
    hay = " ".join([
        getattr(rec, "title", "") or "",
        " ".join(getattr(rec, "tags", None) or []),
        getattr(rec, "path", "") or "",
        (getattr(rec, "body", "") or "")[:200],
    ]).lower()
    return any(term in hay for term in labels.relevant_terms)


# --- Run ---------------------------------------------------------------------


@dataclass
class Row:
    config: str
    precision_at_k: float = 0.0
    noise_at_k: float = 0.0
    latency_ms_p50: float = 0.0
    detail: list[dict[str, Any]] = field(default_factory=list)


def _scored_prompts(labels: LabelSet) -> int:
    """How many prompts contribute to precision (have a known answer)."""
    return sum(1 for p in labels.prompts if p.relevant or p.expect_ids)


def run_config(mem: Any, cfg: Cfg, k: int, labels: LabelSet) -> Row:
    lat: list[float] = []
    prec_hits = 0
    prec_total = 0
    noise_hits = 0
    detail: list[dict[str, Any]] = []
    n_prompts = len(labels.prompts) or 1
    for prompt in labels.prompts:
        scored = prompt.relevant or bool(prompt.expect_ids)
        query = f"{labels.session_context}\n{prompt.text}" if cfg.context and labels.session_context else prompt.text
        t0 = time.time()
        hits = mem.search(query, limit=k * 4, mode=cfg.mode)
        lat.append((time.time() - t0) * 1000)
        hits = [h for h in hits if h.score is None or h.score >= cfg.floor]
        if cfg.exclude_archived:
            hits = [h for h in hits if not _is_noise(h, labels)]
        top = hits[:k]
        noise_hits += sum(1 for h in top if _is_noise(h, labels))
        if scored:
            prec_total += k
            prec_hits += sum(1 for h in top if _is_relevant(h, prompt, labels))
        detail.append({
            "prompt": prompt.text[:48],
            "scored": scored,
            "top": [
                {"title": (h.title or "")[:40], "score": round(h.score or 0, 3),
                 "noise": _is_noise(h, labels), "relevant": _is_relevant(h, prompt, labels)}
                for h in top
            ],
        })
    lat.sort()
    return Row(
        config=cfg.name,
        precision_at_k=round(prec_hits / prec_total, 3) if prec_total else 0.0,
        noise_at_k=round(noise_hits / (n_prompts * k), 3),
        latency_ms_p50=round(lat[len(lat) // 2], 1) if lat else 0.0,
        detail=detail,
    )


def evaluate(mem: Any, *, k: int = 3, labels: LabelSet | None = None,
             configs: list[Cfg] | None = None) -> list[Row]:
    labels = labels or DEFAULT_LABELS
    configs = configs or default_configs()
    return [run_config(mem, cfg, k, labels) for cfg in configs]


# --- Reporting ---------------------------------------------------------------


def rows_to_table(rows: list[Row], k: int) -> str:
    lines = [
        f"\nRecall eval — precision@{k} (answerable prompts) / noise@{k} (all)\n",
        f"{'config':<18} {'prec@k':>7} {'noise@k':>8} {'p50 ms':>8}",
        "-" * 45,
    ]
    for r in rows:
        lines.append(f"{r.config:<18} {r.precision_at_k:>7} {r.noise_at_k:>8} {r.latency_ms_p50:>8}")
    lines.append("\nHigher prec@k + lower noise@k is better. Baseline = first config.")
    return "\n".join(lines)


# The recall hook (UserPromptSubmit) has a ~5s end-to-end budget; cold MLX
# load eats ~2s, leaving ~3s for embed + search + format. A config whose p50
# search latency exceeds this can't be the *hook's* default mode even if it
# wins on precision — flag that tradeoff instead of recommending blindly.
_HOOK_SEARCH_BUDGET_MS = 3000.0


def recommend(rows: list[Row]) -> str:
    """Concrete next-step suggestion: pick the config with the best
    precision (tie-break: lower noise), and if it beats the baseline, map it
    to the MEMO_* knobs that reproduce it. Warns when the winner's p50 search
    latency would blow the recall-hook budget."""
    if not rows:
        return "no configs evaluated."
    baseline = rows[0]
    best = max(rows, key=lambda r: (r.precision_at_k, -r.noise_at_k))
    if best.config == baseline.config:
        return "Baseline config already wins — no knob change recommended."
    dp = best.precision_at_k - baseline.precision_at_k
    dn = best.noise_at_k - baseline.noise_at_k
    cfg = next((c for c in default_configs() if c.name == best.config), None)
    knobs = ""
    if cfg is not None:
        knobs = (f"  export MEMO_RECALL_MODE={cfg.mode}\n"
                 f"  export MEMO_RECALL_MIN_SIM={cfg.floor}")
        if cfg.exclude_archived:
            knobs += "\n  (and keep archive exclusion on in the recall hook)"
    out = (f"Best config: {best.config} "
           f"(prec {dp:+.3f}, noise {dn:+.3f} vs baseline).\n{knobs}")
    if best.latency_ms_p50 > _HOOK_SEARCH_BUDGET_MS:
        out += (f"\n  ⚠ p50 search {best.latency_ms_p50:.0f}ms exceeds the "
                f"~{_HOOK_SEARCH_BUDGET_MS:.0f}ms recall-hook budget — best for "
                f"`memo ask`/chat, but keep a faster mode for the hook.")
    return out


def fingerprint_corpus(mem: Any) -> str:
    """Cheap corpus identity for cache keying: record count + db mtime."""
    try:
        count = mem.store.count()
    except Exception:
        count = -1
    try:
        mtime = int(Path(mem.cfg.db_path).stat().st_mtime)
    except Exception:
        mtime = 0
    return f"{count}:{mtime}"
