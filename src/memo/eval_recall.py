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
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# v1 is additive: the optional per-prompt `project` field (harvested from
# grounding.log) is part of v1 — old label files without it keep loading.
LABELS_SCHEMA = "memo.eval_recall.labels.v1"

_ID_RE = re.compile(r"[0-9a-f]{8,64}")


@dataclass
class Prompt:
    text: str
    relevant: bool = False  # contributes to precision@K when True
    expect_ids: list[str] = field(default_factory=list)
    expect_associative_ids: tuple[str, ...] = ()  # ids reachable via graph hop, not pure vec/BM25
    # Project context the prompt was asked from — harvested from grounding.log's
    # `project` field (the `project:<slug>` tag `current_project_tag` produced)
    # or hand-written (a bare name also works). None = no project context;
    # schema-additive: old label files simply lack the key.
    project: str | None = None


# Alias so tests/code can import `Label` as the per-prompt record type.
Label = Prompt


def _label_from_dict(d: dict) -> Label:
    """Build a Label from a JSON dict.

    Accepts ``"prompt"`` or ``"text"`` as the text key so that both
    new-style (``"prompt"``) and existing (``"text"``) JSON files are
    supported.
    """
    text = str(d.get("prompt") or d.get("text") or "")
    return Prompt(
        text=text,
        relevant=bool(d.get("relevant", False)),
        expect_ids=[str(x) for x in (d.get("expect_ids") or [])],
        expect_associative_ids=tuple(str(x) for x in (d.get("expect_associative_ids") or ())),
        project=str(d["project"]) if d.get("project") else None,
    )


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
                "prompts": [
                    (
                        p.text,
                        p.relevant,
                        sorted(p.expect_ids),
                        sorted(p.expect_associative_ids),
                        p.project,
                    )
                    for p in self.prompts
                ],
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
        "synapse",
        "memflow",
        "memo",
        "3-pilares",
        "stack",
        "tier-2",
        "tier 2",
        "arquitectura",
        "recall",
        "provenance",
        "consciousness",
        "consciencia",
        "onboarding",
        "daemon",
        "mcp",
    },
    noise_tags={
        "04-archive",
        "old",
        "moka",
        "foda",
        "swot",
        "aws",
        "aws-tagging",
        "hr",
        "1a1",
        "companies",
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
        elif isinstance(p, dict) and (p.get("text") or p.get("prompt")):
            prompts.append(_label_from_dict(p))
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
    # RankKnobs field -> value pins, applied ON TOP of the live flag/overlay
    # resolution (knobs_from_flags(overrides=...)). Grid configs use this to
    # probe MMR/synthesis variants without touching the environment.
    knob_overrides: dict[str, Any] | None = None
    # Apply the hook's post-rank injection filters (MEMO_RECALL_SKIP_BELOW
    # floor + MEMO_RECALL_GAP_THRESHOLD trim) to the ranked list before
    # scoring. Default False keeps the nightly trend comparable.
    injection_fidelity: bool = False


def default_configs() -> list[Cfg]:
    return [
        Cfg("A vec/0.60/keep", "vec", 0.60, exclude_archived=False),
        Cfg("B vec/0.72/excl", "vec", 0.72, exclude_archived=True),
        Cfg("C hyb/0.40/excl", "hybrid", 0.40, exclude_archived=True),
        Cfg("D hyb/0.40/ctx", "hybrid", 0.40, exclude_archived=True, context=True),
        # MMR / synthesis-boost variants vs the A baseline (same mode/floor/keep,
        # only the knob differs — the delta is attributable to the knob).
        Cfg("E mmr/0.3", "vec", 0.60, exclude_archived=False, knob_overrides={"mmr_lambda": 0.3}),
        Cfg("F mmr/0.5", "vec", 0.60, exclude_archived=False, knob_overrides={"mmr_lambda": 0.5}),
        Cfg("G mmr/0.7", "vec", 0.60, exclude_archived=False, knob_overrides={"mmr_lambda": 0.7}),
        Cfg(
            "H synth/0.05",
            "vec",
            0.60,
            exclude_archived=False,
            knob_overrides={"synthesis_boost": 0.05},
        ),
        Cfg(
            "I synth/0.10",
            "vec",
            0.60,
            exclude_archived=False,
            knob_overrides={"synthesis_boost": 0.10},
        ),
    ]


def select_configs(
    names: list[str] | tuple[str, ...] | None = None, *, quick: bool = False
) -> list[Cfg]:
    """Return the eval configs requested by CLI/user input.

    Names accept either the short letter (``A``/``B``/``C``/``D``) or the full
    display name. ``quick`` defaults to the single fast baseline config unless
    explicit names were supplied.
    """
    configs = default_configs()
    requested = list(names or [])
    if quick and not requested:
        requested = ["A"]
    if not requested:
        return configs

    by_key: dict[str, Cfg] = {}
    for cfg in configs:
        short = cfg.name.split(" ", 1)[0].lower()
        by_key[short] = cfg
        by_key[cfg.name.lower()] = cfg

    selected: list[Cfg] = []
    seen: set[str] = set()
    for raw in requested:
        key = raw.strip().lower()
        selected_cfg = by_key.get(key)
        if selected_cfg is None:
            valid = ", ".join(c.name.split(" ", 1)[0] for c in configs)
            raise ValueError(f"unknown recall eval config {raw!r}; valid: {valid}")
        if selected_cfg.name not in seen:
            selected.append(selected_cfg)
            seen.add(selected_cfg.name)
    return selected


def limit_label_set(labels: LabelSet, max_prompts: int | None) -> LabelSet:
    """Return a label set capped to the first ``max_prompts`` prompts."""
    if max_prompts is None or max_prompts >= len(labels.prompts):
        return labels
    return LabelSet(
        prompts=labels.prompts[:max_prompts],
        relevant_terms=set(labels.relevant_terms),
        noise_tags=set(labels.noise_tags),
        noise_path_fragments=tuple(labels.noise_path_fragments),
        session_context=labels.session_context,
    )


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
    hay = " ".join(
        [
            getattr(rec, "title", "") or "",
            " ".join(getattr(rec, "tags", None) or []),
            getattr(rec, "path", "") or "",
            (getattr(rec, "body", "") or "")[:200],
        ]
    ).lower()
    return any(term in hay for term in labels.relevant_terms)


# --- Run ---------------------------------------------------------------------


@dataclass
class Row:
    config: str
    precision_at_k: float = 0.0
    noise_at_k: float = 0.0
    assoc_precision_at_k: float = 0.0  # fraction of expect_associative_ids the associative engine surfaces from the top-K seeds
    latency_ms_p50: float = 0.0
    detail: list[dict[str, Any]] = field(default_factory=list)


def _scored_prompts(labels: LabelSet) -> int:
    """How many prompts contribute to precision (have a known answer)."""
    return sum(1 for p in labels.prompts if p.relevant or p.expect_ids)


def _project_tag_for(project: str | None) -> str | None:
    """Normalize a label's ``project`` field to the stored tag format.

    Harvested labels already carry the exact tag ``current_project_tag``
    produced (``project:<slug>`` — grounding.log stores it verbatim), so a
    prefixed value passes through unchanged; a hand-written bare name is
    slugified and prefixed the same way ``current_project_tag`` would."""
    p = (project or "").strip()
    if not p:
        return None
    if p.startswith("project:"):
        return p
    from memo.project import slugify_project

    slug = slugify_project(p)
    return f"project:{slug}" if slug else None


ProgressCallback = Callable[[Cfg, int, int], None]


def run_config(
    mem: Any,
    cfg: Cfg,
    k: int,
    labels: LabelSet,
    *,
    progress: ProgressCallback | None = None,
) -> Row:
    lat: list[float] = []
    prec_hits = 0
    prec_total = 0
    noise_hits = 0
    assoc_hits = 0
    assoc_total = 0
    detail: list[dict[str, Any]] = []
    n_prompts = len(labels.prompts) or 1
    # Honest associative metric: measure what the associative engine actually
    # surfaces from the top-K seeds, not whether vector search already found the
    # graph-neighbor ids. Load the codegraph layer once (degrades to None).
    from memo.associative import associate
    from memo.recall_logic import (
        apply_injection_filters,
        knobs_from_flags,
        make_vec_cosine,
        rank_hits,
    )

    try:
        from memo import codegraph_loader

        _assoc_cg = codegraph_loader.load()[0]
    except Exception:
        _assoc_cg = None
    # Recall-faithful knobs: pin only the eval-specific fields (top_k=k,
    # min_sim=cfg.floor, min_body_chars=0, mode); every OTHER knob
    # (mmr_lambda, synthesis_boost, project/global boosts, contextual)
    # inherits the LIVE flag/overlay resolution — same as the hook — and
    # cfg.knob_overrides pins individual fields on top for grid variants.
    # No base project_tag/cwd: per-label project (when a label carries one)
    # is applied per prompt inside the loop; project-less labels rank with
    # project_tag=None (tiers stay inert).
    knobs = knobs_from_flags(
        top_k=k,
        min_sim=cfg.floor,
        min_body_chars=0,
        mode=cfg.mode,
        overrides=cfg.knob_overrides,
    )
    for index, prompt in enumerate(labels.prompts, start=1):
        if progress is not None:
            progress(cfg, index, len(labels.prompts))
        scored = prompt.relevant or bool(prompt.expect_ids)
        query = (
            f"{labels.session_context}\n{prompt.text}"
            if cfg.context and labels.session_context
            else prompt.text
        )
        t0 = time.time()
        hits = mem.search(query, limit=k * 4, mode=cfg.mode)
        lat.append((time.time() - t0) * 1000)
        # Rank exactly as the daemon does (shared rank_hits): dedup + the hybrid
        # true-cosine gate + the Phase-2 graph_boost seam — so the eval measures
        # the real ranking, not a hand-rolled floor filter. cfg.floor -> min_sim.
        vc = make_vec_cosine(mem, query) if cfg.mode == "hybrid" else None
        # Per-label project fidelity: a label harvested with a project context
        # ranks with project_tag set — the same 3-tier boosts the hook applies
        # from cwd — gated on project_boost > 0 exactly like knobs_from_flags'
        # cwd resolution. A cfg.knob_overrides pin of project_tag wins over the
        # label (overrides beat everything); project-less labels keep the base
        # knobs (project_tag=None).
        prompt_knobs = knobs
        p_tag = _project_tag_for(prompt.project)
        if p_tag and knobs.project_boost > 0 and "project_tag" not in (cfg.knob_overrides or {}):
            prompt_knobs = replace(knobs, project_tag=p_tag)
        ranked = rank_hits(hits, prompt_knobs, vec_cosine=vc)
        if cfg.injection_fidelity:
            # Hook-faithful injection: the same post-rank skip-below/gap
            # filters _recall_logic applies before injecting (shared helper).
            ranked = apply_injection_filters(ranked)
        if cfg.exclude_archived:
            ranked = [h for h in ranked if not _is_noise(h, labels)]
        top = ranked[:k]
        noise_hits += sum(1 for h in top if _is_noise(h, labels))
        if scored:
            prec_total += k
            prec_hits += sum(1 for h in top if _is_relevant(h, prompt, labels))
        if prompt.expect_associative_ids:
            assoc_total += len(prompt.expect_associative_ids)
            seed_ids = [getattr(h, "id", "") for h in top if getattr(h, "id", "")]
            assoc_ids = [
                a.id
                for a in associate(
                    seed_ids,
                    store=mem.graph,
                    codegraph_adj=_assoc_cg,
                    hops=2,
                    limit=20,
                    exclude_ids=frozenset(seed_ids),
                    min_activation=0.0,
                )
            ]
            for aid in prompt.expect_associative_ids:
                if any(_id_matches(a, [aid]) for a in assoc_ids):
                    assoc_hits += 1
        detail.append(
            {
                "prompt": prompt.text[:48],
                "scored": scored,
                "top": [
                    {
                        "title": (h.title or "")[:40],
                        "score": round(h.score or 0, 3),
                        "noise": _is_noise(h, labels),
                        "relevant": _is_relevant(h, prompt, labels),
                    }
                    for h in top
                ],
            }
        )
    lat.sort()
    return Row(
        config=cfg.name,
        precision_at_k=round(prec_hits / prec_total, 3) if prec_total else 0.0,
        noise_at_k=round(noise_hits / (n_prompts * k), 3) if (n_prompts * k) else 0.0,
        assoc_precision_at_k=round(assoc_hits / assoc_total, 3) if assoc_total else 0.0,
        latency_ms_p50=round(lat[len(lat) // 2], 1) if lat else 0.0,
        detail=detail,
    )


def evaluate(
    mem: Any,
    *,
    k: int = 3,
    labels: LabelSet | None = None,
    configs: list[Cfg] | None = None,
    progress: ProgressCallback | None = None,
) -> list[Row]:
    labels = labels or DEFAULT_LABELS
    configs = configs or default_configs()
    return [run_config(mem, cfg, k, labels, progress=progress) for cfg in configs]


# --- Reporting ---------------------------------------------------------------


def rows_to_table(rows: list[Row], k: int) -> str:
    lines = [
        f"\nRecall eval — precision@{k} (answerable) / noise@{k} (all) / assoc@{k} (graph-neighbor labels)\n",
        f"{'config':<18} {'prec@k':>7} {'noise@k':>8} {'assoc@k':>8} {'p50 ms':>8}",
        "-" * 55,
    ]
    for r in rows:
        lines.append(
            f"{r.config:<18} {r.precision_at_k:>7} {r.noise_at_k:>8} {r.assoc_precision_at_k:>8} {r.latency_ms_p50:>8}"
        )
    lines.append("\nHigher prec@k + lower noise@k is better. Baseline = first config.")
    lines.append(
        "assoc@k: fraction of graph-neighbor expect_associative_ids the associative "
        "engine surfaces from the top-K seeds (the path vector search alone missed)."
    )
    return "\n".join(lines)


# The recall hook (UserPromptSubmit) has a ~5s end-to-end budget; cold MLX
# load eats ~2s, leaving ~3s for embed + search + format. A config whose p50
# search latency exceeds this can't be the *hook's* default mode even if it
# wins on precision — flag that tradeoff instead of recommending blindly.
_HOOK_SEARCH_BUDGET_MS = 3000.0


def best_row(rows: list[Row]) -> Row:
    """The winning config: highest precision@K, tie-break lowest noise@K."""
    return max(rows, key=lambda r: (r.precision_at_k, -r.noise_at_k))


# RankKnobs field -> MEMO_* flag, so `recommend` can map a winning config's
# knob_overrides to the env exports that reproduce it.
_KNOB_FIELD_TO_FLAG = {
    "top_k": "MEMO_RECALL_TOP_K",
    "min_sim": "MEMO_RECALL_MIN_SIM",
    "min_body_chars": "MEMO_RECALL_MIN_BODY_CHARS",
    "mode": "MEMO_RECALL_MODE",
    "project_boost": "MEMO_RECALL_PROJECT_BOOST",
    "global_boost": "MEMO_RECALL_GLOBAL_BOOST",
    "contextual": "MEMO_RECALL_CONTEXTUAL",
    "mmr_lambda": "MEMO_RECALL_MMR_LAMBDA",
    "synthesis_boost": "MEMO_RECALL_SYNTHESIS_BOOST",
}


def recommend(rows: list[Row]) -> str:
    """Concrete next-step suggestion: pick the config with the best
    precision (tie-break: lower noise), and if it beats the baseline, map it
    to the MEMO_* knobs that reproduce it. Warns when the winner's p50 search
    latency would blow the recall-hook budget."""
    if not rows:
        return "no configs evaluated."
    baseline = rows[0]
    best = best_row(rows)
    if best.config == baseline.config:
        return "Baseline config already wins — no knob change recommended."
    dp = best.precision_at_k - baseline.precision_at_k
    dn = best.noise_at_k - baseline.noise_at_k
    cfg = next((c for c in default_configs() if c.name == best.config), None)
    knobs = ""
    if cfg is not None:
        knobs = f"  export MEMO_RECALL_MODE={cfg.mode}\n  export MEMO_RECALL_MIN_SIM={cfg.floor}"
        for field_name, value in (cfg.knob_overrides or {}).items():
            flag_name = _KNOB_FIELD_TO_FLAG.get(field_name)
            if flag_name:
                knobs += f"\n  export {flag_name}={value}"
        if cfg.exclude_archived:
            knobs += "\n  (and keep archive exclusion on in the recall hook)"
    out = f"Best config: {best.config} (prec {dp:+.3f}, noise {dn:+.3f} vs baseline).\n{knobs}"
    if best.latency_ms_p50 > _HOOK_SEARCH_BUDGET_MS:
        out += (
            f"\n  ⚠ p50 search {best.latency_ms_p50:.0f}ms exceeds the "
            f"~{_HOOK_SEARCH_BUDGET_MS:.0f}ms recall-hook budget — best for "
            f"`memo ask`/chat, but keep a faster mode for the hook."
        )
    return out


# --- Regression gate ---------------------------------------------------------


@dataclass
class GateResult:
    passed: bool
    message: str
    precision_at_k: float
    noise_at_k: float
    baseline_precision: float
    baseline_noise: float


def gate_metrics(rows: list[Row]) -> dict[str, float]:
    """The single (precision@K, noise@K) pair the gate tracks — the best config."""
    b = best_row(rows)
    return {"precision_at_k": b.precision_at_k, "noise_at_k": b.noise_at_k}


def check_gate(rows: list[Row], baseline: dict[str, float], *, tol: float = 1e-9) -> GateResult:
    """Compare the current best config against a saved baseline.

    The gate FAILS if precision@K dropped below, or noise@K rose above, the
    baseline (beyond `tol`). `tol` absorbs float noise; widen it to allow a
    small accepted drift.
    """
    m = gate_metrics(rows)
    bp = float(baseline.get("precision_at_k", 0.0))
    bn = float(baseline.get("noise_at_k", 1.0))
    prec_ok = m["precision_at_k"] >= bp - tol
    noise_ok = m["noise_at_k"] <= bn + tol
    passed = prec_ok and noise_ok
    if passed:
        message = (
            f"PASS — prec@k {m['precision_at_k']:.3f} >= {bp:.3f}, "
            f"noise@k {m['noise_at_k']:.3f} <= {bn:.3f}"
        )
    else:
        parts = []
        if not prec_ok:
            parts.append(f"precision@k {m['precision_at_k']:.3f} < baseline {bp:.3f}")
        if not noise_ok:
            parts.append(f"noise@k {m['noise_at_k']:.3f} > baseline {bn:.3f}")
        message = "FAIL — " + "; ".join(parts)
    return GateResult(passed, message, m["precision_at_k"], m["noise_at_k"], bp, bn)


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


# --- Auto-harvest labels from the grounding log ------------------------------


def harvest_labels(
    state_dir: Path,
    *,
    strong: float = 0.5,
    specific_margin: float = 0.0,
    max_labels: int = 200,
    sim_threshold: float = 0.6,
) -> list[dict[str, Any]]:
    """Mine ground-truth recall labels from ``grounding.log``.

    A grounding row records that a recalled memory was actually USED in an
    answer (``used_score = max(lexical, embed_cosine)``) — ground truth BY
    CONSTRUCTION, no hand-labeling. Rows at/above ``strong`` (and, when a
    ``specific_score`` is present, above ``specific_margin``) are joined to
    their prompt via ``recall_hook.log`` and emitted as
    ``{text, relevant: True, expect_ids:[<8-hex recall id>]}``. Re-asks of the
    same question collapse by prompt token-Jaccard, unioning their grounded
    ids, so the set grows from what actually mattered instead of by hand.

    Schema-additive ``project``: a grounding row's ``project`` field (the
    ``project:<slug>`` tag the hook resolved from cwd) propagates into the
    label — first non-null wins within a cluster; rows without one leave the
    key absent — so the eval can replay the label with the same project
    boosts the original recall ranked under.
    """
    from memo.dashboard import read_grounding_log
    from memo.dashboard_metrics import _jaccard, _reask_tokens
    from memo.grounding import _prompt_for_turn

    rows = read_grounding_log(state_dir, limit=4000)
    clusters: list[dict[str, Any]] = []
    for r in rows:
        try:
            used = float(r.get("used_score") or 0.0)
        except (TypeError, ValueError):
            continue
        if used < strong:
            continue
        spec = r.get("specific_score")
        if isinstance(spec, (int, float)) and spec <= specific_margin:
            continue
        sid = r.get("session_id")
        turn = r.get("turn")
        rid = str(r.get("recall_id") or "")
        if not sid or turn is None or len(rid) < 8:
            continue
        prompt = _prompt_for_turn(state_dir, str(sid), int(turn))
        if not prompt or len(prompt.strip()) < 8:
            continue
        tok = _reask_tokens(prompt)
        ts = r.get("ts") or ""
        proj = str(r.get("project") or "").strip() or None
        for c in clusters:
            if _jaccard(tok, c["tokens"]) >= sim_threshold:
                c["expect_ids"].add(rid)
                if proj and not c.get("project"):
                    c["project"] = proj  # first non-null wins
                if ts > c["ts"]:
                    c["ts"] = ts
                    c["text"] = prompt
                break
        else:
            clusters.append(
                {"tokens": tok, "text": prompt, "expect_ids": {rid}, "ts": ts, "project": proj}
            )
    clusters.sort(key=lambda c: c["ts"], reverse=True)
    out: list[dict[str, Any]] = []
    for c in clusters[:max_labels]:
        label: dict[str, Any] = {
            "text": c["text"],
            "relevant": True,
            "expect_ids": sorted(c["expect_ids"]),
        }
        if c.get("project"):
            label["project"] = c["project"]
        out.append(label)
    return out


def merge_label_prompts(
    existing: list[dict[str, Any]],
    harvested: list[dict[str, Any]],
    *,
    sim_threshold: float = 0.6,
) -> list[dict[str, Any]]:
    """Merge harvested labels into an existing prompt list. A harvested label
    Jaccard-similar to an existing one unions its ``expect_ids`` into that
    entry instead of adding a duplicate; otherwise it is appended. The
    optional ``project`` field merges first-non-null-wins: an existing entry
    keeps its own, an entry without one adopts the harvested label's."""
    from memo.dashboard_metrics import _jaccard, _reask_tokens

    merged = [dict(p) for p in existing]
    toks = [_reask_tokens(str(p.get("text") or "")) for p in merged]
    for h in harvested:
        h_tok = _reask_tokens(h["text"])
        for i, p in enumerate(merged):
            if _jaccard(h_tok, toks[i]) >= sim_threshold:
                ids = {str(x) for x in (p.get("expect_ids") or [])} | set(h["expect_ids"])
                p["expect_ids"] = sorted(ids)
                if h["expect_ids"]:
                    p["relevant"] = True
                if not p.get("project") and h.get("project"):
                    p["project"] = h["project"]
                break
        else:
            merged.append(h)
            toks.append(h_tok)
    return merged
