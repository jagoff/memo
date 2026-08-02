"""`memo dream graduate-flags` — dark-feature (flag) graduation pipeline.

Every default-off ``*_ENABLED`` bool flag MUST declare a :class:`GateSpec` in
``GATES`` (enforced by ``tests/test_dream_flags.py`` — a new dark feature
cannot merge without declaring its graduation gate):

- ``recall``  — auto-measurable: a nightly ON/OFF A/B through the
  recall-faithful eval (``Cfg.flag_overrides`` env-pin seam, same corpus and
  (precision@K, -noise@K) objective as ``dream_tune``). A flag that wins
  ``MEMO_FLAG_GRADUATION_WIN_NIGHTS`` consecutive measurements — each also
  passing the latency-headroom and curated no-regression gates — graduates to
  ON via the tuned overlay. Reversible: a later night whose live metrics
  regress vs the graduation baseline reverts it (retry after a cooldown).
- ``tuner``   — already owned by an existing nightly tuner pass (e.g. HyDE /
  graph-retrieval); tracked here for the deadline sweep, never double-measured.
- ``manual``  — not auto-measurable by the recall eval (UX banners, ingest
  quality, ops, meta passes); ``reason`` documents why. A human graduates it
  by setting the flag via env or ``memo config set``.

Deadline sweep: any dark flag still un-graduated after
``MEMO_FLAG_GRADUATION_DEADLINE_DAYS`` becomes a ``cull_candidate`` in the
receipt and ``memo dream graduate-flags --status`` — flip it with a real gate
or delete the code path. Deletion stays human.

Distinct from ``dream_graduate.py`` (quarantined-MEMORY graduation): this
module graduates FEATURE FLAGS. Default ON as of v4.3.0
(``MEMO_DREAM_FLAG_GRADUATION_ENABLED``; opt out with =0).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from memo.dream_tune import _curated_label_set, _regressed, _scalar_overlay, build_labels
from memo.eval_recall import Cfg, LabelSet, evaluate, gate_metrics, limit_label_set
from memo.flags import REGISTRY, FlagSpec, flag_float, flag_int
from memo.tuned_overlay import read_overlay, write_overlay

_STATE_FILE = "flag_graduation.json"
_MIN_SIM = "MEMO_RECALL_MIN_SIM"
_TRUTHY = ("1", "true", "yes", "on")
# A winning candidate whose eval p50 exceeds the OFF config's p50 by more than
# this factor is rejected regardless of precision (same guard as the rank-knob
# tuner; skipped when the OFF p50 rounds to 0 — stub/tiny corpora).
FLAG_LATENCY_HEADROOM = 1.25

GateKind = Literal["recall", "tuner", "manual"]

# Gates the nightly code-drift pass (`cli_dream_passes._run_code_drift`); the
# canonical FlagSpec lives in flags_misc.py with the other dream pass flags.
CODE_DRIFT_FLAG = "MEMO_DREAM_CODE_DRIFT_ENABLED"
# Gates auto-repair inside the code-drift pass; canonical FlagSpec in
# flags_misc.py next to MEMO_DREAM_CODE_DRIFT_ENABLED.
CODE_REPAIR_FLAG = "MEMO_DREAM_CODE_REPAIR_ENABLED"
# Dark scalar flags (default 0.0 = OFF) that owe a graduation gate like any
# bool dark flag: the A/B seam pins them "0"/"1" (0.0 = off, 1.0 = full
# boost), so the recall gate measures them unchanged. Explicit allowlist —
# most float knobs are tuner territory, not graduation candidates.
_DARK_SCALAR_FLAGS = ("MEMO_RECALL_CODE_PROXIMITY_BOOST",)


@dataclass(frozen=True)
class GateSpec:
    """How one dark ``*_ENABLED`` flag earns (or is exempted from) auto-ON."""

    flag: str
    kind: GateKind
    # tuner: which pass owns it; manual: why it cannot auto-measure.
    reason: str
    # Eval mode for recall gates (vec keeps the A/B on the fast hook path).
    mode: str = "vec"
    # Companion env pins required for the ON measurement (and written to the
    # overlay alongside the flag when it graduates).
    extra_flags: tuple[tuple[str, str], ...] = ()


def _g(
    flag: str,
    kind: GateKind,
    reason: str,
    *,
    mode: str = "vec",
    extra_flags: tuple[tuple[str, str], ...] = (),
) -> tuple[str, GateSpec]:
    return flag, GateSpec(flag, kind, reason, mode=mode, extra_flags=extra_flags)


# The graduation contract: one entry per dark *_ENABLED flag. Completeness is
# enforced by tests — adding a dark flag without declaring its gate fails CI.
GATES: dict[str, GateSpec] = dict(
    (
        # --- recall: nightly ON/OFF A/B via the eval flag_overrides seam ------
        _g("MEMO_HYPE_ENABLED", "recall", "read-path HyPE fold; measurable in vec retrieval"),
        _g("MEMO_ENTITY_RETRIEVAL_ENABLED", "recall", "query-time entity-overlap boost"),
        _g("MEMO_CONTRADICT_PENALTY_ENABLED", "recall", "rank-time penalty on contradicted hits"),
        _g("MEMO_GRAPH_SIGNAL_ENABLED", "recall", "graph ranking signal; doc says eval-gated"),
        _g(
            "MEMO_RECALL_CODE_PROXIMITY_BOOST",
            "recall",
            "code-proximity additive boost (dark float, 0.0 = OFF); recall A/B "
            "via the eval flag_overrides seam — the ON pin '1' measures boost=1.0",
        ),
        _g(
            "MEMO_GRAPH_OUTCOME_SIGNAL_ENABLED",
            "recall",
            "roi modulation of graph boosts; needs the graph signal on",
            extra_flags=(("MEMO_GRAPH_SIGNAL_ENABLED", "1"),),
        ),
        # --- tuner: an existing nightly pass owns the flip --------------------
        _g("MEMO_HYDE_ENABLED", "tuner", "owned by dream_tune.run_hyde_pass"),
        _g(
            "MEMO_GRAPH_RETRIEVAL_ENABLED",
            "manual",
            "deprecated inert compatibility switch; no serving path consumes it",
        ),
        _g(
            "MEMO_GRAPH_EXPANSION_ENABLED",
            "manual",
            "deprecated inert compatibility switch; no serving path consumes it",
        ),
        # --- manual: not recall-measurable; human flips via env/config set ----
        _g(
            "MEMO_GRAPH_PROJECTION_ENABLED",
            "manual",
            "materialized read-model lifecycle; activate through graph-config.md",
        ),
        _g(
            "MEMO_GRAPH_CODE_TRACE_ENABLED",
            "manual",
            "read-only traceability projection; activate through graph-config.md",
        ),
        _g(
            "MEMO_GRAPH_DISCOVERY_ENABLED",
            "manual",
            "read-only discovery packet; activate through graph-config.md",
        ),
        _g("MEMO_GUARD_ENABLED", "manual", "UX banner; gate = user judgement on interjections"),
        _g("MEMO_INTERJECT_ENABLED", "manual", "UX banner; gate = user judgement"),
        _g(
            "MEMO_RECALL_CODE_REFS_ENABLED",
            "manual",
            "render-layer code citation lines; no retrieval effect — gate = token "
            "cost / user judgement",
        ),
        # --- Negative Recall (⛔ AVOID channel): measured by avoid@k in
        #     `memo eval recall`, not by the recall gate's (precision@k,-noise@k)
        #     _wins (surfacing failure_patterns excluded from normal recall does
        #     not move precision@k on the curated set), so all four flip manually.
        _g(
            "MEMO_NEGATIVE_RECALL_ENABLED",
            "manual",
            "negative-recall ⛔ channel; gate = avoid@k in memo eval recall, human flips",
        ),
        _g(
            "MEMO_NEGATIVE_RECALL_CAPTURE_ENABLED",
            "manual",
            "ingest/capture quality; gate = human review of minted failure_patterns",
        ),
        _g(
            "MEMO_NEGATIVE_RECALL_REINFORCE_ENABLED",
            "manual",
            "outcome-loop write (confidence/ROI on failure_patterns); gate = memo roi",
        ),
        _g(
            "MEMO_NEGATIVE_RECALL_TRIGGER_ENABLED",
            "manual",
            "context-risk trigger; measurable via avoid@k on release/delete prompts, human flips",
        ),
        _g("MEMO_GRAPH_REASON_ENABLED", "manual", "attribution metadata only, no ranking effect"),
        _g("MEMO_VERDICT_ENABLED", "manual", "next-turn reaction telemetry; no retrieval effect"),
        _g("MEMO_MAINT_SLEEP_CYCLE_ENABLED", "manual", "background maintenance; op cost decision"),
        _g(
            "MEMO_DYNAMIC_MANDATE_SYNC_ENABLED",
            "manual",
            "ops: auto-refresh mandate rule blocks in opted-in repos; gate = user judgement",
        ),
        _g("MEMO_CRUSHER_ENABLED", "manual", "ingest token economy; gate = memo eval tokens"),
        _g("MEMO_VLM_CAPTION_ENABLED", "manual", "ingest-time VLM cost; gate = ingest quality"),
        _g("MEMO_UPDATE_CHECK_ENABLED", "manual", "ops/update policy; human-only"),
        _g("MEMO_ASK_GAPS_ENABLED", "manual", "ask-path UX nudge; no retrieval metric"),
        _g(
            "MEMO_PROACTIVE_ENABLED",
            "manual",
            "proactive engine (statusline badge, urgent push, memo digest); UX/ops "
            "surface, not recall-measurable",
        ),
        _g("MEMO_SECRET_STORAGE_ENABLED", "manual", "security opt-in; human-only, never auto"),
        _g(
            "MEMO_SAMPLING_SYNTH_ENABLED",
            "manual",
            "MCP client-sampling for synthesis; not recall-measurable — quality "
            "gates in memo's synthesis evals, human flips via config",
        ),
        # --- manual (meta): flags gating nightly passes themselves ------------
        _g(
            CODE_DRIFT_FLAG,
            "manual",
            "meta: gates the code-drift pass; needs a trusted fresh codegraph index, "
            "not recall-measurable — human flips",
        ),
        _g(
            CODE_REPAIR_FLAG,
            "manual",
            "repairs correctos observados en receipts de N noches; "
            "falso-repair = ref apuntando a símbolo equivocado",
        ),
        _g("MEMO_DREAM_TUNE_ENABLED", "manual", "meta: gates the tuner pass; op cost decision"),
        _g("MEMO_DREAM_TUNE_BOOST_ENABLED", "manual", "meta: gates the boost explorer"),
        _g("MEMO_DREAM_HYDE_TUNE_ENABLED", "manual", "meta: gates the HyDE A/B pass"),
        _g(
            "MEMO_DREAM_RETRIEVAL_TUNE_ENABLED",
            "manual",
            "deprecated inert compatibility switch; tuner invocation was removed",
        ),
        _g("MEMO_DREAM_ANTICIPATE_ENABLED", "manual", "meta: gates the anticipate pass"),
        _g("MEMO_DREAM_HYPE_ENABLED", "manual", "meta: gates the nightly HyPE indexer"),
        _g(
            "MEMO_DREAM_VECTOR_HYGIENE_ENABLED",
            "manual",
            "meta: rebuildable cache/vec0 compaction; operational cost and retention policy require a human flip",
        ),
        _g(
            "MEMO_DREAM_VECTOR_VIEWS_ENABLED",
            "manual",
            "meta: nightly derived title/tag views; embedding cost and recall gate require a human flip",
        ),
        _g("MEMO_DREAM_COMMUNITIES_ENABLED", "manual", "meta: gates the communities pass"),
        _g("MEMO_DREAM_ENTITY_CANON_ENABLED", "manual", "meta: gates the entity-canon pass"),
        _g("MEMO_DREAM_EDGE_VERIFY_ENABLED", "manual", "meta: gates the edge-verify pass"),
        _g("MEMO_DREAM_FOLDER_ABSTRACTS_ENABLED", "manual", "meta: gates folder abstracts"),
        _g("MEMO_DREAM_DISTILL_ENABLED", "manual", "meta: gates the distill pass"),
        _g("MEMO_DREAM_BRIDGES_ENABLED", "manual", "meta: gates the bridges pass"),
        _g(
            "MEMO_DREAM_CONSOLIDATE_EPISODES_ENABLED", "manual", "meta: gates episode consolidation"
        ),
        _g("MEMO_DREAM_PROFILE_ENABLED", "manual", "meta: gates profile distillation"),
        _g("MEMO_DREAM_RETAG_GLOBAL_ENABLED", "manual", "meta: gates the retag pass"),
        _g("MEMO_DREAM_CHRONICLE_ENABLED", "manual", "meta: gates the chronicle diary pass"),
        # MEMO_DREAM_VALIDITY_EXTRACT_ENABLED / MEMO_DREAM_GRADUATION_ENABLED /
        # MEMO_DREAM_FLAG_GRADUATION_ENABLED graduated to default-ON in v4.3.0 —
        # no longer dark flags, so they carry no gate (test_no_graduated_or_stale_gates).
    )
)


def dark_flags() -> list[FlagSpec]:
    """Every default-off ``*_ENABLED`` bool flag in the registry, plus the
    :data:`_DARK_SCALAR_FLAGS` allowlist — the set ``GATES`` must cover."""
    bools = [
        s
        for s in REGISTRY.values()
        if s.kind == "bool" and s.name.endswith("_ENABLED") and not s.opt_out and not s.default
    ]
    return bools + [REGISTRY[name] for name in _DARK_SCALAR_FLAGS if name in REGISTRY]


def _human_value(name: str) -> str | None:
    """The human-pinned raw value for ``name`` (env first, then markdown
    config); None when not pinned. Graduation never overrides a human decision
    (env/config beat the overlay in flag resolution)."""
    v = os.environ.get(name)
    if v is not None:
        return v
    try:
        from memo.config_md import flag_values

        return flag_values(os.environ).get(name)
    except Exception:
        return None


def human_owned(name: str) -> bool:
    return _human_value(name) is not None


# --- state --------------------------------------------------------------------


def _state_path(state_dir: Path) -> Path:
    return Path(state_dir) / _STATE_FILE


def load_state(state_dir: Path) -> dict[str, Any]:
    try:
        doc = json.loads(_state_path(state_dir).read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state_dir: Path, state: dict[str, Any]) -> None:
    p = _state_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


# --- measurement ----------------------------------------------------------------


def measure_flag(
    mem: Any, labels: LabelSet, *, k: int, spec: GateSpec, enabled: bool, floor: float
) -> dict[str, float]:
    """prec@K / noise@K / p50 with ``spec.flag`` pinned on/off through the
    ``Cfg.flag_overrides`` env-pin seam (generalises ``measure_hyde`` — the seam
    for flags read inside ``Memory.search``/ranking that RankKnobs can't reach)."""
    pins = {spec.flag: "1" if enabled else "0"}
    if enabled:
        pins.update(dict(spec.extra_flags))
    cfg = Cfg(
        name=f"{spec.flag}={'on' if enabled else 'off'}",
        mode=spec.mode,
        floor=floor,
        exclude_archived=True,
        flag_overrides=pins,
    )
    rows = evaluate(mem, k=k, labels=labels, configs=[cfg])
    metrics = gate_metrics(rows)
    metrics["latency_ms_p50"] = round(rows[0].latency_ms_p50, 1) if rows else 0.0
    return metrics


def _wins(on: dict[str, float], off: dict[str, float]) -> bool:
    return (on["precision_at_k"], -on["noise_at_k"]) > (
        off["precision_at_k"],
        -off["noise_at_k"],
    )


# --- pass -----------------------------------------------------------------------


def _knobs() -> dict[str, int]:
    def _i(name: str, default: int) -> int:
        v = flag_int(name)
        return default if v is None else v

    return {
        "win_nights": _i("MEMO_FLAG_GRADUATION_WIN_NIGHTS", 3),
        "max_per_night": _i("MEMO_FLAG_GRADUATION_MAX_PER_NIGHT", 3),
        "deadline_days": _i("MEMO_FLAG_GRADUATION_DEADLINE_DAYS", 45),
        "retry_days": _i("MEMO_FLAG_GRADUATION_RETRY_DAYS", 14),
        "max_prompts": _i("MEMO_FLAG_GRADUATION_MAX_PROMPTS", 80),
    }


def _age_days(entry: dict[str, Any], field: str, today: date) -> int | None:
    raw = entry.get(field)
    if not raw:
        return None
    try:
        return (today - date.fromisoformat(str(raw))).days
    except ValueError:
        return None


def _finalize_pass_status(res: dict[str, Any]) -> None:
    """Set the aggregate receipt status from its per-flag outcomes."""
    if res["status"] == "noop" and res["measured"]:
        res["status"] = "measured"
    if res["graduated"] or res["reverted"]:
        res["status"] = "applied"


def _reset_rolled_back_flags(
    flags_state: dict[str, Any], overlay: dict[str, Any], outcomes: dict[str, Any]
) -> None:
    """Return graduated flags missing from the live overlay to tracking."""
    for name, entry in flags_state.items():
        if entry.get("status") == "graduated" and not overlay.get(name):
            entry.update({"status": "tracking", "streak": 0})
            entry.pop("baseline", None)
            outcomes[name] = {"verdict": "overlay_rollback_reset"}


def run_flag_graduation_pass(
    cfg: Any,
    mem: Any,
    *,
    k: int = 5,
    min_used_score: float = 0.5,
    dry_run: bool = False,
    today: date | None = None,
) -> dict[str, Any]:
    """One nightly flag-graduation pass. Returns a receipt fragment; never
    raises (mirrors the dream_tune passes). ``dry_run`` measures but never
    writes state or overlay."""
    res: dict[str, Any] = {"status": "noop", "flags": {}}
    try:
        today = today or date.today()
        knobs = _knobs()
        state = load_state(cfg.state_dir)
        flags_state: dict[str, Any] = state.setdefault("flags", {})

        # Track every dark flag from the night it first appears.
        for spec_reg in dark_flags():
            flags_state.setdefault(
                spec_reg.name,
                {"first_tracked": today.isoformat(), "streak": 0, "status": "tracking"},
            )

        # Human-pinned truthy flag = graduated by the human, not by us.
        for name, entry in flags_state.items():
            hv = _human_value(name)
            if hv is not None and hv.strip().lower() in _TRUTHY:
                entry["status"] = "human_graduated"

        labels, curated_used = build_labels(cfg, min_used_score=min_used_score)
        labels = limit_label_set(labels, knobs["max_prompts"])
        res["n_labels"] = len(labels.prompts)
        res["curated_used"] = curated_used
        floor = flag_float(_MIN_SIM)
        floor = 0.5 if floor is None else floor
        overlay = read_overlay(cfg.state_dir)

        # Out-of-band overlay rollback (deleting tuned_params.json or
        # `memo dream tune --rollback` — both documented) removes graduated
        # keys without touching graduation state. A flag still 'graduated' in
        # state but absent from the overlay is stranded: it resolves OFF live,
        # the regression guard skips it, and _eligible never re-measures it.
        # Reset it to tracking so it re-enters the A/B pool and deadline sweep.
        _reset_rolled_back_flags(flags_state, overlay, res["flags"])

        # Post-graduation regression guard: an overlay-graduated flag whose
        # live metrics regressed vs its graduation baseline is reverted (its
        # overlay key removed -> default OFF restored), then cools down.
        if labels.prompts:
            for name, entry in flags_state.items():
                gate = GATES.get(name)
                if (
                    gate is None
                    or entry.get("status") != "graduated"
                    or not isinstance(entry.get("baseline"), dict)
                    or not overlay.get(name)
                ):
                    continue
                live = measure_flag(mem, labels, k=k, spec=gate, enabled=True, floor=floor)
                res["flags"][name] = {"verdict": "guard", "live": live}
                if _regressed(live, entry["baseline"]) and not dry_run:
                    from memo import dream_tune_online

                    # One overlay change per proof cycle: like every other overlay
                    # writer, defer while a tuner experiment is pending or a revert
                    # just happened this cycle — a revert write here would bump
                    # params_version and orphan that same-night pending's cohort.
                    if dream_tune_online.has_unresolved_pending(
                        cfg.state_dir
                    ) or dream_tune_online.in_revert_cooldown(cfg.state_dir):
                        res["flags"][name]["verdict"] = "deferred_pending"
                        continue
                    params = _scalar_overlay(cfg.state_dir)
                    params.pop(name, None)
                    for extra, _v in gate.extra_flags:
                        params.pop(extra, None)
                    write_overlay(cfg.state_dir, params, {"set_by": "dream-flag-grad-revert"})
                    overlay = read_overlay(cfg.state_dir)
                    entry.update(
                        {"status": "reverted", "streak": 0, "reverted_at": today.isoformat()}
                    )
                    res["flags"][name]["verdict"] = "reverted"

        # Candidates: recall-gated, resolved OFF, not human-pinned, not cooling
        # down after a revert. Least-recently-measured first; cap per night.
        def _eligible(name: str) -> bool:
            gate = GATES.get(name)
            entry = flags_state[name]
            if gate is None or gate.kind != "recall":
                return False
            if entry.get("status") in ("graduated", "human_graduated"):
                return False
            if human_owned(name) or overlay.get(name):
                return False
            cooled = _age_days(entry, "reverted_at", today)
            return not (cooled is not None and cooled < knobs["retry_days"])

        candidates = sorted(
            (n for n in flags_state if _eligible(n)),
            key=lambda n: str(flags_state[n].get("last_measured") or ""),
        )[: knobs["max_per_night"]]
        res["measured"] = candidates

        if labels.prompts:
            curated = _curated_label_set(cfg.state_dir)
            for name in candidates:
                gate = GATES[name]
                entry = flags_state[name]
                off = measure_flag(mem, labels, k=k, spec=gate, enabled=False, floor=floor)
                on = measure_flag(mem, labels, k=k, spec=gate, enabled=True, floor=floor)
                verdict: dict[str, Any] = {"off": off, "on": on}
                win = _wins(on, off)
                budget = off.get("latency_ms_p50", 0.0) * FLAG_LATENCY_HEADROOM
                if win and budget > 0 and on.get("latency_ms_p50", 0.0) > budget:
                    win, verdict["latency_rejected"] = False, True
                if win and curated is not None:
                    c_off = measure_flag(mem, curated, k=k, spec=gate, enabled=False, floor=floor)
                    c_on = measure_flag(mem, curated, k=k, spec=gate, enabled=True, floor=floor)
                    verdict["curated"] = {"off": c_off, "on": c_on}
                    if _regressed(c_on, c_off):
                        win, verdict["curated_rejected"] = False, True
                entry["streak"] = entry.get("streak", 0) + 1 if win else 0
                entry["last_measured"] = today.isoformat()
                verdict["streak"] = entry["streak"]
                verdict["verdict"] = "win" if win else "lose"
                if win and entry["streak"] >= knobs["win_nights"]:
                    from memo import dream_tune_online

                    if dry_run:
                        verdict["verdict"] = "would_graduate"
                    elif dream_tune_online.has_unresolved_pending(
                        cfg.state_dir
                    ) or dream_tune_online.in_revert_cooldown(cfg.state_dir):
                        # One overlay change per proof cycle: writing now would
                        # bump params_version and expire the pending online
                        # verification. Keep the streak; graduate next night.
                        verdict["verdict"] = "deferred_pending"
                    else:
                        params = _scalar_overlay(cfg.state_dir)
                        params[name] = True
                        for extra, v in gate.extra_flags:
                            params[extra] = v.strip().lower() in _TRUTHY
                        write_overlay(
                            cfg.state_dir,
                            params,
                            {
                                "set_by": "dream-flag-graduation",
                                "flag": name,
                                "baseline_prec": on["precision_at_k"],
                                "baseline_noise": on["noise_at_k"],
                            },
                        )
                        overlay = read_overlay(cfg.state_dir)
                        entry.update({"status": "graduated", "baseline": on})
                        verdict["verdict"] = "graduated"
                res["flags"][name] = verdict
        elif candidates:
            res["status"] = "no_labels"

        # Deadline sweep: still un-graduated past the deadline -> cull candidate.
        culls: list[str] = []
        for name, entry in flags_state.items():
            if entry.get("status") not in ("tracking", "reverted", "cull_candidate"):
                continue
            age = _age_days(entry, "first_tracked", today)
            if age is not None and age > knobs["deadline_days"]:
                entry["status"] = "cull_candidate"
                culls.append(name)
        res["cull_candidates"] = sorted(culls)
        res["graduated"] = sorted(
            n for n, v in res["flags"].items() if v.get("verdict") == "graduated"
        )
        res["reverted"] = sorted(
            n for n, v in res["flags"].items() if v.get("verdict") == "reverted"
        )
        _finalize_pass_status(res)

        if not dry_run:
            save_state(cfg.state_dir, state)
    except Exception as exc:  # surfaced into the receipt, never silent
        res["status"] = "error"
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


# --- status (CLI) ---------------------------------------------------------------


def status_rows(cfg: Any, *, today: date | None = None) -> list[dict[str, Any]]:
    """One row per dark flag: gate kind, status, streak, tracked-for days and
    days left before the cull deadline — the inventory view behind
    ``memo dream graduate-flags --status``."""
    today = today or date.today()
    knobs = _knobs()
    flags_state = load_state(cfg.state_dir).get("flags", {})
    overlay = read_overlay(cfg.state_dir)
    rows = []
    for spec_reg in sorted(dark_flags(), key=lambda s: s.name):
        gate = GATES.get(spec_reg.name)
        entry = flags_state.get(spec_reg.name, {})
        age = _age_days(entry, "first_tracked", today)
        status = entry.get("status", "untracked")
        hv = _human_value(spec_reg.name)
        if hv is not None and hv.strip().lower() in _TRUTHY:
            status = "human_graduated"
        rows.append(
            {
                "flag": spec_reg.name,
                "kind": gate.kind if gate else "MISSING_GATE",
                "reason": gate.reason if gate else "",
                "status": status,
                "streak": entry.get("streak", 0),
                "tracked_days": age,
                "days_left": None if age is None else max(0, knobs["deadline_days"] - age),
                "overlay_on": bool(overlay.get(spec_reg.name)),
            }
        )
    return rows


__all__ = [
    "CODE_DRIFT_FLAG",
    "CODE_REPAIR_FLAG",
    "FLAG_LATENCY_HEADROOM",
    "GATES",
    "GateSpec",
    "dark_flags",
    "human_owned",
    "load_state",
    "measure_flag",
    "run_flag_graduation_pass",
    "save_state",
    "status_rows",
]
