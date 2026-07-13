from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import click

from memo.config import Config
from memo.flags import flag_bool, flag_float, flag_int, flag_str
from memo.recall_logic import session_budget_scale  # re-export for tests + local use

_log = logging.getLogger("memo.cli_recall_hook")

_TRIVIAL_WORDS: frozenset[str] = frozenset(
    {
        "yes",
        "no",
        "ok",
        "sure",
        "yep",
        "nope",
        "continue",
        "go",
        "ahead",
        "proceed",
        "sí",
        "si",
        "dale",
        "listo",
        "gracias",
        "thanks",
        "k",
        "cool",
        "perfect",
    }
)


def maybe_inject_verbosity_steering(system_prompt: str, level: int) -> str:
    """Append idempotent verbosity steering block to system prompt.

    Levels (cumulative, byte-stable):
    0: No steering (return unchanged)
    1: "Skip preamble and postamble. Start with substance."
    2: "Skip preamble/postamble. Never restate code/diffs; reference by path+line. After tool success, continue without narrating."
    3: "Minimum tokens. Fragments OK. No preamble, no rationale unless asked."
    """
    VERBOSITY_TEXTS = {
        0: "",
        1: "Skip preamble and postamble. Start with substance.",
        2: "Skip preamble/postamble. Never restate code/diffs; reference by path+line. After tool success, continue without narrating.",
        3: "Minimum tokens. Fragments OK. No preamble, no rationale unless asked.",
    }

    level = max(0, min(3, level))  # Clamp
    if level == 0:
        return system_prompt

    SENTINEL_START = "<headroom_recall_verbosity>"
    SENTINEL_END = "</headroom_recall_verbosity>"

    # Check if already injected (idempotency)
    if SENTINEL_START in system_prompt and SENTINEL_END in system_prompt:
        return system_prompt

    steering_text = VERBOSITY_TEXTS[level]
    steering_block = f"\n{SENTINEL_START}{level}\n{steering_text}\n{SENTINEL_END}"

    return system_prompt + steering_block


@click.command(name="recall-hook")
def recall_hook() -> None:
    """UserPromptSubmit hook — inject relevant memories as additionalContext."""
    try:
        cfg = Config.from_env()
    except Exception:
        print("{}")
        sys.exit(0)

    mem: Any | None = None

    def _close_memory() -> None:
        if mem is not None:
            with contextlib.suppress(Exception):
                mem.close()

    def _bail(reason: str = "") -> None:
        if reason and flag_bool("MEMO_RECALL_DEBUG"):
            print(f"# memo recall-hook: {reason}", file=sys.stderr)
        if reason:
            try:
                from memo.dashboard import append_recall_log

                append_recall_log(
                    cfg.state_dir,
                    prompt="",
                    hits=[],
                    via="bail",
                    reason=reason,
                )
            except Exception as exc:
                _log.debug("bail recall-log write failed: %s", exc)
        _close_memory()
        print("{}")
        sys.exit(0)

    try:
        raw = sys.stdin.read()
        if not raw.strip():
            _bail("empty stdin")
            return
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        _bail(f"stdin parse fail: {exc}")
        return

    prompt = (payload.get("prompt") or "").strip()
    _sid = (payload.get("session_id") or "").strip() or None

    if flag_bool("MEMO_RECALL_DISABLE"):
        # Ablation cohort: recall is OFF for this turn. Stamp it (via="disabled",
        # with prompt/session/turn) so `memo roi`/`memo tokens` can report real
        # with-vs-without deltas. No embed/search/Memory import — this path is
        # one JSON parse + one small file append, far inside the 5s budget.
        _turn_off: int | None = None
        if _sid:
            try:
                from memo import session as _session_mod

                _turn_off = _session_mod.next_turn(cfg.state_dir, _sid)
            except Exception as e:
                _log.debug("next_turn (disabled path) failed: %s", e)
        try:
            from memo.dashboard import append_recall_log

            append_recall_log(
                cfg.state_dir,
                prompt=prompt,
                hits=[],
                via="disabled",
                session_id=_sid,
                turn=_turn_off,
                client=flag_str("MEMO_RECALL_CLIENT"),
            )
        except Exception as exc:
            _log.debug("disabled recall-log write failed: %s", exc)
        print("{}")
        sys.exit(0)
    _mc = flag_int("MEMO_RECALL_MIN_PROMPT_CHARS")
    min_chars = 12 if _mc is None else _mc

    if flag_bool("MEMO_RECALL_SKIP_SLASH") and prompt.startswith("/"):
        head, _, rest = prompt[1:].partition(" ")
        rest = rest.strip()
        _sm = flag_int("MEMO_RECALL_SLASH_MIN_ARG_CHARS")
        slash_min = 8 if _sm is None else _sm
        denylist = {
            c.strip().lower()
            for c in (flag_str("MEMO_RECALL_SLASH_DENYLIST") or "").split(",")
            if c.strip()
        }
        if len(rest) < slash_min:
            _bail("slash command (no args)")
            return
        if head.lower() in denylist:
            _bail(f"slash command (noise: {head.lower()})")
            return
        prompt = rest

    if len(prompt) < min_chars:
        expanded = ""
        n_turns = flag_int("MEMO_RECALL_SHORT_EXPAND_TURNS") or 0
        if _sid and n_turns > 0 and flag_bool("MEMO_RECALL_EXPAND_CONTEXT"):
            try:
                from memo import session as _session_mod

                prior = _session_mod.recent_prompts(cfg.state_dir, _sid, n_turns)
                prior = [p.strip() for p in prior if p.strip() and p.strip() != prompt]
                if prior:
                    expanded = "\n".join([*prior, prompt]).strip()
            except Exception as e:
                _log.debug("recent_prompts failed: %s", e)
                expanded = ""
        if len(expanded) >= min_chars:
            prompt = expanded
        else:
            _bail(f"prompt too short ({len(prompt)} < {min_chars})")
            return

    if flag_bool("MEMO_RECALL_TRIVIAL_BAIL"):
        _stripped = re.sub(r"[^\w\s]", "", prompt)
        _words = _stripped.split()
        if len(_words) <= 3 and any(w.lower() in _TRIVIAL_WORDS for w in _words):
            _bail("trivial prompt")
            return

    _client = flag_str("MEMO_RECALL_CLIENT")
    _turn: int | None = None
    if _sid:
        try:
            from memo import session as _session_mod

            _turn = _session_mod.next_turn(cfg.state_dir, _sid)
            _session_mod.stamp_recall_turn(cfg.state_dir, _sid, _turn)
        except Exception as e:
            _log.debug("next_turn failed: %s", e)
            _turn = None

    _t0 = time.time()
    try:
        from memo.recall_server import connect_and_recall

        _raw_float = flag_float("MEMO_RECALL_DAEMON_TIMEOUT")
        if _raw_float is not None and _raw_float >= 0.1:
            _daemon_timeout = _raw_float
        else:
            _daemon_timeout = max(0.2, (2000 if (_dmt := flag_int("MEMO_RECALL_DAEMON_TIMEOUT_MS")) is None else _dmt) / 1000.0)
        _daemon_result = connect_and_recall(
            cfg.state_dir,
            prompt=prompt,
            cwd=payload.get("cwd"),
            timeout=_daemon_timeout,
            session_id=_sid,
            turn=_turn,
            client=_client,
        )
        if _daemon_result is not None:
            _latency_ms = int((time.time() - _t0) * 1000)
            if flag_bool("MEMO_RECALL_DEBUG"):
                print(f"# memo recall-hook: daemon hit ({_latency_ms} ms)", file=sys.stderr)
            print(_daemon_result)
            try:
                from memo import recall_metrics

                recall_metrics.stamp(
                    cfg.state_dir,
                    total_ms=(time.time() - _t0) * 1000.0,
                    path="daemon",
                    hits=recall_metrics.count_hits(_daemon_result),
                )
            except Exception as exc:
                _log.debug("recall metrics stamp failed: %s", exc)
            sys.exit(0)
    except Exception as _daemon_exc:
        try:
            from memo.dashboard import append_recall_log

            append_recall_log(
                cfg.state_dir,
                prompt=prompt,
                hits=[],
                via="daemon_error",
                error=f"{type(_daemon_exc).__name__}: {_daemon_exc}",
            )
        except Exception as exc:
            _log.debug("daemon-error recall-log write failed: %s", exc)

    payload_cwd = payload.get("cwd")

    from memo.recall_logic import (
        apply_injection_filters,
        knobs_from_flags,
        make_vec_cosine,
        rank_hits,
    )

    # Single-source knob resolution — the SAME builder the daemon path
    # (_recall_logic) uses, so the two paths cannot diverge on ranking
    # (project tiers, preference/graph/mmr/synthesis knobs included).
    knobs = knobs_from_flags(cwd=payload_cwd)

    # Session-mode adjustments (subprocess-only: MEMFLOW_SESSION_MODE is a
    # per-session env var the long-lived daemon process cannot see).
    _session_mode = os.environ.get("MEMFLOW_SESSION_MODE", "").strip().lower()
    if _session_mode == "focus":
        knobs = replace(knobs, top_k=min(knobs.top_k, 2), min_sim=max(knobs.min_sim, 0.65))
    elif _session_mode == "explore":
        knobs = replace(knobs, top_k=max(knobs.top_k, 5), min_sim=min(knobs.min_sim, 0.4))
    elif _session_mode == "maintenance":
        knobs = replace(knobs, top_k=1, min_sim=max(knobs.min_sim, 0.70))

    _bc = flag_int("MEMO_RECALL_BODY_CHARS")
    body_chars = 400 if _bc is None else _bc
    _tb = flag_int("MEMO_RECALL_TOKEN_BUDGET")
    token_budget = _tb or 0

    # Adaptive budget: scale by prompt length
    if flag_bool("MEMO_RECALL_ADAPTIVE_BUDGET") and token_budget > 0 and prompt:
        prompt_len = len(prompt)
        # Short prompts (clarity) get more budget; long prompts leave room
        if prompt_len < 50:
            token_budget = int(min(token_budget * 1.5, 800))
        elif prompt_len > 300:
            token_budget = int(max(token_budget * 0.6, 200))
        # Mid-range stays as-is

    # Session cumulative budget decay: once the session has consumed more than
    # MEMO_RECALL_SESSION_TOKEN_BUDGET tokens of recall context, halve the
    # per-turn budget (floored at _SESSION_BUDGET_FLOOR). Default OFF (0).
    _sess_budget = flag_int("MEMO_RECALL_SESSION_TOKEN_BUDGET") or 0
    if _sess_budget > 0 and token_budget > 0 and _sid:
        try:
            from memo.dashboard import read_context_cost_log

            _cum = sum(
                (int(e.get("chars") or 0) + 3) // 4
                for e in read_context_cost_log(cfg.state_dir)
                if e.get("kind") == "recall" and e.get("session_id") == _sid
            )
            token_budget = session_budget_scale(_cum, _sess_budget, token_budget)
        except Exception as exc:
            _log.debug("session budget scale failed: %s", exc)

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    if knobs.mode == "hybrid":
        os.environ.setdefault(
            "MEMO_RERANK_INPUT_K",
            str(10 if (_rik := flag_int("MEMO_RECALL_RERANK_INPUT_K")) is None else _rik),
        )

    if knobs.mode in ("vec", "hybrid") and not flag_bool("MEMO_RECALL_FORCE_MODE"):
        # Cold-start downgrade — the subprocess equivalent of the daemon's
        # embedder-warm check (a cold MLX load would blow the 5s hook budget).
        try:
            _signal = cfg.state_dir / ".prewarm_ts"
            _warm = _signal.exists() and (time.time() - float(_signal.read_text().strip())) < 3600
            if not _warm:
                if flag_bool("MEMO_RECALL_DEBUG"):
                    print("# memo recall-hook: cold start — downgrading to bm25", file=sys.stderr)
                knobs = replace(knobs, mode="bm25")
        except Exception as exc:
            _log.debug("warm-signal read failed, staying in %s mode: %s", knobs.mode, exc)

    top_k = knobs.top_k
    mode = knobs.mode
    search_k = top_k * 3 if (knobs.project_tag or knobs.contextual) else top_k
    from memo.tiers import REFERENCE_TYPES

    exclude_types = set(REFERENCE_TYPES) if flag_bool("MEMO_RECALL_EXCLUDE_REFERENCE") else None
    try:
        from memo.memory import Memory

        mem = Memory(cfg)
    except Exception as exc:
        _bail(f"search failed: {exc}")
        return

    # Ranking pipeline — identical to the daemon path (_recall_logic):
    # rank_hits with the hybrid cosine gate, preference boost and the graph
    # seam, then the shared skip-below/gap injection filters.
    _vec_cosine = make_vec_cosine(mem, prompt)

    _prefs: Any | None = None
    if knobs.contextual:
        with contextlib.suppress(Exception):
            _prefs = mem.contextual.context.get_preferences()

    _graph_boost: Callable[[list[Any]], list[Any]] | None = None
    _gpw = flag_float("MEMO_RECALL_GRAPH_PROXIMITY_WEIGHT") or 0.0
    if flag_bool("MEMO_RECALL_GRAPH_PROXIMITY") and _gpw > 0:
        with contextlib.suppress(Exception):
            from memo.graph_proximity import extract_query_entities, graph_boost_factory

            _graph_boost = graph_boost_factory(
                mem.graph, extract_query_entities(prompt, mem.graph), weight=_gpw
            )

    def _rank(query_text: str) -> list:
        try:
            hits = mem.search(
                query_text, limit=search_k, mode=mode, recency=True, exclude_types=exclude_types
            )
        except Exception as exc:
            if flag_bool("MEMO_RECALL_DEBUG"):
                print(f"# memo recall-hook: search failed: {exc}", file=sys.stderr)
            return []
        return rank_hits(
            hits, knobs, vec_cosine=_vec_cosine, preferences=_prefs, graph_boost=_graph_boost
        )

    qualifying = _rank(prompt)

    if not qualifying and flag_bool("MEMO_RECALL_EXPAND_CONTEXT"):
        from memo.recall_logic import _session_context

        _ctx = _session_context(mem, exclude_types)
        if _ctx:
            qualifying = _rank(f"{_ctx}\n{prompt}")
            if qualifying and flag_bool("MEMO_RECALL_DEBUG"):
                print(
                    f"# memo recall-hook: query expansion recovered {len(qualifying)} hits",
                    file=sys.stderr,
                )

    qualifying = apply_injection_filters(qualifying)

    _guard_banner: str | None = None
    _guard_ids: list[str] = []
    _guard_sim_threshold = flag_float("MEMO_GUARD_SIM_THRESHOLD") or 0.6
    if flag_bool("MEMO_GUARD_ENABLED") and qualifying:
        from memo.guard import guard_banner as _gb
        from memo.guard import guard_candidates as _gc

        _guard_banner = _gb(prompt, qualifying, sim_threshold=_guard_sim_threshold)
        if _guard_banner:
            _guard_ids = [
                getattr(h, "id", "")
                for h in _gc(prompt, qualifying, sim_threshold=_guard_sim_threshold)[:1]
            ]

    _interject_banner: str | None = None
    if qualifying:
        from memo import interject as _ij

        _interject_banner = _ij.evaluate_and_render(
            cfg, mem, prompt=prompt, hits=qualifying, sim_threshold=_guard_sim_threshold,
        )

    def _stamp_metrics(n_hits: int) -> None:
        # ``hits`` is the POST-session-dedup injected count (0 on a bail) so
        # the subprocess line is comparable with the daemon path, which counts
        # the final rendered output. ``total_ms`` stays end-to-end from _t0.
        try:
            from memo import recall_metrics

            recall_metrics.stamp(
                cfg.state_dir,
                total_ms=(time.time() - _t0) * 1000.0,
                path="subprocess",
                hits=n_hits,
            )
        except Exception as exc:
            _log.debug("recall metrics stamp failed: %s", exc)

    relevant = qualifying[:top_k]

    # Precision gate (Lever 3): suppress when the top hit's score falls in a
    # band that has historically never been grounded.  Reads a small cached
    # JSON — cheap for the 5 s recall-hook budget.  Default OFF.
    if flag_bool("MEMO_RECALL_PRECISION_GATE") and relevant:
        try:
            from memo.token_meter import load_precision_bands
            from memo.token_meter import suppress_score as _pg_suppress

            _pg_bands = load_precision_bands(cfg.state_dir)
            if _pg_bands and _pg_suppress(relevant[0].score, _pg_bands):
                _stamp_metrics(0)
                _bail("precision-gated (learned zero-grounding band)")
                return
        except Exception as _pg_exc:
            _log.debug("precision gate check failed: %s", _pg_exc)

    if flag_bool("MEMO_RECALL_INTRA_DEDUP") and len(relevant) > 1:
        from memo.recall_logic import collapse_near_dups

        _thr = flag_float("MEMO_RECALL_INTRA_DEDUP_THRESHOLD")
        relevant = collapse_near_dups(relevant, threshold=0.8 if _thr is None else _thr)
    # Rank-overflow nudge (the hits just below the top-K cut) — same split the
    # daemon path renders, distinct from the graph-associative nudge below.
    nudge = qualifying[top_k : top_k + 2]

    if relevant and knobs.contextual:
        with contextlib.suppress(Exception):
            mem.contextual.record_search(prompt, [h.id for h in relevant])

    _latency_ms_subprocess = int((time.time() - _t0) * 1000)
    try:
        from memo.dashboard import append_recall_log

        append_recall_log(
            cfg.state_dir,
            prompt=prompt,
            hits=[
                {"id": h.id, "score": h.score, "title": h.title, "snippet": (h.body or "")[:240]}
                for h in relevant
            ],
            mode=mode,
            latency_ms=_latency_ms_subprocess,
            via="subprocess",
            session_id=_sid,
            turn=_turn,
            client=_client,
        )
    except Exception as exc:
        _log.debug("subprocess recall-log write failed: %s", exc)

    if not relevant:
        _stamp_metrics(0)
        _bail(f"no hits above min_sim={knobs.min_sim}")
        return

    # Session dedup: filter IDs already injected in earlier turns (already in context window).
    _prev_recalled: dict[str, int] = {}
    if _sid and _turn is not None:
        try:
            from memo import session as _session_mod

            _prev_recalled = _session_mod.get_recalled_ids(cfg.state_dir, _sid)
        except Exception as e:
            _log.debug("get_recalled_ids failed: %s", e)
            _prev_recalled = {}
    if _prev_recalled:
        relevant = [h for h in relevant if h.id not in _prev_recalled]
    _stamp_metrics(len(relevant))
    if not relevant:
        _bail("all hits already recalled this session")
        return

    from memo.recall_logic import (
        CITE_INSTRUCTION,
        render_recall_balanced,
        render_recall_compact,
        render_recall_context,
    )

    def _est_tokens(s: str) -> int:
        return max(1, len(s) // 4)

    _recall_format = flag_str("MEMO_RECALL_FORMAT")
    # Auto mode: choose format based on budget and hit count
    if _recall_format == "auto":
        if (token_budget > 0 and token_budget <= 300) or len(relevant) >= 5:
            _recall_format = "compact"
        elif token_budget > 800:
            _recall_format = "full"
        else:
            _recall_format = "balanced"
    # Compute associative nudge once for all formats — degrades to [] on any error.
    from memo.recall_assoc import build_nudge, render_associative_line

    try:
        _nudge = build_nudge(mem, relevant)
    except Exception:
        _nudge = []

    # Trust dossier (MEMO_HIT_DOSSIER, default off): one batched pairs_for_ids
    # lookup over the top-K ids — never per-hit — so the hook stays cheap.
    _disputed_by: dict[str, list[str]] = {}
    if flag_bool("MEMO_HIT_DOSSIER"):
        try:
            _ids = [h.id for h in relevant]
            for _p in mem.contradict_store.pairs_for_ids(
                _ids, status="open"
            ) + mem.contradict_store.pairs_for_ids(_ids, status="competing"):
                _disputed_by.setdefault(_p.memory_id_a, []).append(_p.memory_id_b)
                _disputed_by.setdefault(_p.memory_id_b, []).append(_p.memory_id_a)
        except Exception:
            _disputed_by = {}

    if _recall_format == "compact":
        context = render_recall_compact(
            relevant, token_budget=token_budget, disputed_by=_disputed_by,
            state_dir=mem.cfg.state_dir,
        )
    elif _recall_format == "balanced":
        context = render_recall_balanced(relevant, token_budget=token_budget, turn=_turn)
    else:
        context = render_recall_context(
            relevant,
            nudge,  # rank-overflow nudge — mirrors the daemon path's top_k split
            turn=_turn,
            body_chars=body_chars,
            token_budget=token_budget,
            disputed_by=_disputed_by,
            state_dir=mem.cfg.state_dir,
        )
    context = render_associative_line(context, _nudge, token_budget=token_budget)
    if flag_bool("MEMO_RECALL_CITE_INSTRUCTION"):
        context = f"{context}\n{CITE_INSTRUCTION}"

    # Apply verbosity steering (L4 token savings) if enabled
    from memo.flags_recall import flag_recall_verbosity_level
    verbosity_level = flag_recall_verbosity_level()
    if verbosity_level > 0:
        context = maybe_inject_verbosity_steering(context, verbosity_level)

    if token_budget > 0 and flag_bool("MEMO_RECALL_DEBUG"):
        approx = _est_tokens(context)
        print(f"# memo recall-hook: ~{approx} tokens (budget {token_budget})", file=sys.stderr)

    try:
        from memo.dashboard import append_context_cost_log

        append_context_cost_log(
            cfg.state_dir,
            kind="recall",
            chars=len(context),
            client=_client,
            session_id=_sid,
            turn=_turn,
        )
    except Exception as exc:
        _log.debug("context-cost log write failed: %s", exc)

    if _interject_banner:
        context = f"{_interject_banner}\n\n{context}"
    if _guard_banner:
        from memo.guard import log_guard_fire

        context = f"{_guard_banner}\n\n{context}"
        log_guard_fire(cfg.state_dir, prompt=prompt, ids=_guard_ids)

    output: dict[str, Any] = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }
    # Human-visible presence line — decoration only, never blocks the recall.
    _sysmsg = ""
    if flag_bool("MEMO_RECALL_SYSTEM_MESSAGE"):
        try:
            from memo.recall_logic import build_system_message

            _sysmsg = build_system_message(relevant)
        except Exception as exc:
            _log.debug("recall system-message build failed: %s", exc)

    # Cross-client "※ memo recap:" line (mirrors Claude Code's native recap).
    # Cheap cadence check (one JSON read + int compare) writing to the SAME
    # pending-notification file capture already uses, so every OTHER client
    # picks it up via the `notification` field on its next memo_* MCP call.
    # On Claude Code specifically, also fold the same line into
    # systemMessage below so it renders as a proactive, user-visible dim
    # line straight in the transcript — the closest memo can get to CC's
    # own native recap, on the one channel memo actually controls.
    # Best-effort; never raises; never blocks the recall.
    _recap_line = ""
    if _sid:
        try:
            from memo.cli_recap import maybe_write_recap

            _recap_line = maybe_write_recap(cfg.state_dir, _sid) or ""
        except Exception as exc:
            _log.debug("recap write failed: %s", exc)

    if _sysmsg or _recap_line:
        from memo.cli_recap import compose_system_message

        _combined = compose_system_message(_sysmsg, _recap_line)
        if _combined:
            output["systemMessage"] = _combined

    print(json.dumps(output, ensure_ascii=False))

    # Persist newly recalled IDs so future turns can dedup them
    if _sid and _turn is not None and relevant:
        try:
            from memo import session as _session_mod

            new_ids = {h.id: _turn for h in relevant if h.id not in _prev_recalled}
            _session_mod.mark_ids_recalled(cfg.state_dir, _sid, new_ids)
        except Exception as e:
            _log.debug("mark_ids_recalled failed: %s", e)

    try:
        from memo import presence

        presence.bump(cfg.state_dir, recalls=len(relevant))
    except Exception as e:
        _log.debug("presence bump failed: %s", e)

    _close_memory()
    sys.exit(0)
