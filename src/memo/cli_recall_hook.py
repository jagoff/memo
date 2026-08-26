from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import signal
import sqlite3
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import click

from memo.config import Config
from memo.flags import flag_bool, flag_float, flag_int, flag_str

# re-export for tests + local use (adaptive_token_budget /
# maybe_inject_verbosity_steering moved to recall_logic so the daemon path
# shares them; eval_tokens + tests still import them from here)
from memo.recall_logic import (
    RankKnobs,
    adaptive_token_budget,
    maybe_inject_verbosity_steering,
    recall_search_budget_ms,
    session_budget_scale,
)

_log = logging.getLogger("memo.cli_recall_hook")

# Small and fixed, not a share of the main search budget: the rollup is an
# additive extra query, not a replacement for the primary search, and must
# stay cheap enough to never meaningfully compete with it for the hook's
# wall-clock budget. See fetch_chunk_parent_hits for what this buys.
_CHUNK_PARENT_HOOK_LIMIT = 5
_CHUNK_PARENT_HOOK_BUDGET_MS = 400.0


def _apply_chunk_parent_rollup(mem: Any, query_text: str, mode: str, hits: list[Any]) -> list[Any]:
    """Chunk->parent rollup (MEMO_RECALL_CHUNK_PARENT, default off — see
    fetch_chunk_parent_hits for the full rationale). Skipped on the bm25
    downgrade for the same reason the recency band is: don't re-stall a
    daemon we just fell away from with a second query. Extracted out of
    `recall_hook`'s nested `_rank` to keep that closure's complexity budget
    unchanged (see the quality gate)."""
    if not (flag_bool("MEMO_RECALL_CHUNK_PARENT") and mode != "bm25"):
        return hits
    from memo.recall_logic import apply_recency_band, fetch_chunk_parent_hits

    return apply_recency_band(
        hits,
        fetch_chunk_parent_hits(
            mem,
            query_text,
            mode=mode,
            limit=_CHUNK_PARENT_HOOK_LIMIT,
            budget_ms=_CHUNK_PARENT_HOOK_BUDGET_MS,
        ),
    )


def _rank_overflow_omitted(qualifying: list[Any], pre_filter: list[Any], top_k: int) -> list[Any]:
    """Hits omitted from the injection: the rank-overflow tail below the nudge
    (``qualifying[top_k + 2:]``) plus any hit dropped by the injection filters
    (present in ``pre_filter`` but not in ``qualifying``). Computed exactly like
    the daemon path (recall_logic) so ``MEMO_RECALL_OMISSIONS_TAIL`` counts
    identically across both paths."""
    omitted = list(qualifying[top_k + 2 :])
    if qualifying and len(qualifying) < len(pre_filter):
        kept = {h.id for h in qualifying}
        omitted.extend(h for h in pre_filter if h.id not in kept)
    return omitted


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

# Budget cap for the subprocess-fallback query embed. A busy/warming daemon
# still answers `ping`, so the fallback `Memory` auto-routes embeds back through
# the socket; without a cap that wait is the 30s `_QUERY_TIMEOUT_S`, which blows
# Claude Code's hook kill. 4s leaves headroom under the 12s kill for the bm25
# downgrade that `_rank` falls to when a capped embed fails.
_FALLBACK_EMBED_TIMEOUT_S = 4.0


# Whether THIS module armed ITIMER_REAL, and what handler it displaced. The
# timer is process-global, so the disarm must be able to tell "ours" from "the
# host's" — see _disarm_deadline.
_deadline_armed = False
_prev_alarm_handler: Any = None


def _admit_prompt(prompt: str, bail: Callable[[str], None]) -> str:
    """The text recall should run on, or `bail` (which exits) for a machine turn.

    Called ahead of the MEMO_RECALL_DISABLE short-circuit on purpose: both arms
    of the ablation must exclude the same turns, or the two cohorts `memo
    tokens` compares stop being the same population and the net-savings number
    means nothing. A turn that mixed plumbing with a real question comes back as
    the question alone.
    """
    if not flag_bool("MEMO_RECALL_SKIP_MACHINE_PROMPTS"):
        return prompt
    from memo.recall_admission import admit

    admitted, why = admit(prompt)
    if admitted is not None:
        return admitted
    bail(f"machine prompt ({why})")
    raise SystemExit(0)  # `bail` already exits; explicit so the type is `str`


def _arm_deadline(started_at: float, bail: Callable[[str], None]) -> None:
    """Arm the fallback's wall-clock cap, if this platform and config want one.

    Every stage of the fallback has its own guard (a 4s embed cap, a bm25
    downgrade), yet the measured subprocess path still reached p95 9.5s and a
    126.7s worst case over 1500 live fires. This is the outer bound.

    Delivered between bytecodes, so it bounds waits that pass through the
    interpreter but cannot preempt a single blocking C call — a sqlite lock held
    while `memo maintain` writes is exactly that case, and `busy_timeout` is its
    lever, not this. The cap is the outer bound, not the whole answer.
    """
    _hb = flag_int("MEMO_RECALL_HOOK_BUDGET_MS")
    budget_s = (10000 if _hb is None else _hb) / 1000.0
    if budget_s <= 0 or not hasattr(signal, "SIGALRM"):
        return

    def _on_deadline(_signum: int, _frame: Any) -> None:
        # Disarm before doing anything: `bail` writes a log line, and a second
        # alarm landing inside that write would re-enter here.
        _disarm_deadline()
        bail(f"hook budget exceeded ({budget_s:g}s)")

    global _deadline_armed, _prev_alarm_handler
    with contextlib.suppress(ValueError, OSError):
        # ValueError: not the main thread (no signal delivery available).
        _prev_alarm_handler = signal.signal(signal.SIGALRM, _on_deadline)
        signal.setitimer(signal.ITIMER_REAL, max(0.1, budget_s - (time.time() - started_at)))
        _deadline_armed = True


def _disarm_deadline() -> None:
    """Cancel the recall-hook's wall-clock alarm, and only ever that one.

    ITIMER_REAL is process-global and shared. The hook normally owns its own
    process, but when it runs inside a host that also uses SIGALRM — pytest's
    `--timeout`, for one — clearing the timer unconditionally would cancel the
    host's alarm instead of ours. So this is a no-op unless `_arm_deadline`
    actually armed something. Idempotent, and never raises: a non-main thread
    has no timer to cancel and must not fail the caller.
    """
    global _deadline_armed
    if not _deadline_armed or not hasattr(signal, "SIGALRM"):
        return
    _deadline_armed = False
    with contextlib.suppress(ValueError, OSError):
        signal.setitimer(signal.ITIMER_REAL, 0)
        if _prev_alarm_handler is not None:
            signal.signal(signal.SIGALRM, _prev_alarm_handler)


def apply_session_mode(knobs: RankKnobs, session_mode: str) -> RankKnobs:
    """Apply bounded per-session ranking adjustments."""
    if session_mode == "focus":
        return replace(knobs, top_k=min(knobs.top_k, 2), min_sim=max(knobs.min_sim, 0.65))
    if session_mode == "explore":
        return replace(knobs, top_k=max(knobs.top_k, 5), min_sim=min(knobs.min_sim, 0.4))
    if session_mode == "maintenance":
        return replace(knobs, top_k=1, min_sim=max(knobs.min_sim, 0.70))
    return knobs


def _proactive_urgent_line(cfg: Config) -> str:
    """Pull the optional urgent nudge without growing the recall-hook entrypoint."""
    if not flag_bool("MEMO_PROACTIVE_ENABLED"):
        return ""
    try:
        from datetime import UTC as _UTC
        from datetime import datetime as _datetime

        from memo.proactive.engine import pull_urgent
        from memo.proactive.store import ProactiveStore
        from memo.proactive.surfaces import render_urgent_line

        now_dt = _datetime.now(tz=_UTC)
        with ProactiveStore(cfg.state_dir / "proactive.db") as store:
            urgent = pull_urgent(store, now=now_dt.isoformat(), day=now_dt.date().isoformat())
        return render_urgent_line(urgent) if urgent is not None else ""
    except Exception as exc:
        _log.debug("recall proactive urgent failed: %s", exc)
        return ""


def _append_coordination_block(cfg: Config, session_id: str | None, context: str) -> str:
    """Append this session's pending `<memo-coordination>` directives.

    Pure sqlite read (no LLM, no network) kept out of the entrypoint so the
    recall-hook complexity budget is untouched. Fail-open: any failure returns
    the context unchanged — the hook must never die. The broad tuple (instead
    of ``except Exception``) keeps the broad-exception ratchet budget intact."""
    try:
        from memo.coordination import deliver_pending_block

        block = deliver_pending_block(cfg, session_id)
    except (
        ImportError,
        OSError,
        ValueError,
        sqlite3.Error,
        AttributeError,
        KeyError,
        TypeError,
        RuntimeError,
    ) as exc:
        _log.debug("coordination block failed: %s", exc)
        return context
    if not block:
        return context
    return f"{context}\n\n{block}" if context else block


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
        # Every exit path calls this, so it is where the wall-clock deadline is
        # disarmed. An armed itimer outlives the hook when the hook is invoked
        # in-process rather than as its own short-lived process (tests, any
        # embedding host): the alarm then fires later, inside unrelated code,
        # and the handler's sys.exit unwinds whatever happened to be running.
        _disarm_deadline()
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

    # Ahead of the MEMO_RECALL_DISABLE short-circuit on purpose — see
    # _admit_prompt. Exits the process on a machine turn.
    prompt = _admit_prompt(prompt, _bail)

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

        _daemon_timeout = max(
            0.2,
            (2000 if (_dmt := flag_int("MEMO_RECALL_DAEMON_TIMEOUT_MS")) is None else _dmt)
            / 1000.0,
        )
        # Float alias wins only when moved off its registry default —
        # otherwise its default (2.0) would shadow an explicit _MS setting.
        from memo.flags import REGISTRY as _REG

        _raw_float = flag_float("MEMO_RECALL_DAEMON_TIMEOUT")
        if (
            _raw_float is not None
            and _raw_float >= 0.1
            and _raw_float != _REG["MEMO_RECALL_DAEMON_TIMEOUT"].default
        ):
            _daemon_timeout = _raw_float
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
            # Daemon warming/lock-bail marker ({"busy": true}) — not a recall
            # result: fall through to the subprocess path (which cold-start
            # downgrades to bm25) instead of injecting nothing for the whole
            # warmup window. A legit empty recall stays "{}" and is printed.
            with contextlib.suppress(Exception):
                _parsed = json.loads(_daemon_result)
                if isinstance(_parsed, dict) and _parsed.get("busy"):
                    if flag_bool("MEMO_RECALL_DEBUG"):
                        print(
                            "# memo recall-hook: daemon busy — subprocess fallback",
                            file=sys.stderr,
                        )
                    _daemon_result = None
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

    _arm_deadline(_t0, _bail)

    payload_cwd = payload.get("cwd")

    from memo.recall_logic import knobs_from_flags

    # Single-source knob resolution — the SAME builder the daemon path
    # (_recall_logic) uses, so the two paths cannot diverge on ranking
    # (project tiers, preference/graph/mmr/synthesis knobs included).
    knobs = knobs_from_flags(cwd=payload_cwd)

    # Optional per-session ranking preset.
    _session_mode = (flag_str("MEMO_RECALL_SESSION_MODE") or "").strip().lower()
    knobs = apply_session_mode(knobs, _session_mode)

    _bc = flag_int("MEMO_RECALL_BODY_CHARS")
    body_chars = 400 if _bc is None else _bc
    _tb = flag_int("MEMO_RECALL_TOKEN_BUDGET")
    token_budget = _tb or 0

    # Adaptive budget: scale by prompt length
    if flag_bool("MEMO_RECALL_ADAPTIVE_BUDGET") and token_budget > 0 and prompt:
        token_budget = adaptive_token_budget(token_budget, len(prompt))

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
        # search_ops reads cfg.rerank_input_k — a static pydantic field fixed
        # once by Config.from_env() above, NOT re-read from env — so the env
        # setdefault alone is a no-op on this path: Memory(cfg) below reuses the
        # already-built cfg and the shrink never reaches the reranker, leaving
        # the highest-latency subprocess fallback reranking the full pool (30)
        # under the 5s budget. Reflect the resolved pool size onto cfg so the
        # shrink actually takes effect. (setdefault keeps an explicit operator
        # MEMO_RERANK_INPUT_K authoritative — and cfg already carries that same
        # value, so no divergence.) The daemon path returns above, unchanged.
        with contextlib.suppress(ValueError):
            cfg = cfg.model_copy(update={"rerank_input_k": int(os.environ["MEMO_RERANK_INPUT_K"])})

    if knobs.mode in ("vec", "hybrid") and not flag_bool("MEMO_RECALL_FORCE_MODE"):
        # Cold-start downgrade — the subprocess equivalent of the daemon's
        # embedder-warm check (a cold MLX load would blow the 5s hook budget).
        # An unreadable/corrupt warm signal counts as NOT warm: failing open
        # here would pay the cold MLX load inside the hook budget.
        _warm = False
        try:
            _signal = cfg.state_dir / ".prewarm_ts"
            _warm = _signal.exists() and (time.time() - float(_signal.read_text().strip())) < 3600
        except Exception as exc:
            _log.debug("warm-signal read failed, treating as cold: %s", exc)
        if not _warm:
            if flag_bool("MEMO_RECALL_DEBUG"):
                print("# memo recall-hook: cold start — downgrading to bm25", file=sys.stderr)
            knobs = replace(knobs, mode="bm25")

    mode = knobs.mode

    # Budget guard for the fallback embed: a busy/warming daemon answers `ping`
    # (so `Memory` routes embeds through the socket) but not recall — cap the
    # client timeout to the residual budget and require the daemon, so a stalled
    # embed raises fast instead of cold-loading a 2nd MLX copy that fights the
    # daemon for the GPU lock. `_rank` downgrades to bm25 on that failure.
    # setdefault keeps an explicit operator override authoritative.
    if mode in ("vec", "hybrid"):
        os.environ.setdefault("MEMO_EMBEDDER_CLIENT_TIMEOUT", str(_FALLBACK_EMBED_TIMEOUT_S))
        os.environ.setdefault("MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON", "1")
    try:
        from memo.memory import Memory

        mem = Memory(cfg)
    except Exception as exc:
        _bail(f"search failed: {exc}")
        return

    # Shared pipeline: search → rank → filter → dedup → format → render
    from memo.recall_logic import run_recall_pipeline

    _prev_recalled_ids: set[str] = set()
    if _sid and _turn is not None:
        try:
            from memo import session as _session_mod

            _prev_recalled_ids = set(_session_mod.get_recalled_ids(cfg.state_dir, _sid))
        except Exception as e:
            _log.debug("get_recalled_ids failed: %s", e)

    def _stamp_metrics(n_hits: int) -> None:
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

    result = run_recall_pipeline(
        mem=mem,
        query_text=prompt,
        knobs=knobs,
        turn=_turn,
        session_id=_sid or "",
        state_dir=cfg.state_dir,
        previous_turn_ids=_prev_recalled_ids or None,
        via="subprocess",
    )

    _stamp_metrics(len(result.get("relevant", [])))

    if not result or "_status" in result:
        # Pipeline returned empty or status-only: handle subprocess-specific bail cases.
        _status = result.get("_status", "no_hits") if result else "no_hits"
        if _status == "search_failed":
            _bail("search failed — absence unproven, no marker")
        elif _status == "all_recalled":
            _bail("all hits already recalled this session")
        else:
            _bail(f"no hits above min_sim={knobs.min_sim}")
        return

    context = result["additionalContext"]
    relevant = result["relevant"]

    # Guard / interject banners — computed here (not in the pipeline) because
    # the subprocess needs to control the exact prepend order:
    # interject → guard → avoid → context.
    _guard_banner: str | None = None
    _guard_ids: list[str] = []
    _guard_sim_threshold = flag_float("MEMO_GUARD_SIM_THRESHOLD") or 0.6
    if flag_bool("MEMO_GUARD_ENABLED") and relevant:
        from memo.guard import guard_banner as _gb
        from memo.guard import guard_candidates as _gc

        _guard_banner = _gb(prompt, relevant, sim_threshold=_guard_sim_threshold)
        if _guard_banner:
            _guard_ids = [
                getattr(h, "id", "")
                for h in _gc(prompt, relevant, sim_threshold=_guard_sim_threshold)[:1]
            ]

    _interject_banner: str | None = None
    if relevant:
        from memo import interject as _ij

        _interject_banner = _ij.evaluate_and_render(
            cfg,
            mem,
            prompt=prompt,
            hits=relevant,
            sim_threshold=_guard_sim_threshold,
        )

    if _interject_banner:
        context = f"{_interject_banner}\n\n{context}"
    if _guard_banner:
        from memo.guard import log_guard_fire

        context = f"{_guard_banner}\n\n{context}"
        log_guard_fire(cfg.state_dir, prompt=prompt, ids=_guard_ids)

    # Cross-agent coordination directives (<memo-coordination>): pending
    # directives for this session, appended to the injected context so the
    # agent acts on them this turn. One indexed sqlite read — zero LLM, zero
    # network — and each side is stamped delivered exactly once (see
    # memo.coordination.deliver_pending_block). Best-effort, never blocks.
    context = _append_coordination_block(cfg, _sid, context)

    output: dict[str, Any] = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }
    # Human-visible presence line — decoration only, never blocks the recall.
    _sysmsg = ""
    if flag_bool("MEMO_RECALL_SYSTEM_MESSAGE"):
        _sysmsg = result.get("systemMessage", "")

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

    # Proactive urgent nudge — the recall hook's systemMessage is the only
    # synchronous channel Claude Code renders to the user (the Stop hook is
    # async, its stdout discarded). `pull_urgent` owns the push slot: it
    # respects the cooldown/daily-cap so this stays "útil sin molestar", and it
    # is MLX-free (one sqlite read + rank) so it fits the recall budget.
    # Best-effort; never blocks the recall.
    _urgent_line = _proactive_urgent_line(cfg)

    if _sysmsg or _recap_line or _urgent_line:
        from memo.cli_recap import compose_system_message

        _combined = compose_system_message(_sysmsg, _recap_line, _urgent_line)
        if _combined:
            output["systemMessage"] = _combined

    print(json.dumps(output, ensure_ascii=False))

    # Record what the model was just shown, so the MCP read tools can skip
    # re-sending it later in this session. Deliberately placed AFTER the
    # print above, not alongside it — the hook installs its own SIGALRM
    # wall-clock cap (_arm_deadline; measured p95 9.5s against a 10s cap over
    # 1500 live fires, so the alarm landing late is the norm, not an edge
    # case), and any exit between a write and the print would leave the
    # ledger asserting bodies the model never actually received. Same
    # ordering discipline as `mark_ids_recalled` just below, and the same
    # reason: only record post-facto, after delivery is no longer in doubt.
    # Fail-open by contract: the recall hook has a 5s budget and must never
    # break or slow down on a bookkeeping write. Uses identity._session_id()
    # (env var), not the payload-derived _sid above — verified to match what
    # the MCP side's _effective_session_id() resolves to for the same Claude
    # Code session (both inherit CLAUDE_CODE_SESSION_ID), so hook and MCP
    # writers key the same ledger file without coordination. A distinct
    # local (not _sid) so a client that exports no session env var can't
    # clobber the payload-derived _sid the rest of this function relies on.
    _emitted = result.get("emitted_sink", [])
    if flag_bool("MEMO_EMITTED_LEDGER") and _emitted:
        try:
            from memo import emitted_ledger as _el
            from memo.identity import _session_id as _identity_session_id

            # Mirror apply_ledger's safe_hits guard (server_common.py): an
            # empty-text pair records nothing lost (an empty string never
            # digests real content — n=0 can only ever match another n=0
            # emission) but WOULD overwrite a richer prior entry for the
            # same id, silently killing suppression for that memory for the
            # rest of the session. Drop before writing, not after.
            _pairs = [(_id, _body) for _id, _body in _emitted if _id and _body]
            _ledger_sid = _identity_session_id() if _pairs else None
            if _ledger_sid:
                _now = int(time.time())
                _ref = _el.mint_ref([_id for _id, _ in _pairs], _now, prefix="memo-h")
                _el.append(
                    cfg.state_dir,
                    _ledger_sid,
                    [_el.Entry.for_text(_id, _body, _ref, _now, "hook") for _id, _body in _pairs],
                )
        except Exception:  # noqa: S110  # fail-open: a ledger write failure just re-emits later
            pass

    try:
        from memo import presence

        presence.bump(cfg.state_dir, recalls=len(relevant))
    except Exception as e:
        _log.debug("presence bump failed: %s", e)

    _close_memory()
    sys.exit(0)
