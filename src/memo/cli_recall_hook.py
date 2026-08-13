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

    from memo.recall_logic import (
        apply_injection_filters,
        knobs_from_flags,
        make_vec_cosine,
        rank_hits,
        uncertain_exclusion,
    )

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

    top_k = knobs.top_k
    mode = knobs.mode
    search_k = top_k * 3 if (knobs.project_tag or knobs.contextual) else top_k
    from memo.tiers import REFERENCE_TYPES

    exclude_types = set(REFERENCE_TYPES) if flag_bool("MEMO_RECALL_EXCLUDE_REFERENCE") else None
    # Negative Recall (daemon parity, recall_logic._recall_excluded_types): drop
    # failure_pattern from normal recall so anti-memories surface only in the ⛔
    # AVOID block below. OFF ⇒ they flow into normal recall exactly as today.
    if flag_bool("MEMO_NEGATIVE_RECALL_ENABLED"):
        from memo.negative_recall import FAILURE_PATTERN_TYPE

        exclude_types = (exclude_types or set()) | {FAILURE_PATTERN_TYPE}
    # '_uncertain' quarantine (default on) — daemon parity (recall_logic).
    exclude_tags = uncertain_exclusion()
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

    # Ranking pipeline — identical to the daemon path (_recall_logic):
    # rank_hits with the hybrid cosine gate and preference boost, then the
    # shared skip-below/gap injection filters. Graph ordering already ran in search.
    _vec_cosine = make_vec_cosine(mem, prompt)

    _prefs: Any | None = None
    if knobs.contextual:
        with contextlib.suppress(Exception):
            _prefs = mem.contextual.context.get_preferences()

    # Epistemic gate for the empty marker: "memo has no record" may only be
    # asserted after a search that RAN successfully and qualified nothing. A
    # search exception (locked DB, store error) keeps the silent `{}` bail —
    # parity with the daemon path, whose search except in recall_logic returns
    # "{}" before the marker gate.
    _search_ok = False

    def _rank(
        query_text: str,
        *,
        _knobs: RankKnobs = knobs,
        _mode: str = mode,
        _vc: Callable[[Any], float | None] | None = _vec_cosine,
    ) -> list:
        nonlocal _search_ok
        try:
            hits = mem.search(
                query_text,
                limit=search_k,
                mode=_mode,
                recency=True,
                budget_ms=recall_search_budget_ms(),
                exclude_types=exclude_types,
                exclude_tags=exclude_tags,
            )
        except Exception as exc:
            # A vec/hybrid embed can stall on a busy daemon socket (capped to
            # _FALLBACK_EMBED_TIMEOUT_S above). Downgrade to bm25 — no embed, no
            # cold-load GPU fight — so recall stays within the hook budget
            # instead of bailing to an empty result.
            if _mode in ("vec", "hybrid"):
                if flag_bool("MEMO_RECALL_DEBUG"):
                    print(
                        f"# memo recall-hook: {_mode} embed failed ({exc}); bm25 fallback",
                        file=sys.stderr,
                    )
                return _rank(
                    query_text, _knobs=replace(_knobs, mode="bm25"), _mode="bm25", _vc=None
                )
            if flag_bool("MEMO_RECALL_DEBUG"):
                print(f"# memo recall-hook: search failed: {exc}", file=sys.stderr)
            return []
        _search_ok = True
        # Recency band (daemon parity, recall_logic): re-fetch recent hits above
        # the floor so freshness isn't lost to the similarity cut. Default OFF.
        # Skipped on the bm25 downgrade — fetch_recency_band would re-embed and
        # re-stall on the same busy daemon we just fell away from.
        _band_days = flag_int("MEMO_RECALL_RECENCY_BAND_DAYS") or 0
        if _band_days > 0 and _mode != "bm25":
            from memo.recall_logic import apply_recency_band, fetch_recency_band

            hits = apply_recency_band(
                hits,
                fetch_recency_band(
                    mem, days=_band_days, exclude_types=exclude_types, floor=_knobs.min_sim
                ),
            )
        # query=prompt (the original prompt, NOT query_text which may be the
        # expanded context) matches every daemon-path rank_hits call
        # (recall_logic._recall_logic): without it _is_broad_query(None) is
        # always False, so altitude-boost's broad= gate never fires here →
        # silent daemon/subprocess ranking divergence.
        return rank_hits(hits, _knobs, vec_cosine=_vc, preferences=_prefs, query=prompt)

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

    # Negative Recall (⛔ AVOID) — daemon parity (recall_logic). A preemptive,
    # high-precision pass over type=failure_pattern anti-memories, rendered as a
    # distinct block and excluded from the normal section above. Reuses the
    # cached query embedding; budget-gated; can_embed is False on the bm25
    # downgrade so it never cold-loads MLX. Default OFF ⇒ "". An ⛔ can fire even
    # when normal recall is empty, so it is computed before the empty returns.
    from memo.recall_logic import _avoid_only_output, _negative_recall_block

    _avoid_block = _negative_recall_block(
        mem,
        prompt,
        exclude_tags=exclude_tags,
        token_budget=token_budget,
        can_embed=(mode in ("vec", "hybrid")),
    )

    pre_filter = qualifying
    qualifying = apply_injection_filters(qualifying)
    # Unmatched-term gate (daemon parity, recall_logic): drop the whole injection
    # when no hit lexically covers the query's salient terms. Default OFF.
    if flag_bool("MEMO_RECALL_UNMATCHED_TERM_GATE") and qualifying:
        from memo.recall_logic import unmatched_term_gate

        if unmatched_term_gate(prompt, qualifying):
            qualifying = []

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
            cfg,
            mem,
            prompt=prompt,
            hits=qualifying,
            sim_threshold=_guard_sim_threshold,
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

    # Pre-top-K paraphrase collapse (daemon parity, recall_logic): drop lexical
    # near-dups from the over-fetched pool BEFORE truncation so they don't crowd
    # out distinct results. Default ON. Distinct from the post-top-K intra-dedup
    # below (MEMO_RECALL_INTRA_DEDUP), which can't recover a crowded-out result.
    if flag_bool("MEMO_RECALL_DEDUP_COLLAPSE") and len(qualifying) > 1:
        from memo.recall_logic import collapse_near_dups

        qualifying = collapse_near_dups(
            qualifying, threshold=flag_float("MEMO_RECALL_INTRA_DEDUP_THRESHOLD") or 0.8
        )

    relevant = qualifying[:top_k]

    # Precision gate (Lever 3): suppress when the top hit's score falls in a
    # band that has historically never been grounded.  Reads a small cached
    # JSON — cheap for the 5 s recall-hook budget.  Default OFF.
    # Decision: this suppresses an EXISTING record, so the empty-recall marker
    # ("no recorded memories") would be epistemically false here — the gate
    # stays a silent `{}` bail, same as the daemon path (recall_logic).
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
    # Omitted tail (rank-overflow beyond the nudge + injection-filtered hits),
    # computed exactly like the daemon path (recall_logic) so
    # MEMO_RECALL_OMISSIONS_TAIL counts identically across both paths.
    omitted = _rank_overflow_omitted(qualifying, pre_filter, top_k)

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
        # An ⛔ anti-memory can fire even when normal recall is empty — surface it
        # alone (daemon parity: recall_logic returns _avoid_only_output here).
        if _avoid_block:
            _close_memory()
            print(_avoid_only_output(_avoid_block))
            sys.exit(0)
        if not _search_ok:
            # No search ran successfully — absence of record is UNPROVEN, so
            # the epistemic marker must not fire. Silent `{}`, daemon parity.
            _bail("search failed — absence unproven, no marker")
            return
        # Search ran, nothing qualified — in a real session, emit the one-line
        # epistemic marker (MEMO_RECALL_EMPTY_MARKER, default on) instead of a
        # silent bail, so "memo has no record" is distinguishable from "memo
        # did not look". Pure formatting; the empty recall-log entry above
        # already recorded the event.
        if _sid:
            from memo.recall_logic import render_empty_recall_output

            _empty = render_empty_recall_output()
            if _empty is not None:
                if flag_bool("MEMO_RECALL_DEBUG"):
                    print(
                        f"# memo recall-hook: no hits above min_sim={knobs.min_sim}"
                        " — emitting empty marker",
                        file=sys.stderr,
                    )
                _close_memory()
                print(_empty)
                sys.exit(0)
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
        # Normal hits were all already recalled this session; an ⛔ that fired
        # this turn is still worth surfacing on its own (daemon parity).
        if _avoid_block:
            _close_memory()
            print(_avoid_only_output(_avoid_block))
            sys.exit(0)
        _bail("all hits already recalled this session")
        return

    from memo.recall_logic import (
        CITE_INSTRUCTION,
        render_by_format,
        resolve_recall_format,
    )

    def _est_tokens(s: str) -> int:
        return max(1, len(s) // 4)

    # Format steering — shared with the daemon path (recall_logic).
    _recall_format = resolve_recall_format(token_budget, len(relevant))
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

    _emitted: list[tuple[str, str]] = []
    context = render_by_format(
        _recall_format,
        relevant,
        nudge,  # rank-overflow nudge — mirrors the daemon path's top_k split
        turn=_turn,
        body_chars=body_chars,
        token_budget=token_budget,
        omitted=omitted,  # daemon parity — MEMO_RECALL_OMISSIONS_TAIL count
        disputed_by=_disputed_by,
        state_dir=cfg.state_dir,
        emitted_sink=_emitted,
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

    # ⛔ AVOID block sits at the very top — a distinct anti-memory warning above
    # the normal recall. Prepended last so it wins the topmost position (daemon
    # parity, recall_logic).
    if _avoid_block:
        context = f"{_avoid_block}\n\n{context}"

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
