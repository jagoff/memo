from __future__ import annotations

import json
import logging
import os
import re
import sys
import time

import click

from memo.config import Config
from memo.flags import flag_bool, flag_float, flag_int, flag_str

_log = logging.getLogger("memo.cli_recall_hook")

_RECALL_CONTEXTS: tuple[tuple[str, re.Pattern[str], set[str]], ...] = (
    (
        "code",
        re.compile(r"\b(implement|fix|debug|test|refactor|deploy|build|install)\b", re.I),
        {"decision", "bug", "preference"},
    ),
    (
        "decision",
        re.compile(r"\b(should i|which|choose|decide|recommend|tradeoff|vs\.?|versus)\b", re.I),
        {"decision", "fact"},
    ),
    (
        "write",
        re.compile(r"\b(write|document|explain|describe|summarize|draft)\b", re.I),
        {"note", "fact", "reference"},
    ),
)


@click.command(name="recall-hook")
def recall_hook() -> None:
    """UserPromptSubmit hook — inject relevant memorias as additionalContext."""
    try:
        cfg = Config.from_env()
    except Exception:
        print("{}")
        sys.exit(0)

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
        print("{}")
        sys.exit(0)

    if flag_bool("MEMO_RECALL_DISABLE"):
        _bail("disabled via MEMO_RECALL_DISABLE")
        return

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
    min_chars = flag_int("MEMO_RECALL_MIN_PROMPT_CHARS") or 12

    if flag_bool("MEMO_RECALL_SKIP_SLASH") and prompt.startswith("/"):
        head, _, rest = prompt[1:].partition(" ")
        rest = rest.strip()
        slash_min = flag_int("MEMO_RECALL_SLASH_MIN_ARG_CHARS") or 8
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
            except Exception:
                expanded = ""
        if len(expanded) >= min_chars:
            prompt = expanded
        else:
            _bail(f"prompt too short ({len(prompt)} < {min_chars})")
            return

    _client = flag_str("MEMO_RECALL_CLIENT")
    _turn: int | None = None
    if _sid:
        try:
            from memo import session as _session_mod

            _turn = _session_mod.next_turn(cfg.state_dir, _sid)
            _session_mod.stamp_recall_turn(cfg.state_dir, _sid, _turn)
        except Exception:
            _turn = None

    _t0 = time.time()
    try:
        from memo.recall_server import connect_and_recall

        _raw_float = flag_float("MEMO_RECALL_DAEMON_TIMEOUT")
        if _raw_float is not None and _raw_float >= 0.1:
            _daemon_timeout = _raw_float
        else:
            _daemon_timeout = max(0.2, (flag_int("MEMO_RECALL_DAEMON_TIMEOUT_MS") or 2000) / 1000.0)
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

    top_k = flag_int("MEMO_RECALL_TOP_K") or 3
    _ms = flag_float("MEMO_RECALL_MIN_SIM")
    min_sim = 0.5 if _ms is None else _ms
    body_chars = flag_int("MEMO_RECALL_BODY_CHARS") or 400
    token_budget = flag_int("MEMO_RECALL_TOKEN_BUDGET") or 0
    _pb = flag_float("MEMO_RECALL_PROJECT_BOOST")
    project_boost = 0.15 if _pb is None else _pb

    _session_mode = os.environ.get("MEMFLOW_SESSION_MODE", "").strip().lower()
    if _session_mode == "focus":
        top_k = min(top_k, 2)
        min_sim = max(min_sim, 0.65)
    elif _session_mode == "explore":
        top_k = max(top_k, 5)
        min_sim = min(min_sim, 0.4)
    elif _session_mode == "maintenance":
        top_k = 1
        min_sim = max(min_sim, 0.70)

    payload_cwd = payload.get("cwd")

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    mode = flag_str("MEMO_RECALL_MODE") or "vec"
    if mode == "hybrid":
        os.environ.setdefault(
            "MEMO_RERANK_INPUT_K",
            str(flag_int("MEMO_RECALL_RERANK_INPUT_K") or 10),
        )

    if mode in ("vec", "hybrid") and not flag_bool("MEMO_RECALL_FORCE_MODE"):
        try:
            _signal = cfg.state_dir / ".prewarm_ts"
            _warm = _signal.exists() and (time.time() - float(_signal.read_text().strip())) < 3600
            if not _warm:
                if flag_bool("MEMO_RECALL_DEBUG"):
                    print("# memo recall-hook: cold start — downgrading to bm25", file=sys.stderr)
                mode = "bm25"
        except Exception as exc:
            _log.debug("warm-signal read failed, staying in %s mode: %s", mode, exc)

    project_tag = None
    if project_boost > 0:
        try:
            from memo.project import current_project_tag

            project_tag = current_project_tag(payload_cwd)
        except Exception:
            project_tag = None
    search_k = top_k * 3 if project_tag else top_k
    from memo.tiers import REFERENCE_TYPES

    exclude_types = set(REFERENCE_TYPES) if flag_bool("MEMO_RECALL_EXCLUDE_REFERENCE") else None
    try:
        from memo.memory import Memory

        mem = Memory(cfg)
    except Exception as exc:
        _bail(f"search failed: {exc}")
        return

    _mbc = flag_int("MEMO_RECALL_MIN_BODY_CHARS")
    min_body_chars = 40 if _mbc is None else _mbc
    staleness_days = flag_int("MEMO_RECALL_STALENESS_DAYS") or 0

    def _search_filter(query_text: str) -> list:
        try:
            hits = mem.search(
                query_text, limit=search_k, mode=mode, recency=True, exclude_types=exclude_types
            )
        except Exception as exc:
            if flag_bool("MEMO_RECALL_DEBUG"):
                print(f"# memo recall-hook: search failed: {exc}", file=sys.stderr)
            return []
        if project_tag:
            from memo.recall_server import _apply_project_boost

            hits = _apply_project_boost(hits, project_tag, project_boost)
        hits = hits[:top_k]
        rel = [h for h in hits if h.score is None or h.score >= min_sim]
        if min_body_chars > 0:
            rel = [h for h in rel if len((h.body or "").strip()) >= min_body_chars]
        if staleness_days > 0:
            from datetime import UTC as _UTC
            from datetime import datetime as _dt

            _now = _dt.now(_UTC)
            stale_threshold = min_sim * 1.5
            filtered: list = []
            for h in rel:
                try:
                    updated = _dt.fromisoformat(h.updated)
                    if updated.tzinfo is None:
                        updated = updated.replace(tzinfo=_UTC)
                    days = (_now - updated).total_seconds() / 86400
                    if days > staleness_days and (h.score or 0.0) < stale_threshold:
                        continue
                except Exception:
                    pass
                filtered.append(h)
            rel = filtered
        from memo.recall_server import dedup_hits

        return dedup_hits(rel)

    relevant = _search_filter(prompt)

    if not relevant and flag_bool("MEMO_RECALL_EXPAND_CONTEXT"):
        from memo.recall_server import _session_context

        _ctx = _session_context(mem, exclude_types)
        if _ctx:
            relevant = _search_filter(f"{_ctx}\n{prompt}")
            if relevant and flag_bool("MEMO_RECALL_DEBUG"):
                print(
                    f"# memo recall-hook: query expansion recovered {len(relevant)} hits",
                    file=sys.stderr,
                )

    if relevant and flag_bool("MEMO_RECALL_ADAPTIVE_CONTEXT"):
        _boost_types: set[str] = set()
        for _ctx_name, _ctx_pat, _ctx_types in _RECALL_CONTEXTS:
            if _ctx_pat.search(prompt):
                _boost_types |= _ctx_types
                break
        if _boost_types:
            from dataclasses import replace as _dc_replace

            _boosted = [
                _dc_replace(h, score=round((h.score or 0.0) * 1.25, 6))
                if h.type in _boost_types
                else h
                for h in relevant
            ]
            _boosted.sort(key=lambda h: h.score or 0.0, reverse=True)
            relevant = _boosted

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
        _bail(f"no hits above min_sim={min_sim}")
        return

    # Session dedup: IDs already injected in earlier turns get a short reference
    # instead of full body, saving ~380 chars each time.
    _prev_recalled: dict[str, int] = {}
    if _sid and _turn is not None:
        try:
            from memo import session as _session_mod

            _prev_recalled = _session_mod.get_recalled_ids(cfg.state_dir, _sid)
        except Exception:
            _prev_recalled = {}

    from memo.recall_server import RECALL_DIRECTIVE, RECALL_FOOTER, RECALL_HEADER

    # Directive-once: skip 111-token directive after the first recall turn.
    include_directive = (
        _turn is None
        or _turn <= 1
        or not flag_bool("MEMO_RECALL_DIRECTIVE_ONCE")
    )

    footer = RECALL_FOOTER
    lines = [RECALL_HEADER]
    if include_directive:
        lines += [RECALL_DIRECTIVE, ""]
    else:
        lines.append("")

    def _est_tokens(s: str) -> int:
        return max(1, len(s) // 4)

    def _effective_body_chars(score: float | None) -> int:
        if not flag_bool("MEMO_RECALL_SCORE_ADAPTIVE_BODY") or score is None:
            return body_chars
        if score >= 0.85:
            return int(body_chars * 1.5)
        if score < 0.65:
            return max(80, body_chars // 2)
        return body_chars

    # Budget governs hit content only — deduct fixed overhead so the token cap
    # isn't silently exceeded by header/directive/footer.
    budget_chars_for_hits: int | None
    if token_budget > 0:
        overhead = sum(len(ln) + 1 for ln in lines) + len(footer) + 1 + 80
        budget_chars_for_hits = max(0, token_budget * 4 - overhead)
    else:
        budget_chars_for_hits = None

    used_chars = 0

    for h in relevant:
        prev_turn = _prev_recalled.get(h.id)
        if prev_turn is not None:
            ref_line = f"**[{h.id[:8]}] {h.title}** _(ya citado, turno {prev_turn})_"
            block_lines = [ref_line, ""]
            block = "\n".join(block_lines)
            if budget_chars_for_hits is None:
                lines.extend(block_lines)
            else:
                remaining = budget_chars_for_hits - used_chars
                if remaining <= 0:
                    break
                if len(block) <= remaining:
                    lines.extend(block_lines)
                    used_chars += len(block)
            continue

        eff_body = _effective_body_chars(h.score)
        score_tag = f" (score {h.score:.2f})" if h.score is not None else ""
        body = (h.body or "").strip().replace("\n", " ")
        if len(body) > eff_body:
            body = body[:eff_body].rstrip() + "…"
        block_lines = [f"**[{h.id[:8]}] {h.title}**{score_tag}"]
        if h.tags:
            block_lines.append(f"_tags_: {', '.join(h.tags)}")
        if body:
            block_lines.append(f"> {body}")
        block_lines.append("")
        block = "\n".join(block_lines)

        if budget_chars_for_hits is None:
            lines.extend(block_lines)
            continue

        remaining = budget_chars_for_hits - used_chars
        if remaining <= 0:
            break
        if len(block) <= remaining:
            lines.extend(block_lines)
            used_chars += len(block)
        else:
            if body:
                head_len = len(block_lines[0]) + 1
                tags_len = (len(block_lines[1]) + 1) if h.tags else 0
                avail = max(0, remaining - head_len - tags_len - 3)
                if avail > 20:
                    trunc_body = body[:avail].rstrip() + "…"
                    block_lines[-2 if h.tags else -1] = f"> {trunc_body}"
                    lines.extend(block_lines)
            break

    lines.append(footer)
    if token_budget > 0 and flag_bool("MEMO_RECALL_DEBUG"):
        approx = _est_tokens("\n".join(lines))
        print(f"# memo recall-hook: ~{approx} tokens (budget {token_budget})", file=sys.stderr)

    # Prepend any pending notification from a previous async idle-maintenance run.
    pending_notif_path = cfg.state_dir / "pending_idle_notification.txt"
    if pending_notif_path.exists():
        try:
            notif = pending_notif_path.read_text(encoding="utf-8").strip()
            if notif:
                lines = [notif, ""] + lines
            pending_notif_path.unlink(missing_ok=True)
        except Exception:
            pass

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "\n".join(lines),
        }
    }
    print(json.dumps(output, ensure_ascii=False))

    # Persist newly recalled IDs so future turns can dedup them
    if _sid and _turn is not None and relevant:
        try:
            from memo import session as _session_mod

            new_ids = {h.id: _turn for h in relevant if h.id not in _prev_recalled}
            _session_mod.mark_ids_recalled(cfg.state_dir, _sid, new_ids)
        except Exception:
            pass

    sys.exit(0)
