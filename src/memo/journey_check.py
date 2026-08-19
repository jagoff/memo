"""User-journey verification harness — the engine behind ``memo journey-check``.

Orchestration only. A small set of single-purpose ``Check`` functions each
exercise ONE real user-facing code path (capture, recall, ask, the token
ledger, recall UX messages, Day-0 install, live wiring) against a **seeded,
isolated** store — a throwaway ``MEMO_DATA_DIR`` / ``MEMO_STATE_DIR`` — plus
two read-only smokes against the real install. The live corpus is never
touched. ``run_all`` aggregates the ``CheckResult``s and computes an exit code
(nonzero on any ``fail``).

Design notes:
- Each check reuses existing internals (``capture_core._extract_and_save``,
  ``recall_logic._recall_logic``, ``Memory.ask``, ``token_ledger``,
  ``cli_diag`` doctor signals). Nothing here re-implements retrieval or capture.
- MLX invariant: ``mlx`` / ``mlx-lm`` imports stay deferred; embedding-dependent
  checks return ``skip`` when ``mlx_lm`` is not importable (non-Apple-Silicon).
- A check that raises never crashes the run — it becomes a ``fail`` carrying the
  exception string as evidence.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

# ── Result vocabulary ────────────────────────────────────────────────────────
PASS = "pass"  # noqa: S105 — status label, not a credential
FAIL = "fail"
WARN = "warn"
SKIP = "skip"
_STATUSES = frozenset({PASS, FAIL, WARN, SKIP})

# Distinct nonce tokens seeded into the isolated store so a recall/ask hit is
# unambiguous and a negative prompt's cleanliness is a plain substring check.
_FACT_NONCE = "ZPH-JOURNEYCHECK-4F2A9"
_INSTALL_NONCE = "JOURNEYCHECK-INSTALL-8B1C"

# Matching / non-matching prompts for the seeded fact. The negative prompt shares
# no lexical or semantic overlap with the target OR any decoy, so a correctly
# gated recall returns nothing for it.
_MATCH_PROMPT = "What is the deploy token for the Zephyr project?"
_UNRELATED_PROMPT = "What is the tallest mountain in the world and how tall is it?"
_UNKNOWN_ATTR_PROMPT = "What is the Zephyr project's Slack webhook URL?"

# Decoy memories on unrelated technical topics, seeded ALONGSIDE the target so the
# store is representative (not a single-memory corpus). A single-memory store is
# unrepresentative: with MEMO_RECALL_EXPAND_CONTEXT on, an empty-gate recovery
# re-queries with the session's only memory (the target itself), surfacing it for
# any prompt — a false "leak" that never reproduces against a real corpus.
_DECOYS: list[tuple[str, str, str]] = [
    (
        "Postgres connection pool sizing",
        "Set pgbouncer default_pool_size to 20 per app instance to avoid exhausting "
        "Postgres max_connections under burst load.",
        "decision",
    ),
    (
        "React useEffect cleanup leak",
        "A missing cleanup return in useEffect leaked websocket listeners; always "
        "return an unsubscribe function from the effect.",
        "bug",
    ),
    (
        "Kubernetes pod resource limits",
        "Each API pod requests 250m CPU and 512Mi memory with limits at double that "
        "so it survives burst traffic without eviction.",
        "fact",
    ),
    (
        "Rust lifetime elision",
        "The borrow checker elides lifetimes on single-input-reference functions, so "
        "most getters need no explicit annotation.",
        "note",
    ),
    (
        "GraphQL N+1 batching with dataloader",
        "We batch nested resolver reads through a per-request dataloader to collapse "
        "the N+1 into a single SQL round trip.",
        "decision",
    ),
    (
        "Preferred code-review tone",
        "Prefer terse, evidence-first code review comments: location, problem, fix, "
        "with no praise padding.",
        "preference",
    ),
]

_RECALL_BUDGET_S = 5.0
_TOKEN_SAVINGS_RECALLS = 5


@dataclass
class CheckResult:
    """One check's verdict. ``status`` ∈ {pass, fail, warn, skip}."""

    name: str
    status: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


# A Check is a callable taking the shared context and returning a CheckResult.
Check = Callable[["JourneyContext"], CheckResult]


def _mlx_available() -> bool:
    """True when the MLX runtime is importable (deferred import — invariant)."""
    try:
        import mlx_lm  # noqa: F401
    except Exception:
        return False
    return True


class JourneyContext:
    """Isolated, seeded store for the store-dependent checks.

    Builds a ``Config`` rooted at fresh tmp dirs (never ``from_env`` — that would
    read the developer's real data_dir/markdown config), forces in-process
    embedding (``MEMO_EMBEDDER_VIA_DAEMON=0`` so a warm socket at the real
    state_dir can't be consulted for the tmp store), warms the embedder so
    latency measurements are warm, and seeds one known fact. Teardown removes the
    tmp tree and restores the env.
    """

    def __init__(self, *, need_store: bool = True) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="memo-journeycheck-"))
        self.mlx = _mlx_available()
        self.mem: Any | None = None
        self.seeded: dict[str, Any] = {}
        # Force in-process embedding (so a warm socket at the real state_dir can't
        # be consulted for the tmp store) via a managed set/restore. patch.dict
        # snapshots the prior value and restores it on close() — no raw os.environ
        # read of a behavior flag (facade.py owns the three-way daemon decision).
        self._via_daemon_patch = mock.patch.dict(os.environ, {"MEMO_EMBEDDER_VIA_DAEMON": "0"})
        self._via_daemon_patch.start()

        from memo.config import Config

        data = self._tmp / "data"
        state = self._tmp / "state"
        data.mkdir(parents=True)
        state.mkdir(parents=True)
        self.cfg = Config(data_dir=data, vault_path=None, state_dir=state, reranker_enabled=False)
        if need_store and self.mlx:
            self._setup_store()

    def _setup_store(self) -> None:
        from memo.memory import Memory

        self.mem = Memory(self.cfg)
        # Pay the cold MLX load now so per-check latency is measured warm.
        with contextlib.suppress(Exception):
            self.mem.embedder.embed_query("journey-check warmup")
        rec = self.mem.save(
            content=(
                f"The deploy token for the Zephyr project is {_FACT_NONCE}. "
                "Use it only for CI releases, and rotate it after each launch."
            ),
            title="Zephyr project deploy token",
            type_="fact",
            tags=["project:zephyr-journeycheck", "deploy"],
            auto_project=False,
        )
        self.seeded["fact_id"] = rec.id
        self.seeded["fact_nonce"] = _FACT_NONCE
        # Decoys make the store representative so a negative prompt actually gates
        # to empty (see _DECOYS). The target still dominates the match prompt.
        for title, body, type_ in _DECOYS:
            self.mem.save(
                content=body,
                title=title,
                type_=type_,
                tags=["journeycheck-decoy"],
                auto_project=False,
            )
        self.seeded["decoy_count"] = len(_DECOYS)

    def close(self) -> None:
        if self.mem is not None:
            with contextlib.suppress(Exception):
                self.mem.close()
        with contextlib.suppress(RuntimeError):
            self._via_daemon_patch.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def __enter__(self) -> JourneyContext:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _skip_no_mlx(name: str) -> CheckResult:
    return CheckResult(name, SKIP, "MLX runtime unavailable (mlx_lm not importable)")


def _recall(
    ctx: JourneyContext, prompt: str, *, session_id: str | None = None
) -> tuple[dict, float]:
    """Run the real recall pipeline in-process and return (parsed_output, latency_s).

    Uses ``recall_logic._recall_logic`` — the same entry the warm recall daemon
    serves — and invokes its deferred logger so the consult lands in recall.log
    (feeds the token ledger). Returns the parsed hook-JSON dict (``{}`` on an
    empty recall) so callers can inspect ``additionalContext`` / ``systemMessage``.
    """
    from memo.recall_logic import _recall_logic

    t0 = time.time()
    out, deferred = _recall_logic(
        prompt,
        None,
        ctx.mem,
        ctx.cfg,
        session_id=session_id,
        t0=t0,
        client="journey-check",
    )
    latency = time.time() - t0
    if deferred is not None:
        with contextlib.suppress(Exception):
            deferred()
    parsed: dict = {}
    with contextlib.suppress(Exception):
        loaded = json.loads(out)
        if isinstance(loaded, dict):
            parsed = loaded
    return parsed, latency


def _additional_context(parsed: dict) -> str:
    hook = parsed.get("hookSpecificOutput")
    if isinstance(hook, dict):
        return str(hook.get("additionalContext") or "")
    return ""


# ── Checks ───────────────────────────────────────────────────────────────────
def check_auto_save(ctx: JourneyContext) -> CheckResult:
    """Auto-save: a synthetic exchange through the real capture path saves durable
    memories, and a hallucinated ``type='state'`` insight is COERCED (not dropped
    — guards PR #99)."""
    if not ctx.mlx or ctx.mem is None:
        return _skip_no_mlx("auto-save")

    from memo import capture_core
    from memo.memory.record import _VALID_TYPES

    def _stub_extract(*_a: Any, **_k: Any) -> list[dict[str, Any]]:
        return [
            {
                "title": "Adopt Qwen3-4B embeddings for memo recall",
                "type": "decision",
                "body": (
                    "We decided to switch memo's embedder to the Qwen3-4B model "
                    "because it lifted same-topic recall precision by twelve percent "
                    "on the committed regression set."
                ),
                "tags": ["memo", "mlx", "embeddings"],
            },
            {
                # Invalid type the extract prompt's state-change guidance invites.
                "title": "memo recall daemon stays warm over a socket",
                "type": "state",
                "body": (
                    "The recall daemon keeps a warm MLX embedder resident behind a "
                    "unix socket so the UserPromptSubmit hook stays under its five "
                    "second budget on every single turn."
                ),
                "tags": ["memo", "daemon", "recall"],
            },
        ]

    _orig = capture_core.extract_insights
    capture_core.extract_insights = _stub_extract  # type: ignore[assignment]
    try:
        result = capture_core._extract_and_save(
            ctx.mem,
            ctx.cfg,
            "How did we improve memo recall and keep the hook fast?",
            "We switched to Qwen3-4B and the recall daemon stays warm over a socket.",
            auto_project=False,
        )
    finally:
        capture_core.extract_insights = _orig  # type: ignore[assignment]

    saved = list(result.get("saved") or [])
    failures = int(result.get("save_failures") or 0)
    records = list(result.get("saved_records") or [])
    types = [str(r.get("type") or "") for r in records]
    coerced_ok = bool(records) and all(t in _VALID_TYPES for t in types)
    # Both candidates (incl. the invalid 'state') must reach the store with 0
    # failures; without coercion the 'state' one would raise inside save().
    ok = len(saved) >= 2 and failures == 0 and coerced_ok
    status = PASS if ok else FAIL
    detail = f"capture saved {len(saved)}, {failures} failed; types={types}"
    return CheckResult(
        "auto-save",
        status,
        detail,
        {"saved": len(saved), "save_failures": failures, "saved_types": types},
    )


def check_auto_recall(ctx: JourneyContext) -> CheckResult:
    """Auto-recall: the matching prompt surfaces the seeded id within the 5s warm
    budget, and an unrelated prompt surfaces no canary."""
    if not ctx.mlx or ctx.mem is None:
        return _skip_no_mlx("auto-recall")

    nonce = str(ctx.seeded["fact_nonce"])
    fid = str(ctx.seeded["fact_id"])
    parsed, latency = _recall(ctx, _MATCH_PROMPT)
    ctx_text = _additional_context(parsed)
    surfaced = nonce in ctx_text or fid[:8] in ctx_text
    fast = latency < _RECALL_BUDGET_S

    neg_parsed, _ = _recall(ctx, _UNRELATED_PROMPT)
    neg_text = _additional_context(neg_parsed)
    neg_clean = nonce not in neg_text and fid[:8] not in neg_text

    ok = surfaced and fast and neg_clean
    status = PASS if ok else FAIL
    detail = (
        f"seeded id {'top-K' if surfaced else 'MISSING'}, {latency:.2f}s; "
        f"negative {'clean' if neg_clean else 'LEAKED'}"
    )
    return CheckResult(
        "auto-recall",
        status,
        detail,
        {
            "surfaced": surfaced,
            "latency_s": round(latency, 3),
            "within_budget": fast,
            "negative_clean": neg_clean,
        },
    )


def check_uses_memory(ctx: JourneyContext) -> CheckResult:
    """Uses-memory: ``Memory.ask`` on a seeded fact returns the seeded value AND
    cites the id, and abstains on an unknown attribute."""
    if not ctx.mlx or ctx.mem is None:
        return _skip_no_mlx("uses-memory")

    nonce = str(ctx.seeded["fact_nonce"])
    fid = str(ctx.seeded["fact_id"])
    ans = ctx.mem.ask(_MATCH_PROMPT, k=5)
    answer = str(ans.get("answer") or "")
    sources = list(ans.get("sources") or [])
    contains_value = nonce in answer
    cited_ids = {str(s.get("id") or "") for s in sources}
    cites_id = (
        fid[:8] in answer
        or fid in cited_ids
        or any(sid and (fid.startswith(sid) or sid.startswith(fid[:8])) for sid in cited_ids)
    )

    ab = ctx.mem.ask(_UNKNOWN_ATTR_PROMPT, k=5)
    ab_answer = str(ab.get("answer") or "")
    lowered = ab_answer.lower()
    _abstain_markers = (
        "couldn't find",
        "could not find",
        "don't have",
        "do not have",
        "no record",
        "not find",
        "n't find",
        "no information",
    )
    abstained = any(m in lowered for m in _abstain_markers) or not ab.get("sources")

    ok = contains_value and cites_id and abstained
    status = PASS if ok else FAIL
    detail = (
        f"value={'yes' if contains_value else 'NO'}, "
        f"cite={'yes' if cites_id else 'NO'}, "
        f"abstain={'yes' if abstained else 'NO'}"
    )
    return CheckResult(
        "uses-memory",
        status,
        detail,
        {
            "contains_value": contains_value,
            "cites_id": cites_id,
            "abstained": abstained,
            "answer": answer[:400],
            "abstain_answer": ab_answer[:400],
        },
    )


def check_token_savings(ctx: JourneyContext) -> CheckResult:
    """Token-savings: N grounded recalls move the durable ledger's grounded
    count by Δ > 0 (a real, physical count — memo no longer converts it to a
    "tokens saved" estimate; see CHANGELOG)."""
    if not ctx.mlx or ctx.mem is None:
        return _skip_no_mlx("token-savings")

    from memo.token_ledger import roll_up, summarize

    roll_up(ctx.cfg.state_dir)
    before = int(summarize(ctx.cfg.state_dir)["historic"]["grounded"])

    logged = 0
    for _ in range(_TOKEN_SAVINGS_RECALLS):
        parsed, _lat = _recall(ctx, _MATCH_PROMPT)
        if _additional_context(parsed):
            logged += 1

    roll_up(ctx.cfg.state_dir)
    after = int(summarize(ctx.cfg.state_dir)["historic"]["grounded"])
    delta = after - before

    ok = delta > 0
    status = PASS if ok else FAIL
    detail = f"Δ +{delta:,} grounded / {logged} grounded recalls"
    return CheckResult(
        "token-savings",
        status,
        detail,
        {"before": before, "after": after, "delta": delta, "recalls_logged": logged},
    )


def check_ux_messages(ctx: JourneyContext) -> CheckResult:
    """UX-messages: recall emits additionalContext + a systemMessage, capture
    writes the pending-notification file, and ``briefing`` renders non-empty.
    Emits ``warn`` for the known Claude-Code capture-receipt gap."""
    if not ctx.mlx or ctx.mem is None:
        return _skip_no_mlx("ux-messages")

    parsed, _lat = _recall(ctx, _MATCH_PROMPT, session_id="jc-ux")
    add_ctx = _additional_context(parsed)
    sysmsg = str(parsed.get("systemMessage") or "")
    has_ctx = bool(add_ctx.strip())
    has_sys = bool(sysmsg.strip())

    from memo.cli_capture import _write_capture_notification

    _write_capture_notification(
        ctx.cfg.state_dir,
        [{"id": "abcd1234", "title": "Zephyr deploy token", "type": "note"}],
    )
    notif = ctx.cfg.state_dir / "pending_idle_notification.txt"
    has_notif = notif.is_file() and bool(notif.read_text(encoding="utf-8").strip())

    briefing_ok, briefing_err = _briefing_renders(ctx)

    core_ok = has_ctx and has_sys and has_notif and briefing_ok
    evidence: dict[str, Any] = {
        "additional_context": has_ctx,
        "system_message": has_sys,
        "notification_file": has_notif,
        "briefing_renders": briefing_ok,
    }
    if briefing_err:
        evidence["briefing_error"] = briefing_err
    if not core_ok:
        missing = [k for k, v in evidence.items() if v is False]
        return CheckResult(
            "ux-messages", FAIL, f"missing UX signal(s): {', '.join(missing)}", evidence
        )
    # All channels delivered — the only remaining gap is the capture receipt not
    # rendering natively on Claude Code, which is a known WARN, not a failure.
    return CheckResult(
        "ux-messages",
        WARN,
        "delivered; capture receipt not shown natively on Claude Code",
        evidence,
    )


def _briefing_renders(ctx: JourneyContext) -> tuple[bool, str]:
    """Invoke the real ``briefing`` command against the seeded store and report
    whether it rendered non-empty output. Returns (ok, error_str)."""
    try:
        from click.testing import CliRunner

        from memo.cli_briefing import briefing

        env = {
            "MEMO_DATA_DIR": str(ctx.cfg.data_dir),
            "MEMO_STATE_DIR": str(ctx.cfg.state_dir),
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_EMBEDDER_VIA_DAEMON": "0",
        }
        result = CliRunner().invoke(briefing, [], env=env, catch_exceptions=True)
        if result.exit_code != 0:
            return False, f"exit {result.exit_code}"
        return bool((result.output or "").strip()), ""
    except Exception as exc:  # never let briefing break the check
        return False, f"{type(exc).__name__}: {exc}"


def check_install(ctx: JourneyContext) -> CheckResult:
    """Install (Day-0): fresh tmp HOME + the already-installed binary — onboard →
    doctor (no CRITICAL) → first save → first recall surfaces it."""
    binary = shutil.which("memo")
    if not binary:
        return CheckResult("install", SKIP, "`memo` binary not on PATH")

    home = Path(tempfile.mkdtemp(prefix="memo-journeycheck-home-"))
    try:
        env = _fresh_home_env(home)
        steps: dict[str, Any] = {}

        onboard = _run_memo(binary, ["onboard", "--yes", "--json"], env, timeout=240)
        steps["onboard_exit"] = onboard.returncode

        doctor = _run_memo(binary, ["doctor", "--json"], env, timeout=180)
        doctor_ok, doctor_note = _doctor_no_critical(doctor.stdout)
        steps["doctor_ok"] = doctor_ok
        steps["doctor_note"] = doctor_note

        save = _run_memo(
            binary,
            [
                "save",
                f"The Zephyr CI deploy setting is keyed {_INSTALL_NONCE} for launch day.",
                # Nonce in the TITLE too: the compact recall format truncates the
                # body (~50 chars) and would cut the nonce mid-token; the title
                # renders in full, so the surface check stays robust.
                "--title",
                f"Zephyr deploy {_INSTALL_NONCE}",
                "--type",
                "fact",
            ],
            env,
            timeout=180,
        )
        steps["save_exit"] = save.returncode

        recall_in = json.dumps(
            {
                "prompt": f"Tell me about the {_INSTALL_NONCE} deploy setting for Zephyr",
                "session_id": "jc-install",
                "cwd": str(home),
            }
        )
        recall = _run_memo(binary, ["recall-hook"], env, stdin=recall_in, timeout=180)
        recall_surfaced = _INSTALL_NONCE in _recall_hook_context(recall.stdout)
        steps["recall_surfaced"] = recall_surfaced

        ok = onboard.returncode == 0 and doctor_ok and save.returncode == 0 and recall_surfaced
        status = PASS if ok else FAIL
        detail = (
            f"onboard exit={onboard.returncode}, doctor={'ok' if doctor_ok else 'CRITICAL'}, "
            f"save exit={save.returncode}, recall {'ok' if recall_surfaced else 'MISSING'}"
        )
        if not ok:
            steps["doctor_stderr"] = (doctor.stderr or "")[:300]
            steps["save_stderr"] = (save.stderr or "")[:300]
            steps["recall_stderr"] = (recall.stderr or "")[:300]
            steps["recall_stdout"] = (recall.stdout or "")[:400]
        return CheckResult("install", status, detail, steps)
    finally:
        shutil.rmtree(home, ignore_errors=True)


def check_live_wiring(ctx: JourneyContext) -> CheckResult:
    """Live-wiring (read-only, real install): recall hook wired in settings.json,
    recall-daemon warm, and memo/memo-mcp resolve to the same runtime."""
    from memo.config import Config

    hook_ok = False
    with contextlib.suppress(Exception):
        from memo.cli_hooks import recall_hook_wired

        hook_ok = bool(recall_hook_wired())

    daemon_running = False
    with contextlib.suppress(Exception):
        from memo.cli_diag import _recall_daemon_health

        daemon_running = bool(_recall_daemon_health(Config.from_env()).get("running"))

    mixed_runtime = False
    runtime_warnings: list[str] = []
    with contextlib.suppress(Exception):
        from memo.runtime.detect import _runtime_install_report

        runtime_warnings = [str(w) for w in _runtime_install_report().get("warnings", [])]
        mixed_runtime = any("different environments" in w for w in runtime_warnings)

    evidence = {
        "hook_wired": hook_ok,
        "daemon_running": daemon_running,
        "mixed_runtime": mixed_runtime,
        "runtime_warnings": runtime_warnings,
    }
    if not hook_ok or mixed_runtime:
        problem = "recall hook NOT wired" if not hook_ok else "memo/memo-mcp mixed runtime"
        return CheckResult("live-wiring", FAIL, problem, evidence)
    if not daemon_running:
        return CheckResult(
            "live-wiring",
            WARN,
            "hook wired, runtime ok; recall-daemon not warm (subprocess fallback)",
            evidence,
        )
    return CheckResult("live-wiring", PASS, "hook wired, daemon warm, same runtime", evidence)


# ── Install-check subprocess helpers ─────────────────────────────────────────
def _fresh_home_env(home: Path) -> dict[str, str]:
    """A Day-0 environment: no inherited MEMO_* config, a throwaway HOME, and
    fresh empty data/state/config dirs. Network + banner disabled."""
    # Strip MEMO_* AND CLAUDE_* inherited config. A leaked CLAUDE_CONFIG_DIR is
    # the dangerous one: the spawned `memo onboard` calls wire_recall_hook, whose
    # _claude_dir() honors $CLAUDE_CONFIG_DIR *before* the HOME-derived path — so
    # without stripping it the "isolated" subprocess would read-modify-write the
    # developer's REAL settings.json, and the tmp-home cleanup would not revert it.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("MEMO_") and not k.startswith("CLAUDE_")
    }
    data = home / "data"
    state = home / "state"
    config = home / "config"
    claude = home / ".claude"
    for d in (data, state, config, claude):
        d.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(home),
            # Pin Claude Code's config dir inside the sandbox so hook-wiring during
            # `memo onboard` can never escape to the real install.
            "CLAUDE_CONFIG_DIR": str(claude),
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_DATA_DIR": str(data),
            "MEMO_STATE_DIR": str(state),
            "MEMO_CONFIG_DIR": str(config),
            "MEMO_AUTO_UPDATE": "0",
            "MEMO_STARTUP_BANNER": "0",
            "MEMO_EMBEDDER_VIA_DAEMON": "0",
        }
    )
    return env


def _run_memo(
    binary: str,
    args: list[str],
    env: dict[str, str],
    *,
    stdin: str | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [binary, *args],
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _doctor_no_critical(stdout: str) -> tuple[bool, str]:
    """Parse ``memo doctor --json`` and pass when no CRITICAL signal fired: every
    capability import loaded and storage is present. Missing daemon / cold hook
    are non-critical and ignored here."""
    try:
        report = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return False, "doctor did not emit JSON"
    imports = report.get("imports") or []
    failed = [str(i.get("name")) for i in imports if not i.get("ok", True)]
    storage = report.get("storage") or {}
    data_ok = bool((storage.get("data_dir") or {}).get("ok", True))
    if failed:
        return False, f"failed imports: {', '.join(failed)}"
    if not data_ok:
        return False, "data_dir missing"
    return True, "imports+storage ok"


def _recall_hook_context(stdout: str) -> str:
    """Extract additionalContext from a ``memo recall-hook`` stdout line."""
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                return _additional_context(parsed)
    return ""


# ── Orchestration ────────────────────────────────────────────────────────────
_CHECKS: list[tuple[str, Check]] = [
    ("auto-save", check_auto_save),
    ("auto-recall", check_auto_recall),
    ("uses-memory", check_uses_memory),
    ("token-savings", check_token_savings),
    ("ux-messages", check_ux_messages),
    ("install", check_install),
    ("live-wiring", check_live_wiring),
]
CHECK_NAMES: list[str] = [name for name, _ in _CHECKS]

# Checks that read the seeded isolated store (the rest run against the real
# install or a fresh tmp HOME and need no seeding).
_STORE_CHECKS = frozenset(
    {"auto-save", "auto-recall", "uses-memory", "token-savings", "ux-messages"}
)


def compute_exit_code(results: list[CheckResult]) -> int:
    """Nonzero iff any check failed. warn / skip / pass do not fail the gate."""
    return 1 if any(r.status == FAIL for r in results) else 0


def run_all(only: list[str] | None = None) -> tuple[list[CheckResult], int]:
    """Run the selected checks against a seeded isolated store, aggregate, and
    compute the exit code. A raising check becomes a ``fail`` (never crashes)."""
    selected = [(n, fn) for n, fn in _CHECKS if only is None or n in only]
    need_store = any(n in _STORE_CHECKS for n, _ in selected)
    results: list[CheckResult] = []
    with JourneyContext(need_store=need_store) as ctx:
        for name, fn in selected:
            try:
                result = fn(ctx)
            except Exception as exc:
                result = CheckResult(
                    name, FAIL, f"raised {type(exc).__name__}: {exc}", {"exception": str(exc)}
                )
            if result.status not in _STATUSES:
                result = CheckResult(name, FAIL, f"invalid status {result.status!r}")
            results.append(result)
    return results, compute_exit_code(results)
