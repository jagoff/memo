"""The recall daemon's LaunchAgent sets annotation flags it cannot honour.

`launchd/com.memo.recall-daemon.plist` exports ``MEMO_HIT_DOSSIER=1`` and
``MEMO_RECALL_EPISTEMIC_LABELS=1``. Both are read by ``render_recall_context``
(full) and ``render_recall_compact`` — and by neither ``render_recall_balanced``.

The daemon is a separate long-lived process, so it does NOT inherit the recall
hook's environment, and the socket request carries only prompt/cwd/session_id/
turn/client (``recall_client.connect_and_recall``) — no budget, no format. It
therefore resolves ``MEMO_RECALL_TOKEN_BUDGET`` / ``_TOP_K`` / ``_FORMAT`` from
its OWN flag chain, which on a default install yields 600 / 3 / auto. Adaptive
budgeting maps 600 to {800, 600, 360} by prompt length and hits are capped at
top_k, so *every* reachable combination resolves to ``balanced`` — `full` needs
a budget strictly above 800 and the maximum attainable is exactly 800.

Net: on the daemon path those two flags are set-but-unreadable. On the
subprocess fallback they are honoured, because the installed hook command
exports ``MEMO_RECALL_TOKEN_BUDGET=160``, which resolves to ``compact``.
These tests pin both halves of that asymmetry.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from memo.memory import MemoryRecord
from memo.recall_logic import inert_annotation_flags, reachable_recall_formats
from memo.recall_server import _recall_logic

# The budget/top_k the daemon resolves from its own chain on a default install.
_DAEMON_BUDGET = 600
_DAEMON_TOP_K = 3
# What the installed hook command exports for the subprocess fallback
# (see cli_hooks._HOOK_ENV).
_HOOK_BUDGET = 160
_HOOK_TOP_K = 1


def _production_daemon_env(monkeypatch) -> None:
    """Exactly what the live recall-daemon LaunchAgent puts in the environment.

    Deliberately does NOT pin MEMO_RECALL_TOKEN_BUDGET / _TOP_K / _FORMAT: the
    plist sets none of them, and the whole point is what the daemon resolves on
    its own. They are cleared so a developer's shell export cannot fake a pass.
    """
    for name in (
        "MEMO_RECALL_TOKEN_BUDGET",
        "MEMO_RECALL_TOP_K",
        "MEMO_RECALL_FORMAT",
        "MEMO_RECALL_SESSION_TOKEN_BUDGET",
        "MEMO_RECALL_CONFIDENCE_GATE",
        "MEMO_RECALL_ADAPTIVE_BUDGET",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMO_HIT_DOSSIER", "1")
    monkeypatch.setenv("MEMO_RECALL_EPISTEMIC_LABELS", "1")


def _rec(id_: str, title: str, score: float) -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        path=f"notes/{id_}.md",
        title=title,
        type="note",
        tags=[],
        created="2026-05-21T00:00:00+00:00",
        updated="2026-05-21T00:00:00+00:00",
        body="cuerpo de la memoria. " * 12,
        extra={},
        score=score,
    )


class _OneHitStubMemory:
    def __init__(self, hit: MemoryRecord) -> None:
        self._hit = hit

    def search(
        self,
        query,
        limit,
        mode,
        recency=False,
        budget_ms=None,
        exclude_types=None,
        exclude_tags=None,
    ):
        return [self._hit]


# ---------------------------------------------------------------------------
# The real daemon path (`_recall_logic` is what recall_socket.handle calls)
# ---------------------------------------------------------------------------


def test_daemon_path_renders_balanced_and_drops_the_plist_annotation_flags(
    monkeypatch, tmp_path
) -> None:
    """Production daemon config + both plist flags ON -> neither is rendered.

    The prompt is deliberately under 50 chars: adaptive budgeting scales the
    daemon's 600 up to 800, the LARGEST budget it can ever reach and the case
    closest to `full`. `full` requires ``token_budget > 800``, so even here the
    daemon lands on `balanced` — the flags cannot be honoured on any turn.
    """
    _production_daemon_env(monkeypatch)
    # Retrieval gates only: let the stub hit through. Neither touches format.
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    prompt = "que hace el gate pre-push"
    assert len(prompt) < 50, "prompt must hit the budget-1.5x branch"

    result, _log = _recall_logic(
        prompt,
        cwd=None,
        mem=_OneHitStubMemory(_rec("dossier1", "Daemon annotation fact", 0.9)),
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
    )
    context = json.loads(result)["hookSpecificOutput"]["additionalContext"]

    # Balanced shape: "## Memory" + "- [id] Title" (never "**[id] Title**").
    assert "- [dossier1]" in context, context
    assert "**[dossier1]" not in context, context
    # MEMO_HIT_DOSSIER is ON and produced nothing.
    assert "_trust_:" not in context, context
    # MEMO_RECALL_EPISTEMIC_LABELS is ON and produced nothing.
    assert "⟨" not in context, context


def test_hook_subprocess_budget_reaches_compact_which_does_render_them(
    monkeypatch, tmp_path
) -> None:
    """Counterpart: the same flags ARE honoured when the budget picks compact.

    Guards the pinning test above against passing for a boring reason (e.g. the
    annotations silently never render anywhere). Uses the budget the installed
    hook command actually exports, not a synthetic one.
    """
    _production_daemon_env(monkeypatch)
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.setenv("MEMO_RECALL_TOKEN_BUDGET", str(_HOOK_BUDGET))
    monkeypatch.setenv("MEMO_RECALL_TOP_K", str(_HOOK_TOP_K))

    result, _log = _recall_logic(
        "que hace el gate pre-push",
        cwd=None,
        mem=_OneHitStubMemory(_rec("dossier2", "Hook annotation fact", 0.9)),
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
    )
    context = json.loads(result)["hookSpecificOutput"]["additionalContext"]

    assert "_trust_:" in context, context
    assert "⟨" in context, context


# ---------------------------------------------------------------------------
# Reachability model
# ---------------------------------------------------------------------------


def test_daemon_config_can_only_ever_reach_balanced(monkeypatch) -> None:
    _production_daemon_env(monkeypatch)

    assert reachable_recall_formats(_DAEMON_BUDGET, _DAEMON_TOP_K) == {"balanced"}


def test_daemon_config_reports_both_plist_flags_as_inert(monkeypatch) -> None:
    _production_daemon_env(monkeypatch)

    assert inert_annotation_flags(_DAEMON_BUDGET, _DAEMON_TOP_K) == [
        "MEMO_RECALL_EPISTEMIC_LABELS",
        "MEMO_HIT_DOSSIER",
    ]


def test_nothing_is_inert_when_the_hook_budget_makes_compact_reachable(monkeypatch) -> None:
    _production_daemon_env(monkeypatch)

    assert reachable_recall_formats(_HOOK_BUDGET, _HOOK_TOP_K) == {"compact"}
    assert inert_annotation_flags(_HOOK_BUDGET, _HOOK_TOP_K) == []


def test_reachability_accounts_for_the_adaptive_budget_spread(monkeypatch) -> None:
    """A budget of 700 is `balanced` on its own, but adaptive scaling pushes a
    short prompt to 800*... and a long one to 420, so the set must be derived
    from the whole spread rather than the nominal number."""
    _production_daemon_env(monkeypatch)
    monkeypatch.setenv("MEMO_RECALL_ADAPTIVE_BUDGET", "0")
    assert reachable_recall_formats(220, _DAEMON_TOP_K) == {"compact"}

    # 220 -> 330 for a short prompt, which is no longer compact.
    monkeypatch.setenv("MEMO_RECALL_ADAPTIVE_BUDGET", "1")
    assert reachable_recall_formats(220, _DAEMON_TOP_K) == {"compact", "balanced"}
    assert inert_annotation_flags(220, _DAEMON_TOP_K) == []


# ---------------------------------------------------------------------------
# Startup warning is wired into the real daemon boot
# ---------------------------------------------------------------------------


def _boot_daemon(monkeypatch, tmp_path: Path) -> str:
    """Run the REAL recall_socket.run_server with the MLX/socket parts stubbed,
    and return what it wrote to stderr."""
    from memo import recall_socket

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(state_dir))

    stub_mem = SimpleNamespace(
        embedder=SimpleNamespace(embed=lambda texts: [[0.0] for _ in texts]),
        close=lambda: None,
    )
    monkeypatch.setattr("memo.memory.Memory", lambda cfg: stub_mem)

    class _FakeServer:
        def __init__(self, sock_path, cfg, mem) -> None:
            self._stats = None

        def server_close(self) -> None:
            pass

    monkeypatch.setattr(recall_socket, "_RecallServer", _FakeServer)
    monkeypatch.setattr(recall_socket, "_serve_until_shutdown", lambda *a, **k: None)
    monkeypatch.setattr(recall_socket, "_stats_persister", lambda *a, **k: None)
    monkeypatch.setattr(recall_socket.signal, "signal", lambda *a, **k: None)

    import io
    import sys as _sys

    buf = io.StringIO()
    monkeypatch.setattr(_sys, "stderr", buf)
    recall_socket.run_server(state_dir)
    return buf.getvalue()


def test_daemon_startup_names_the_flags_it_cannot_honour(monkeypatch, tmp_path) -> None:
    _production_daemon_env(monkeypatch)

    err = _boot_daemon(monkeypatch, tmp_path)

    assert "MEMO_RECALL_EPISTEMIC_LABELS" in err, err
    assert "MEMO_HIT_DOSSIER" in err, err
    assert "balanced" in err, err


def test_daemon_startup_stays_quiet_when_the_flags_are_readable(monkeypatch, tmp_path) -> None:
    """Negative control: the warning must be conditional, not an unconditional
    print that would pass the test above no matter what."""
    _production_daemon_env(monkeypatch)
    monkeypatch.setenv("MEMO_RECALL_TOKEN_BUDGET", str(_HOOK_BUDGET))
    monkeypatch.setenv("MEMO_RECALL_TOP_K", str(_HOOK_TOP_K))

    err = _boot_daemon(monkeypatch, tmp_path)

    assert "MEMO_HIT_DOSSIER" not in err, err
    assert "WARNING" not in err, err


def test_daemon_startup_stays_quiet_when_no_annotation_flag_is_set(monkeypatch, tmp_path) -> None:
    """A daemon on the default config with the flags OFF has nothing to report,
    even though `balanced` is still the only reachable format."""
    _production_daemon_env(monkeypatch)
    monkeypatch.setenv("MEMO_HIT_DOSSIER", "0")
    monkeypatch.setenv("MEMO_RECALL_EPISTEMIC_LABELS", "0")

    err = _boot_daemon(monkeypatch, tmp_path)

    assert "WARNING" not in err, err
