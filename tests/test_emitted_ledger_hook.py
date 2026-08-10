import builtins
import json
import re
from dataclasses import dataclass
from pathlib import Path

from click.testing import CliRunner

from memo import recall_logic as rl


@dataclass
class _Hit:
    id: str
    title: str
    body: str
    score: float
    tags: tuple[str, ...] = ()


def _hits():
    return [
        _Hit("aaaaaaaa", "First", "x" * 900, 0.80),
        _Hit("bbbbbbbb", "Second", "short body", 0.70),
    ]


def test_sink_records_what_the_full_renderer_emitted():
    sink: list[tuple[str, str]] = []
    out = rl.render_by_format(
        "full", _hits(), [], turn=1, body_chars=400, token_budget=0, emitted_sink=sink
    )
    assert [i for i, _ in sink] == ["aaaaaaaa", "bbbbbbbb"]
    recorded = dict(sink)
    # truncated to the cap, not the stored 900 chars
    assert len(recorded["aaaaaaaa"]) <= 420
    assert recorded["aaaaaaaa"] in out or recorded["aaaaaaaa"].rstrip("…") in out
    assert recorded["bbbbbbbb"] == "short body"


def test_sink_records_empty_body_for_the_compact_renderer():
    sink: list[tuple[str, str]] = []
    rl.render_by_format(
        "compact", _hits(), [], turn=1, body_chars=400, token_budget=0, emitted_sink=sink
    )
    assert [i for i, _ in sink] == ["aaaaaaaa", "bbbbbbbb"]
    assert all(body == "" for _, body in sink)


def test_sink_omits_hits_whose_body_was_dropped_by_the_char_budget():
    """token_budget=20 (max_chars=80) is too small even for the bare title
    prefix, so both hits are dropped before either ever reaches the sink.
    Assert the actually-vacuous case explicitly, rather than looping over an
    empty list and asserting nothing."""
    sink: list[tuple[str, str]] = []
    rl.render_by_format(
        "full", _hits(), [], turn=1, body_chars=400, token_budget=20, emitted_sink=sink
    )
    assert sink == []


def _pin_prefix_length_flags(monkeypatch) -> None:
    """MEMO_HIT_DOSSIER and MEMO_RECALL_EPISTEMIC_LABELS both lengthen a
    hit's prefix (a `_trust_: ...` line and a `⟨label⟩` tag respectively),
    which shifts exactly how many characters the budget-trimmed path has
    left for `available` -- so the trimmed-body length these two tests pin
    to an exact character count is only deterministic once these flags are
    pinned too. Pinned ON, matching the review's measured values (271 /
    401+0): left ambient, `tests/conftest.py` pops both to hermetic-off for
    ordinary pytest runs, while a dev machine with the trust program
    activated (`~/.claude/settings.json` env) exports them ON -- either
    ambient default would otherwise make this test's exact-length
    assertions pass or fail depending on which machine runs it."""
    monkeypatch.setenv("MEMO_HIT_DOSSIER", "1")
    monkeypatch.setenv("MEMO_RECALL_EPISTEMIC_LABELS", "1")


def test_sink_records_the_trimmed_body_not_the_stored_body(monkeypatch):
    """render_recall_context's budget-trimmed exit must record what actually
    got committed to `lines` (the `trimmed_body` local), not the pre-trim
    `body` -- recording the untrimmed 900-char body here would let a later
    call digest content the model was never shown. At token_budget=150
    (max_chars=600) hit 1 doesn't fit the fast path and goes through the
    trim branch; hit 2 is dropped entirely and never reaches the sink."""
    _pin_prefix_length_flags(monkeypatch)
    sink: list[tuple[str, str]] = []
    out = rl.render_by_format(
        "full", _hits(), [], turn=1, body_chars=400, token_budget=150, emitted_sink=sink
    )
    recorded = dict(sink)
    assert list(recorded) == ["aaaaaaaa"]
    assert len(recorded["aaaaaaaa"]) == 271
    assert recorded["aaaaaaaa"] in out


def test_sink_records_full_body_alongside_a_title_only_empty_body(monkeypatch):
    """At token_budget=200 (max_chars=800) hit 1 fits the fast path in full
    (401 chars -- same length as the unconstrained case) and hit 2 falls
    into the budget-trimmed path's title-only branch (available <= 20),
    recording an empty body rather than being dropped outright. Covers the
    `elif ...: emitted_sink.append((hit.id, ""))` branch, previously
    unexercised by any test."""
    _pin_prefix_length_flags(monkeypatch)
    sink: list[tuple[str, str]] = []
    out = rl.render_by_format(
        "full", _hits(), [], turn=1, body_chars=400, token_budget=200, emitted_sink=sink
    )
    recorded = dict(sink)
    assert list(recorded) == ["aaaaaaaa", "bbbbbbbb"]
    assert len(recorded["aaaaaaaa"]) == 401
    assert recorded["aaaaaaaa"] in out
    assert recorded["bbbbbbbb"] == ""


def test_balanced_sink_records_bullet_slices_when_untruncated():
    """fmt="balanced" is the production default for an ordinary hook turn
    (MEMO_RECALL_TOKEN_BUDGET defaults to 600, resolve_recall_format picks
    "balanced" for budgets in 300-800 with fewer than 5 hits) and had no
    test at all before this. At token_budget=0 (no char cap, so the final
    joined-string truncation never fires) the sink must record each hit's
    bullet_text -- the body-derived slice actually rendered -- not the raw
    hit.body."""
    sink: list[tuple[str, str]] = []
    hits = _hits()
    out = rl.render_by_format(
        "balanced", hits, [], turn=1, body_chars=400, token_budget=0, emitted_sink=sink
    )
    recorded = dict(sink)
    assert list(recorded) == ["aaaaaaaa", "bbbbbbbb"]
    assert recorded["aaaaaaaa"] in out
    assert recorded["bbbbbbbb"] in out
    # The mutant that records hit.body instead of bullet_text[i]: hit 1's
    # bullet is a 50-char truncated slice of its 900-char body. If the sink
    # ever equals the raw body, it is recording the wrong thing.
    assert recorded["aaaaaaaa"] != hits[0].body
    assert len(recorded["aaaaaaaa"]) < len(hits[0].body)


def test_balanced_sink_is_empty_when_the_final_truncation_fires():
    """The renderer builds every hit's line first and only truncates the
    JOINED string as a last step, so it cannot tell which per-hit lines
    survived a small budget's slice. Its whole safety argument lives in the
    `elif emitted_sink is not None` -- record nothing at all when that final
    truncation fired, rather than guess which hits' bullets are still intact.
    token_budget=5 (max_chars=20) is far below the ~266-char untruncated
    output for these hits, so it reliably fires the truncation branch."""
    sink: list[tuple[str, str]] = []
    rl.render_by_format(
        "balanced", _hits(), [], turn=1, body_chars=400, token_budget=5, emitted_sink=sink
    )
    assert sink == []


def test_sink_is_optional_and_default_none_changes_nothing():
    with_sink = rl.render_by_format(
        "full", _hits(), [], turn=1, body_chars=400, token_budget=0, emitted_sink=[]
    )
    without = rl.render_by_format("full", _hits(), [], turn=1, body_chars=400, token_budget=0)
    assert with_sink == without


# ---------------------------------------------------------------------------
# The hook's own writer (cli_recall_hook.recall_hook). Everything above this
# line exercises the renderers' sink in isolation; nothing above touches
# `recall_hook` or `emitted_ledger.append` at all, so a `recall_hook` run
# that never wrote to the ledger would still leave every test above green.
# These tests go through the real CLI command via CliRunner, seeded with one
# real memory. The cold-start-downgrades-to-bm25 path (no `.prewarm_ts` in a
# fresh state_dir) means these need no real MLX forward pass -- only a
# stubbed `MLXEmbedder.embed` for the one-time index write at seed time, so
# they stay fast.
# ---------------------------------------------------------------------------


def _seeded_hook_env(monkeypatch, tmp_path: Path, session_id: str) -> dict[str, str]:
    """One real memory in an isolated data/state dir, plus the env a
    `recall-hook` CliRunner invocation needs to actually find and render it.

    MEMO_CONFIG_FILE must point somewhere nonexistent: without it,
    `Config.from_env()` inside the CLI picks up whatever real global
    markdown config this machine has, which can name a different embedder
    model than the plain `Config(...)` object used to seed below -- a
    spurious "index built with X but current config is Y" search failure
    that silently bails the hook (empty hits, nothing to test).
    """
    import memo.embedder as embedder_mod
    from memo.config import Config
    from memo.memory import Memory

    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    cfg = Config(data_dir=data_dir, state_dir=state_dir, embedder_dims=4, reranker_enabled=False)

    monkeypatch.setattr(
        embedder_mod.MLXEmbedder,
        "embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    mem = Memory(cfg)
    mem.save(
        content="Chat pipeline streaming design notes with feedback loop details galore",
        title="Chat design notes",
    )
    mem.close()

    return {
        "MEMO_DATA_DIR": str(data_dir),
        "MEMO_STATE_DIR": str(state_dir),
        "MEMO_CONFIG_FILE": str(tmp_path / "nonexistent-config.toml"),
        "MEMO_EMBEDDER_MODEL": cfg.embedder_model,
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_RECALL_MIN_SIM": "0.0",
        "MEMO_RECALL_MIN_BODY_CHARS": "0",
        "MEMO_SESSION_ID": session_id,
        "MEMO_EMITTED_LEDGER": "1",
    }


def _invoke(payload: dict[str, object], env: dict[str, str]) -> object:
    from memo.cli import cli

    return CliRunner().invoke(cli, ["recall-hook"], input=json.dumps(payload), env=env)


def test_flag_on_hook_run_writes_a_hook_sourced_entry_under_the_env_session_id(
    monkeypatch, tmp_path
):
    """F5.1: the whole point of this task. Nothing above this line in the
    file actually calls `recall_hook` or `emitted_ledger.append` -- this is
    the first test that does.

    The payload's `session_id` ("payload-sess") is deliberately DIFFERENT
    from MEMO_SESSION_ID ("envsess-1"), so this also pins the `_ledger_sid`
    fix (task-6 review, kept as an improvement): the ledger entry must land
    under the env-resolved id, not the payload's. Reverting `_ledger_sid`
    back to reusing the payload-derived `_sid` keeps the rest of the suite
    green but fails this test -- the entry would land under "payload-sess"
    instead."""
    env = _seeded_hook_env(monkeypatch, tmp_path, "envsess-1")
    result = _invoke(
        {"prompt": "chat pipeline streaming design", "session_id": "payload-sess"}, env
    )
    assert result.exit_code == 0

    from memo import emitted_ledger as el

    state_dir = Path(env["MEMO_STATE_DIR"])
    entries = el.read(state_dir, "envsess-1")
    assert entries, "expected the hook to write at least one ledger entry under MEMO_SESSION_ID"
    assert all(e.src == "hook" for e in entries.values())
    assert all(e.ref.startswith("memo-h/") for e in entries.values())
    # Entry.for_text (task-6 review deviation, kept) always populates hp -- a
    # constructor call that dropped it would still pass every other
    # assertion here.
    assert all(e.hp is not None for e in entries.values())
    assert el.read(state_dir, "payload-sess") == {}


def test_flag_off_hook_run_creates_no_ledger_file_at_all(monkeypatch, tmp_path):
    """F5.2."""
    env = _seeded_hook_env(monkeypatch, tmp_path, "hooksess-2")
    env["MEMO_EMITTED_LEDGER"] = "0"
    result = _invoke({"prompt": "chat pipeline streaming design", "session_id": "hooksess-2"}, env)
    assert result.exit_code == 0
    assert not (Path(env["MEMO_STATE_DIR"]) / "emitted").exists()


def test_ledger_write_failure_never_breaks_the_hook_or_changes_its_output(monkeypatch, tmp_path):
    """F5.3: fail-open, the property that protects the 5s budget. A control
    run and a run with `emitted_ledger.append` patched to raise must both
    exit 0 with the same rendered content (memory ids differ across the two
    separately-seeded fixtures, so they're normalized out before comparing)."""
    env_a = _seeded_hook_env(monkeypatch, tmp_path / "a", "hooksess-3a")
    control = _invoke(
        {"prompt": "chat pipeline streaming design", "session_id": "hooksess-3a"}, env_a
    )
    assert control.exit_code == 0

    def _boom(*_a: object, **_kw: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("memo.emitted_ledger.append", _boom)

    env_b = _seeded_hook_env(monkeypatch, tmp_path / "b", "hooksess-3b")
    patched = _invoke(
        {"prompt": "chat pipeline streaming design", "session_id": "hooksess-3b"}, env_b
    )
    assert patched.exit_code == 0

    def _norm(s: str) -> str:
        return re.sub(r"\[[0-9a-f]{8}\]", "[normalized]", s)

    control_ctx = json.loads(control.output)["hookSpecificOutput"]["additionalContext"]
    patched_ctx = json.loads(patched.output)["hookSpecificOutput"]["additionalContext"]
    assert _norm(control_ctx) == _norm(patched_ctx)


def test_print_failure_leaves_the_ledger_empty(monkeypatch, tmp_path):
    """F1: the hook installs its own SIGALRM wall-clock cap (`_arm_deadline`;
    p95 9.5s against a 10s cap over 1500 live fires per its own docstring),
    so an exit landing between a ledger write and the payload print is the
    measured norm, not an edge case. Simulated here by making the payload
    print itself raise -- with the fix (the ledger-write block moved below
    the print, mirroring `mark_ids_recalled`'s existing placement) the write
    is never reached and the ledger stays empty. Before that fix this test
    fails: the ledger already held the entry by the time print raised.

    MEMO_RECALL_HOOK_BUDGET_MS=0 disables `_arm_deadline`'s real SIGALRM
    entirely (see its own `if budget_s <= 0: return`). Without this, forcing
    an exception here skips `_close_memory()`/`_disarm_deadline()` same as it
    would in production -- but in a pytest *process*, a still-armed itimer
    outlives this test and fires up to 10s later inside a LATER, unrelated
    test, re-entering this test's now-stale `_bail` closure and calling
    `sys.exit(0)` from inside it. That is exactly the cross-test pollution
    this test must not cause -- disabling the deadline here tests the
    print/ledger ordering this test is actually about without depending on
    the (real, useful, separately-tested) deadline mechanism at all."""
    env = _seeded_hook_env(monkeypatch, tmp_path, "f1-sess")
    env["MEMO_RECALL_HOOK_BUDGET_MS"] = "0"

    orig_print = builtins.print

    def _boom_on_payload(*args: object, **kwargs: object) -> None:
        if args and isinstance(args[0], str) and '"hookEventName"' in args[0]:
            raise RuntimeError("simulated exit at print")
        orig_print(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "print", _boom_on_payload)

    result = _invoke({"prompt": "chat pipeline streaming design", "session_id": "f1-sess"}, env)
    assert result.exception is not None

    from memo import emitted_ledger as el

    assert el.read(Path(env["MEMO_STATE_DIR"]), "f1-sess") == {}
