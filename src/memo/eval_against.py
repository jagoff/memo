"""Same-corpus, two-revision recall comparison.

The saved-baseline gate compares (current code, current corpus) against
(old code, old corpus). The two deltas are confounded, so it cannot approve a
ranking change. This module evaluates the SAME live corpus twice — once with
the working tree's code, once with the code at a git ref — so the corpus term
cancels and the remaining delta is attributable to the diff.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Runner = Callable[[list[str], dict[str, str], Path], str]

_MAIN_ENTRYPOINT_PATH = "src/memo/__main__.py"


class AgainstError(RuntimeError):
    """A controlled failure in the --against comparison (bad ref, no git,
    a polluted/non-JSON subprocess payload, ...).

    Kept module-local (not a `click.ClickException`) so this module stays a
    leaf: pure and independently unit-testable, with no dependency on click.
    `cli_eval.py` catches this at the CLI boundary and re-raises it as a
    `click.ClickException`, so the user sees a clean one-line failure instead
    of a raw traceback — see m3/MEDIUM-2.
    """


@dataclass
class AgainstResult:
    passed: bool
    message: str
    config: str
    current_precision: float
    ref_precision: float
    current_noise: float
    ref_noise: float
    current_avoid: float = 0.0
    ref_avoid: float = 0.0
    current_leak: float = 0.0
    ref_leak: float = 0.0


@dataclass
class AgainstRun:
    """A parsed `memo eval recall --json` run from the ref side.

    Carries `labels_fingerprint` alongside `rows` because `compare_rows` needs
    it as an identity guard: a relative `--labels` path resolves against the
    ref worktree's cwd, not the caller's, so without this the two sides can
    silently score against different label sets.
    """

    rows: list[dict[str, Any]]
    labels_fingerprint: str


def build_eval_argv(
    *,
    labels_path: str | None,
    k: int,
    profile: str | None,
    configs: tuple[str, ...],
) -> list[str]:
    """The `memo eval recall` argv both sides of the comparison run.

    ``--no-cache`` is not optional: the result cache is keyed on
    corpus+labels+configs+k and lives in ``state_dir``, which every worktree on
    this machine shares. Cached, the ref run would read the current run's
    numbers and every comparison would tie.

    `labels_path` must already be absolute by the time it reaches this
    function — it is forwarded verbatim into a subprocess whose cwd is the
    ref's worktree, not the caller's, so a relative path would resolve
    against the wrong directory.
    """
    argv = ["eval", "recall", "--k", str(k), "--json", "--no-cache"]
    if labels_path:
        argv += ["--labels", labels_path]
    if profile:
        argv += ["--profile", profile]
    for name in configs:
        argv += ["--config", name]
    return argv


def _by_config(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r.get("config") or ""): r for r in rows}


def _label_fingerprint_guard(current_fp: str, ref_fp: str) -> str | None:
    """A FAIL message if the label-set identity guard trips, else None.

    Extracted purely to keep `compare_rows` under its own complexity budget;
    see `compare_rows`'s docstring for the guard's rationale (fails closed on
    a missing ref fingerprint, fails on a mismatch, skipped entirely when the
    caller has no current-side fingerprint to give at all).
    """
    if current_fp and not ref_fp:
        return (
            "FAIL — the ref run did not report a labels_fingerprint (older "
            "code, or an unrecognized payload shape); refusing to compare "
            "without confirming both sides scored the same label set"
        )
    if current_fp and ref_fp and current_fp != ref_fp:
        return (
            "FAIL — label sets differ between the two runs "
            f"({current_fp!r} vs {ref_fp!r}); both sides must evaluate the "
            "SAME label set or the comparison means nothing (check for a "
            "relative --labels path, or a diff that touched the default "
            "label set)"
        )
    return None


def _diff_parts(
    *,
    cp: float,
    rp: float,
    cn: float,
    rn: float,
    ca: float,
    ra: float,
    cl: float,
    rl: float,
    tol: float,
) -> list[str]:
    """The ways the current run failed to hold the ref's numbers — empty if
    none. Extracted purely to keep `compare_rows` under its complexity
    budget."""
    parts: list[str] = []
    if cp < rp - tol:
        parts.append(f"precision@k {cp:.3f} < ref {rp:.3f}")
    if cn > rn + tol:
        parts.append(f"noise@k {cn:.3f} > ref {rn:.3f}")
    if ca < ra - tol:
        parts.append(f"avoid@k {ca:.3f} < ref {ra:.3f}")
    if cl > rl + tol:
        parts.append(f"avoid_leak@k {cl:.3f} > ref {rl:.3f}")
    return parts


def compare_rows(
    current: list[dict[str, Any]],
    ref: list[dict[str, Any]],
    *,
    tol: float = 0.01,
    current_labels_fingerprint: str = "",
    ref_labels_fingerprint: str = "",
) -> AgainstResult:
    """Compare the two runs on the config the REF run ranked best.

    Pinning to the ref's best config stops a different config winning the
    current run from masking a regression in the one that was shipping.

    `current_labels_fingerprint`/`ref_labels_fingerprint` are an identity
    guard, run first — mirroring `check_gate`'s label-fingerprint check.
    Mismatched label sets make any precision/noise delta meaningless, so a
    mismatch fails loudly instead of comparing anyway. The check only engages
    when `current_labels_fingerprint` is given — callers with nothing to
    give at all (including the existing pure-unit tests) are unaffected. But
    once the caller HAS a current-side fingerprint (the CLI always does — see
    `labels.fingerprint()`), a MISSING ref fingerprint fails closed rather
    than silently skipping the guard: an absent ref fingerprint means "we
    don't know if it's the same label set", which is exactly the case this
    guard exists to catch, not a reason to trust it.

    Also enforces the negative-recall ⛔ floors alongside precision/noise:
    `avoid_at_k` (coverage) must not drop, `avoid_leak_at_k` (leakage) must
    not rise. `--update-baseline` once shipped bare `gate_metrics` that
    dropped these silently to "vacuously true forever" — don't reintroduce
    that gap here. A missing ref value for either uses `check_gate`'s own
    non-enforcing default (0.0 coverage floor, 1.0 leak ceiling — both
    permissive), not `or 0.0` for both: the latter would make a missing ref
    leak rate maximally STRICT instead, which is the opposite convention.
    """
    guard_message = _label_fingerprint_guard(current_labels_fingerprint, ref_labels_fingerprint)
    if guard_message:
        return AgainstResult(False, guard_message, "", 0.0, 0.0, 0.0, 0.0)

    if not ref:
        return AgainstResult(False, "FAIL — the ref run produced no rows", "", 0.0, 0.0, 0.0, 0.0)
    if not current:
        return AgainstResult(
            False, "FAIL — the current run produced no rows", "", 0.0, 0.0, 0.0, 0.0
        )

    ref_best = max(ref, key=lambda r: float(r.get("precision_at_k") or 0.0))
    config = str(ref_best.get("config") or "")
    cur = _by_config(current).get(config)
    if cur is None:
        ran = ", ".join(sorted(_by_config(current)))
        return AgainstResult(
            False,
            f"FAIL — config {config!r} was not evaluated in the current run (ran: {ran})",
            config,
            0.0,
            float(ref_best.get("precision_at_k") or 0.0),
            0.0,
            float(ref_best.get("noise_at_k") or 0.0),
            0.0,
            float(ref_best.get("avoid_at_k", 0.0)),
            0.0,
            float(ref_best.get("avoid_leak_at_k", 1.0)),
        )

    cp = float(cur.get("precision_at_k") or 0.0)
    rp = float(ref_best.get("precision_at_k") or 0.0)
    cn = float(cur.get("noise_at_k") or 0.0)
    rn = float(ref_best.get("noise_at_k") or 0.0)
    ca = float(cur.get("avoid_at_k") or 0.0)
    # `.get(key, default)`, not `.get(key) or default`: mirrors check_gate's
    # non-enforcing defaults for a MISSING ref value (0.0 coverage floor, 1.0
    # leak ceiling — see the docstring). `or` would misfire if the ref value
    # were legitimately 0.0 (a real, present avoid_leak_at_k of 0.0 is not
    # "missing", but `0.0 or 1.0` evaluates to 1.0 regardless).
    ra = float(ref_best.get("avoid_at_k", 0.0))
    cl = float(cur.get("avoid_leak_at_k") or 0.0)
    rl = float(ref_best.get("avoid_leak_at_k", 1.0))

    parts = _diff_parts(cp=cp, rp=rp, cn=cn, rn=rn, ca=ca, ra=ra, cl=cl, rl=rl, tol=tol)
    if parts:
        message = f"FAIL [config {config!r}] — " + "; ".join(parts)
        return AgainstResult(False, message, config, cp, rp, cn, rn, ca, ra, cl, rl)

    message = (
        f"PASS [config {config!r}] — prec@k {cp:.3f} vs ref {rp:.3f}, "
        f"noise@k {cn:.3f} vs ref {rn:.3f} (same corpus, both runs uncached)"
    )
    return AgainstResult(True, message, config, cp, rp, cn, rn, ca, ra, cl, rl)


def _run_git(args: list[str], *, cwd: Path) -> str:
    """Run a git command, surfacing stderr in the raised error.

    Plain `check=True` collapses every failure into `returned non-zero exit
    status N`, discarding the actual message (`fatal: invalid reference`,
    `fatal: ... already exists`, ...) — undebuggable when the whole point of
    this module is to run cleanly or fail with a reason.
    """
    try:
        proc = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise AgainstError(f"git {' '.join(args)} failed: {stderr or exc}") from exc
    return proc.stdout.strip()


def resolve_repo_root(start: Path) -> Path:
    """The repo root, resolved via git rather than by counting `__file__`
    parents.

    Counting parents assumes `cli_eval.py` sits at a fixed depth under a
    source checkout; it returns nonsense under the installed uv tool, whose
    site-packages layout doesn't match. `git rev-parse --show-toplevel` works
    from any worktree — main or linked — of whichever repo `start` is inside,
    and fails loudly (via `_run_git`) when there is none.
    """
    return Path(_run_git(["rev-parse", "--show-toplevel"], cwd=start))


def _ref_has_main_entrypoint(ref: str, repo_root: Path) -> bool:
    """Whether `src/memo/__main__.py` exists at `ref`, checked with `git
    cat-file -e` — cheap (no checkout) way to pre-flight before paying for a
    full `git worktree add`.

    A missing path is an ordinary negative result (an old ref), not a
    failure — handled with a plain exit-code check rather than `_run_git`,
    which treats every nonzero exit as exceptional.
    """
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{ref}:{_MAIN_ENTRYPOINT_PATH}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _first_commit_with_main_entrypoint(repo_root: Path) -> str:
    """The earliest commit in this checkout's history that added
    `src/memo/__main__.py` — named in the error `run_against` raises for a
    ref that predates it, so the user has something concrete to check out or
    rebase onto instead of just "doesn't work"."""
    out = _run_git(
        ["log", "--diff-filter=A", "--format=%H", "--follow", "--", _MAIN_ENTRYPOINT_PATH],
        cwd=repo_root,
    )
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return lines[-1] if lines else "unknown"


def _worktree_dest() -> Path:
    """A scratch directory for the temporary comparison worktree.

    Deliberately NOT `repo_root / ".git" / ...`: in a linked worktree — which
    is how this very plan is executed — `.git` is a FILE pointing at the
    shared common dir, not a directory, so nesting a path under it fails with
    `Not a directory`. `tempfile.mkdtemp` sidesteps the question entirely.
    """
    return Path(tempfile.mkdtemp(prefix="memo-eval-against-"))


def _add_worktree(ref: str, repo_root: Path, dest: Path) -> Path:
    _run_git(["worktree", "add", "--detach", "--force", str(dest), ref], cwd=repo_root)
    return dest


def _remove_worktree(repo_root: Path, dest: Path) -> None:
    # Best-effort: this runs in a `finally`, so raising here would replace a
    # real failure from the runner (or from `_add_worktree`, now that it runs
    # inside the same try) with a cleanup failure instead of masking it —
    # still surface the reason (m3) rather than swallowing it outright.
    try:
        _run_git(["worktree", "remove", "--force", str(dest)], cwd=repo_root)
    except AgainstError as exc:
        print(f"warning: {exc}", file=sys.stderr)
        # `git worktree remove` only works on a path git actually registered
        # as a worktree. When `_add_worktree` itself failed (a bad ref), it
        # never got that far — `dest` is just the empty directory
        # `_worktree_dest`'s `tempfile.mkdtemp()` created, which git has no
        # record of and will therefore never clean up. Remove it directly so
        # a typo'd ref doesn't leak a scratch dir per invocation.
        shutil.rmtree(dest, ignore_errors=True)


def _default_runner(argv: list[str], env: dict[str, str], cwd: Path) -> str:
    try:
        proc = subprocess.run(argv, cwd=cwd, env=env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise AgainstError(f"{' '.join(argv)} failed: {stderr or exc}") from exc
    return proc.stdout


def run_against(
    ref: str,
    *,
    repo_root: Path,
    argv: list[str],
    runner: Runner | None = None,
) -> AgainstRun:
    """Evaluate `argv` with the code at `ref`, against the live corpus.

    The ref's code is reached through ``PYTHONPATH=<worktree>/src python -m
    memo``. Invoking the installed `memo` script instead would run the globally
    installed build in both halves of the comparison — which is also why this
    pre-flights that `ref` actually HAS that entrypoint: any ref older than
    the commit that added it would otherwise fail deep inside the subprocess
    with an opaque `-m memo` import error.
    """
    import os

    if not _ref_has_main_entrypoint(ref, repo_root):
        first_sha = _first_commit_with_main_entrypoint(repo_root)
        raise AgainstError(
            f"ref {ref!r} predates {_MAIN_ENTRYPOINT_PATH} (first added in "
            f"{first_sha}), so `PYTHONPATH=<worktree>/src python -m memo` has "
            "no entrypoint to run there. Compare against a ref at or after "
            "that commit."
        )

    run = runner or _default_runner
    dest = _worktree_dest()
    try:
        _add_worktree(ref, repo_root, dest)
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        wt_src = str(dest / "src")
        env["PYTHONPATH"] = f"{wt_src}{os.pathsep}{existing}" if existing else wt_src
        raw = run([sys.executable, "-m", "memo", *argv], env, dest)
    finally:
        _remove_worktree(repo_root, dest)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AgainstError(
            f"ref run at {ref!r} did not produce valid JSON output — the eval "
            f"subprocess likely printed a warning to stdout ahead of its "
            f"--json payload. Parse error: {exc}. First 200 chars of stdout: "
            f"{raw[:200]!r}"
        ) from exc
    if isinstance(payload, dict):
        rows = list(payload.get("rows") or [])
        labels_fingerprint = str(payload.get("labels_fingerprint") or "")
    else:
        rows = list(payload or [])
        labels_fingerprint = ""
    return AgainstRun(rows=rows, labels_fingerprint=labels_fingerprint)
