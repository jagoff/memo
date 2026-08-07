"""Same-corpus, two-revision recall comparison.

The saved-baseline gate compares (current code, current corpus) against
(old code, old corpus). The two deltas are confounded, so it cannot approve a
ranking change. This module evaluates the SAME live corpus twice — once with
the working tree's code, once with the code at a git ref — so the corpus term
cancels and the remaining delta is attributable to the diff.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Runner = Callable[[list[str], dict[str, str], Path], str]


@dataclass
class AgainstResult:
    passed: bool
    message: str
    config: str
    current_precision: float
    ref_precision: float
    current_noise: float
    ref_noise: float


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
    mismatch fails loudly instead of comparing anyway. The check is skipped
    when either fingerprint is empty (unknown), so callers with nothing to
    give — including the existing pure-unit tests — are unaffected.

    Also enforces the negative-recall ⛔ floors alongside precision/noise:
    `avoid_at_k` (coverage) must not drop, `avoid_leak_at_k` (leakage) must
    not rise. `--update-baseline` once shipped bare `gate_metrics` that
    dropped these silently to "vacuously true forever" — don't reintroduce
    that gap here.
    """
    if (
        current_labels_fingerprint
        and ref_labels_fingerprint
        and current_labels_fingerprint != ref_labels_fingerprint
    ):
        message = (
            "FAIL — label sets differ between the two runs "
            f"({current_labels_fingerprint!r} vs {ref_labels_fingerprint!r}); both "
            "sides must evaluate the SAME label set or the comparison means "
            "nothing (check for a relative --labels path, or a diff that "
            "touched the default label set)"
        )
        return AgainstResult(False, message, "", 0.0, 0.0, 0.0, 0.0)

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
        )

    cp = float(cur.get("precision_at_k") or 0.0)
    rp = float(ref_best.get("precision_at_k") or 0.0)
    cn = float(cur.get("noise_at_k") or 0.0)
    rn = float(ref_best.get("noise_at_k") or 0.0)
    ca = float(cur.get("avoid_at_k") or 0.0)
    ra = float(ref_best.get("avoid_at_k") or 0.0)
    cl = float(cur.get("avoid_leak_at_k") or 0.0)
    rl = float(ref_best.get("avoid_leak_at_k") or 0.0)

    parts: list[str] = []
    if cp < rp - tol:
        parts.append(f"precision@k {cp:.3f} < ref {rp:.3f}")
    if cn > rn + tol:
        parts.append(f"noise@k {cn:.3f} > ref {rn:.3f}")
    if ca < ra - tol:
        parts.append(f"avoid@k {ca:.3f} < ref {ra:.3f}")
    if cl > rl + tol:
        parts.append(f"avoid_leak@k {cl:.3f} > ref {rl:.3f}")

    if parts:
        message = f"FAIL [config {config!r}] — " + "; ".join(parts)
        return AgainstResult(False, message, config, cp, rp, cn, rn)

    message = (
        f"PASS [config {config!r}] — prec@k {cp:.3f} vs ref {rp:.3f}, "
        f"noise@k {cn:.3f} vs ref {rn:.3f} (same corpus, both runs uncached)"
    )
    return AgainstResult(True, message, config, cp, rp, cn, rn)


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
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr or exc}") from exc
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


def _worktree_dest(repo_root: Path) -> Path:
    """A scratch directory for the temporary comparison worktree.

    Deliberately NOT `repo_root / ".git" / ...`: in a linked worktree — which
    is how this very plan is executed — `.git` is a FILE pointing at the
    shared common dir, not a directory, so nesting a path under it fails with
    `Not a directory`. `tempfile.mkdtemp` sidesteps the question entirely and
    behaves the same whether `repo_root` is the main checkout or a linked
    worktree.
    """
    return Path(tempfile.mkdtemp(prefix="memo-eval-against-"))


def _add_worktree(ref: str, repo_root: Path, dest: Path) -> Path:
    _run_git(["worktree", "add", "--detach", "--force", str(dest), ref], cwd=repo_root)
    return dest


def _remove_worktree(repo_root: Path, dest: Path) -> None:
    # Best-effort: this runs in a `finally`, so raising here would replace a
    # real failure from the runner with a cleanup failure instead of masking
    # it — still surface the reason (m3) rather than swallowing it outright.
    try:
        _run_git(["worktree", "remove", "--force", str(dest)], cwd=repo_root)
    except RuntimeError as exc:
        print(f"warning: {exc}", file=sys.stderr)


def _default_runner(argv: list[str], env: dict[str, str], cwd: Path) -> str:
    try:
        proc = subprocess.run(argv, cwd=cwd, env=env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"{' '.join(argv)} failed: {stderr or exc}") from exc
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
    installed build in both halves of the comparison.
    """
    import os

    run = runner or _default_runner
    dest = _worktree_dest(repo_root)
    _add_worktree(ref, repo_root, dest)
    try:
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        wt_src = str(dest / "src")
        env["PYTHONPATH"] = f"{wt_src}{os.pathsep}{existing}" if existing else wt_src
        raw = run([sys.executable, "-m", "memo", *argv], env, dest)
    finally:
        _remove_worktree(repo_root, dest)
    payload = json.loads(raw)
    if isinstance(payload, dict):
        rows = list(payload.get("rows") or [])
        labels_fingerprint = str(payload.get("labels_fingerprint") or "")
    else:
        rows = list(payload or [])
        labels_fingerprint = ""
    return AgainstRun(rows=rows, labels_fingerprint=labels_fingerprint)
