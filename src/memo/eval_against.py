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
) -> AgainstResult:
    """Compare the two runs on the config the REF run ranked best.

    Pinning to the ref's best config stops a different config winning the
    current run from masking a regression in the one that was shipping.
    """
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

    parts: list[str] = []
    if cp < rp - tol:
        parts.append(f"precision@k {cp:.3f} < ref {rp:.3f}")
    if cn > rn + tol:
        parts.append(f"noise@k {cn:.3f} > ref {rn:.3f}")

    if parts:
        message = f"FAIL [config {config!r}] — " + "; ".join(parts)
        return AgainstResult(False, message, config, cp, rp, cn, rn)

    message = (
        f"PASS [config {config!r}] — prec@k {cp:.3f} vs ref {rp:.3f}, "
        f"noise@k {cn:.3f} vs ref {rn:.3f} (same corpus, both runs uncached)"
    )
    return AgainstResult(True, message, config, cp, rp, cn, rn)


def _worktree_dest(repo_root: Path) -> Path:
    return repo_root / ".git" / "memo-eval-against"


def _add_worktree(ref: str, repo_root: Path, dest: Path) -> Path:
    subprocess.run(
        ["git", "worktree", "add", "--detach", "--force", str(dest), ref],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return dest


def _remove_worktree(repo_root: Path, dest: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(dest)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


def _default_runner(argv: list[str], env: dict[str, str], cwd: Path) -> str:
    proc = subprocess.run(argv, cwd=cwd, env=env, check=True, capture_output=True, text=True)
    return proc.stdout


def run_against(
    ref: str,
    *,
    repo_root: Path,
    argv: list[str],
    runner: Runner | None = None,
) -> list[dict[str, Any]]:
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
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    return list(rows or [])
