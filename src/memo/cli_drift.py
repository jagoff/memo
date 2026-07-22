"""`memo drift` — flag code changes that violate your durable constraints.

The enforcement half of the constitution (``constitution.py`` is the advisory
half). Reads the same standing rules, runs ``git diff``, and reports added lines
that break a durable prohibition. ``--strict`` exits non-zero so it can gate a
pre-commit / CI hook; the default just warns.
"""

from __future__ import annotations

import subprocess

import click


def _git_diff(*, staged: bool, ref: str | None) -> str:
    args = ["git", "diff", "--unified=0", "--no-color"]
    if staged:
        args.append("--cached")
    if ref:
        args.append(ref)
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout


def _gather_rules() -> list[tuple[str, str]]:
    from memo.cli_common import get_memory
    from memo.config import Config
    from memo.constitution import gather_rules

    cfg = Config.from_env()
    return gather_rules(get_memory(cfg), cfg)


@click.command(name="drift")
@click.option("--staged", is_flag=True, help="Check staged changes (git diff --cached).")
@click.option("--ref", default=None, help="Diff against a ref/commit (e.g. origin/master).")
@click.option(
    "--strict",
    is_flag=True,
    help="Exit non-zero when violations are found (for a pre-commit / CI hook).",
)
def drift(*, staged: bool = False, ref: str | None = None, strict: bool = False) -> None:
    """Flag added code that violates a durable prohibition in memo."""
    from memo.drift_guard import added_lines_from_diff, parse_prohibitions, scan

    diff = _git_diff(staged=staged, ref=ref)
    if not diff.strip():
        click.echo("no changes to check")
        return
    prohibitions = parse_prohibitions(_gather_rules())
    if not prohibitions:
        click.echo("no enforceable prohibitions in memo (a rule needs `never … `pattern``)")
        return
    violations = scan(prohibitions, added_lines_from_diff(diff))
    if not violations:
        click.echo(f"clean — no drift vs {len(prohibitions)} constraint(s)")
        return
    for v in violations:
        click.echo(f"  {v.path}: `{v.pattern}` violates [{v.rule_id[:8]}] {v.rule_text}")
        click.echo(f"    + {v.line}")
    click.echo(f"\n{len(violations)} drift violation(s) vs your durable constraints")
    if strict:
        raise SystemExit(1)
