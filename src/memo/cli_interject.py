"""`memo interject` + `memo ask-gaps` — review the Phase-3 proactive shadow logs.

Interject/ask-gaps are report-only (not auto-graduatable): they SHADOW-COUNT what
they would surface regardless of the enable flag. These commands are how a human
reviews that evidence before flipping MEMO_INTERJECT_ENABLED / MEMO_ASK_GAPS_ENABLED.

NOTE: the group is named ``ask-gaps`` (not ``ask``) — ``memo ask`` is the
pre-existing stable-core question-answering command (cli_search.py); reusing
that name here would silently shadow it.
"""

from __future__ import annotations

from collections import Counter

import click

from memo.config import Config


@click.group(name="interject")
def interject_group() -> None:
    """Inspect / silence the proactive interject banner."""


@interject_group.command(name="shadow")
def interject_shadow() -> None:
    """What interject WOULD (or did) fire on — the flip-decision evidence."""
    from memo.interject import read_shadow

    cfg = Config.from_env()
    rows = read_shadow(cfg.state_dir)
    if not rows:
        click.echo("no interject shadow activity logged")
        return
    rendered = sum(1 for r in rows if r.get("rendered"))
    ids: Counter = Counter()
    for r in rows:
        for i in r.get("ids", []):
            ids[i] += 1
    click.echo(
        f"{len(rows)} would-interject events ({rendered} rendered, {len(rows) - rendered} shadow-only)"
    )
    for i, c in ids.most_common(10):
        click.echo(f"  {c:>4}  {i}")


@interject_group.command(name="silence")
def interject_silence() -> None:
    """Silence interject for the CURRENT session (one-key silence)."""
    from memo.identity import current
    from memo.interject import silence

    cfg = Config.from_env()
    sid = current(cfg).session_id or "_no_session"
    silence(cfg.state_dir, sid)
    click.echo(f"interject silenced for session {sid}")


@click.group(name="ask-gaps")
def ask_group() -> None:
    """Inspect the proactive ask-gaps shadow log."""


@ask_group.command(name="shadow")
def ask_shadow() -> None:
    """What ask WOULD (or did) surface as a briefing question."""
    from memo.ask_gaps import read_shadow

    cfg = Config.from_env()
    rows = read_shadow(cfg.state_dir)
    if not rows:
        click.echo("no ask-gaps shadow activity logged")
        return
    rendered = sum(1 for r in rows if r.get("rendered"))
    click.echo(
        f"{len(rows)} would-ask events ({rendered} rendered, {len(rows) - rendered} shadow-only)"
    )
    top: Counter = Counter()
    for r in rows:
        top[str(r.get("prompt") or "")[:60]] += 1
    for p, c in top.most_common(10):
        click.echo(f"  {c:>4}  {p}")
