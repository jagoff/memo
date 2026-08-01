"""CLI readiness gate for the definitive Memo runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from memo.config import Config


def _json(value: Any) -> None:
    click.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


@click.group(name="definitive")
def definitive_group() -> None:
    """Verify and benchmark Memo's independent-memory guarantees."""


@definitive_group.command(name="check")
def definitive_check_cmd() -> None:
    from memo.definitive import definitive_check
    from memo.memory import Memory

    memory = Memory(Config.from_env())
    try:
        report = definitive_check(memory)
    finally:
        memory.close()
    _json(report)
    if not report["ok"]:
        raise click.ClickException("definitive readiness gate failed")


@definitive_group.command(name="benchmark")
@click.option("--events", default=250, type=click.IntRange(10, 10_000))
@click.option("--minimum-eps", default=25.0, type=click.FloatRange(min=0.01))
def definitive_benchmark_cmd(events: int, minimum_eps: float) -> None:
    from memo.definitive import run_journal_benchmark

    report = run_journal_benchmark(
        events=events,
        min_events_per_second=minimum_eps,
    )
    _json(report)
    if not report["ok"]:
        raise click.ClickException("definitive benchmark gate failed")


@definitive_group.command(name="integration")
@click.option(
    "--receipt",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Write the signed hermetic two-peer integration receipt here.",
)
def definitive_integration_cmd(receipt: Path) -> None:
    """Run message, sync, terminal, ACK, presence, and restart proof."""

    from memo.definitive_integration_runtime import run_hermetic_integration_proof

    report = run_hermetic_integration_proof(receipt_path=receipt)
    _json(report)
    if not report["ok"]:
        raise click.ClickException("definitive integration gate failed")


@definitive_group.command(name="verify-integration")
@click.option(
    "--receipt",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--roster",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option("--trusted-roster-hash", required=True)
@click.option(
    "--evidence-archive",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
def definitive_verify_integration_cmd(
    receipt: Path,
    roster: Path,
    trusted_roster_hash: str,
    evidence_archive: Path,
) -> None:
    """Verify a receipt against an externally pinned roster and evidence."""

    import hashlib

    from memo.definitive_integration import (
        read_integration_receipt,
        read_trusted_roster_snapshot,
        verify_integration_receipt,
    )
    from memo.errors import OperationalError, OperationalErrorCode

    signed = read_integration_receipt(receipt)
    trusted = read_trusted_roster_snapshot(
        roster,
        trusted_roster_hash=trusted_roster_hash,
    )
    verify_integration_receipt(
        signed,
        roster=trusted,
        trusted_roster_hash=trusted_roster_hash,
    )
    archive_sha256 = hashlib.sha256(evidence_archive.read_bytes()).hexdigest()
    if signed.evidence.get("evidence_archive_sha256") != archive_sha256:
        raise OperationalError(
            OperationalErrorCode.INVALID_EVENT,
            "definitive integration evidence archive digest mismatch",
            retryable=False,
        )
    _json(
        {
            "ok": True,
            "receipt_sha256": signed.receipt_sha256,
            "trusted_roster_hash": trusted_roster_hash,
            "evidence_archive_sha256": archive_sha256,
        }
    )


__all__ = ["definitive_group"]
