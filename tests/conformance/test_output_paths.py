"""Every `-o/--out` surface: a clean error or a written file. Never a traceback.

/tmp is a symlink on macOS, which is why the symlink case is not exotic.

Two prior defects motivate this gate:
  - `atomic_write_text` rejected any destination whose parent is a symlink
    (fixed by follow-the-symlink commit `94f5faa6`) -- `memo graph mindmap`
    and `memo federation export` share that primitive.
  - `memo backup --out` and `memo export <fmt>` raw-tracebacked on a missing
    parent directory (fixed by the destination pre-check commit `3a30e660`).

A Click *usage* error (wrong flag, missing required option) also exits
non-zero, and it is easy to mistake for "the command failed cleanly" if you
only check `result.exception`. Click gives `UsageError` its own exit code
(2), distinct from the exit code (1) a plain `ClickException` raises -- a
non-zero exit_code == 2 means the harness called the command wrong, not that
the command's own destination-handling logic ever ran. That is the signal
this module uses to keep the vacuous-pass trap out of these assertions.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.graph import GraphStore

from .conftest import seeded_id

pytestmark = pytest.mark.conformance

_CREATED = "2026-01-01T00:00:00+00:00"
_ENTITY_A = "conformance-mindmap-alpha"
_ENTITY_B = "conformance-mindmap-beta"


def _env(cfg) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),
        "MEMO_STATE_DIR": str(cfg.state_dir),
        "MEMO_EMBEDDER_DIMS": "64",
        # None of these surfaces search or rerank, but Config.from_env()
        # defaults reranker_enabled=True on Apple Silicon unless told
        # otherwise -- keep the lane MLX-free regardless of that default.
        "MEMO_RERANKER_ENABLED": "false",
    }


def _key_file(tmp_path: Path) -> Path:
    """A federation signing key valid enough that `_read_key` accepts it, so
    the only thing left that can fail in the export is the destination path."""
    path = tmp_path / "federation.key"
    path.write_bytes(b"conformance-signing-key-material")
    path.chmod(0o600)
    return path


@pytest.fixture(scope="module")
def graph_seeded(big_corpus) -> None:
    """`graph mindmap` renders `mem.navigator.export_json()`, which is built
    from the entity graph -- populated by `Memory.save()`'s extraction pass, a
    path `big_corpus` never runs (it upserts straight into `VecStore`). Left
    empty, the graph is empty and `graph mindmap` prints "Graph is empty" and
    returns *before* ever reaching the output-path logic under test, which
    would make that row of the table pass vacuously. `record_extraction` is
    the same storage primitive the extraction pass writes through, and it
    needs no MLX.
    """
    store = GraphStore(big_corpus.graph_db)
    try:
        store.record_extraction(
            memory_id=seeded_id(0),
            memory_date=_CREATED,
            entities=[
                {"name": _ENTITY_A, "type": "concept"},
                {"name": _ENTITY_B, "type": "concept"},
            ],
            extracted_at=_CREATED,
        )
    finally:
        store.close()


def _mindmap_argv(dest: Path, tmp_path: Path) -> list[str]:
    return ["graph", "mindmap", _ENTITY_A, "--no-open", "-o", str(dest)]


def _federation_export_argv(dest: Path, tmp_path: Path) -> list[str]:
    return [
        "federation",
        "export",
        str(dest),
        "--principal",
        "conformance-principal",
        "--key-file",
        str(_key_file(tmp_path)),
    ]


def _backup_argv(dest: Path, tmp_path: Path) -> list[str]:
    return ["backup", "--out", str(dest)]


def _export_json_argv(dest: Path, tmp_path: Path) -> list[str]:
    return ["export", "json", str(dest)]


# label, argv-builder. Each builder takes (dest, tmp_path) and returns full argv
# built from each command's REAL signature -- verified by reading the source,
# not guessed from a flag-name pattern:
#   memo graph mindmap [ENTITY] [--depth N] [--node-cap N] [-o/--out PATH] [--open/--no-open]
#   memo federation export OUTPUT_PATH --principal P [--owner O] --key-file PATH
#   memo backup [--out PATH]                      (option on the group itself)
#   memo export json OUTPUT_PATH                   (positional, no flag)
OUTPUT_SURFACES: list[tuple[str, Callable[[Path, Path], list[str]]]] = [
    ("graph mindmap", _mindmap_argv),
    ("federation export", _federation_export_argv),
    ("backup", _backup_argv),
    ("export json", _export_json_argv),
]


def _assert_clean_and_on_topic(result, label: str) -> None:
    """A raw exception (anything Click's own `main()` did not convert into a
    `SystemExit` via `ClickException.show()`) is a traceback bug -- fail on
    it. A `UsageError` exit (code 2) means argv was wrong, not that the
    command's own destination handling ran -- fail on that too, rather than
    silently accepting it as "the surface passed."
    """
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"{label} raised {result.exception!r} instead of a clean error:\n{result.output}"
    )
    assert result.exit_code != 2, (
        f"{label} exited with a Click usage error (wrong flag/argument), which "
        f"never reached the destination-path handling this test checks:\n{result.output}"
    )


@pytest.mark.parametrize("label,argv_for", OUTPUT_SURFACES)
def test_symlinked_parent_is_accepted(big_corpus, graph_seeded, tmp_path, label, argv_for) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    dest = link / "out.dat"

    result = CliRunner().invoke(cli, argv_for(dest, tmp_path), env=_env(big_corpus))

    _assert_clean_and_on_topic(result, label)
    if result.exit_code == 0:
        # The write must actually land -- through the symlink and into the
        # real directory it points at -- not merely "not crash".
        assert (real / "out.dat").is_file(), (
            f"{label} exited 0 but never wrote through the symlinked parent:\n{result.output}"
        )


@pytest.mark.parametrize("label,argv_for", OUTPUT_SURFACES)
def test_missing_parent_gives_a_clean_error(
    big_corpus, graph_seeded, tmp_path, label, argv_for
) -> None:
    dest = tmp_path / "does" / "not" / "exist" / "out.dat"

    result = CliRunner().invoke(cli, argv_for(dest, tmp_path), env=_env(big_corpus))

    _assert_clean_and_on_topic(result, label)
    if result.exit_code == 0:
        # Some surfaces self-heal (mkdir -p the tree); if they claim success
        # the file must actually be there.
        assert dest.is_file(), (
            f"{label} exited 0 without writing the destination it claimed to "
            f"succeed on:\n{result.output}"
        )
    else:
        assert "Traceback" not in result.output
