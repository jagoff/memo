"""The test session's Markdown config dir must start empty and stay per-run.

`tests/conftest.py` points `MEMO_CONFIG_DIR` away from the developer's real
`~/.config/memo` so `Config.from_env()` cannot silently inherit it. That
redirect used to target a FIXED path, `$TMPDIR/memo-test-nonexistent-config-dir`,
and "nonexistent" was an assumption rather than an invariant.

Observed 2026-08-09: some test wrote a Markdown config there without pinning
its own `MEMO_CONFIG_DIR`, and from then on every pytest run on that machine
read the leftovers as real configuration. A stray `models-config.md` holding
`embedder_dims = 1024` / `model_profile = "balanced"` was enough to make
`_embedder_was_pinned()` true, which stops `Config.from_env()` from adopting an
index's embedder profile — 27 tests failed, reproducibly and permanently, until
the directory was deleted by hand. CI never reproduced it: a fresh runner has an
empty TMPDIR, so it looked like one machine being flaky.

These tests pin both halves of the invariant.
"""

from __future__ import annotations

import os
from pathlib import Path

from memo import config_md


def test_config_dir_is_not_the_developers_real_one() -> None:
    real = (Path.home() / ".config" / "memo").resolve()

    assert config_md.config_home() != real


def test_config_dir_starts_empty() -> None:
    """No leftovers from an earlier run may be visible as configuration."""
    assert config_md.field_values() == {}


def test_config_dir_is_unique_to_this_run() -> None:
    """A fixed path lets one polluted run poison every later one."""
    configured = os.environ["MEMO_CONFIG_DIR"]

    assert "memo-test-nonexistent-config-dir" not in configured, (
        "MEMO_CONFIG_DIR is back on the shared fixed path; a stray write there "
        "survives the process and silently configures every later pytest run"
    )
