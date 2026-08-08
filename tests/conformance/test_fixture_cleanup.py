"""big_corpus monkeypatches process env before seeding. If seeding raises
before the fixture reaches its `yield`, the generator never resumes -- so
`mp.undo()` must run on the exception path too, or MEMO_EMBEDDER_DIMS /
MEMO_DATA_DIR / etc. leak into every test that runs later in the session."""

from __future__ import annotations

import os

import pytest

from memo.store import VecStore

from . import conftest as big_corpus_conftest

pytestmark = pytest.mark.conformance

_PATCHED_ENV_KEYS = (
    "MEMO_EMBEDDER_DIMS",
    "MEMO_NONINTERACTIVE",
    "MEMO_DATA_DIR",
    "MEMO_STATE_DIR",
    "MEMO_AUTO_PROJECT_TAG",
    "MEMO_STORE_BY_PROJECT",
    # Set only for the duration of the seeding loop (see conftest.big_corpus),
    # so its revert rides a nested MonkeyPatch rather than the outer one.
    "MEMO_TANTIVY_ENABLED",
)


def test_big_corpus_reverts_env_when_seeding_fails(tmp_path_factory, monkeypatch) -> None:
    before = {key: os.environ.get(key) for key in _PATCHED_ENV_KEYS}

    def _boom(*_args, **_kwargs) -> None:
        raise RuntimeError("simulated seeding failure")

    monkeypatch.setattr(VecStore, "upsert", _boom)

    gen = big_corpus_conftest.big_corpus.__wrapped__(tmp_path_factory, corpus_size=3)
    with pytest.raises(RuntimeError, match="simulated seeding failure"):
        next(gen)

    after = {key: os.environ.get(key) for key in _PATCHED_ENV_KEYS}
    assert after == before
