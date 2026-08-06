"""Regression: the daemon-unreachable fallback notice is advice, not an event.

`memo episodes index` printed the same four-line warning — "daemon unreachable
… falling back to in-process … to start the daemon: …" — once per batch. A
single command emitted it five times before its first line of real output; a
long backfill emits it hundreds of times. The advice is identical every time
and only actionable once.

It now warns once per process (per socket path), and the fallback itself is
unchanged: the work still succeeds in-process.
"""

from __future__ import annotations

import logging

import pytest

from memo import embedder_client


@pytest.fixture(autouse=True)
def _forget_previous_notices():
    embedder_client._reset_fallback_notices()
    yield
    embedder_client._reset_fallback_notices()


@pytest.fixture
def unreachable_daemon(tmp_path, monkeypatch):
    """No socket, no strict flags: the in-process fallback path."""

    class _Stub:
        def embed(self, texts):
            return [[0.0, 0.0, 0.0, 1.0] for _ in texts]

        def embed_query(self, text):
            return [0.0, 0.0, 0.0, 1.0]

    monkeypatch.setattr(embedder_client, "_inproc", lambda: _Stub())
    monkeypatch.setattr(embedder_client, "_try_socket", lambda *a, **k: None)
    monkeypatch.setattr(embedder_client, "_require_daemon", lambda: False)
    return tmp_path


def test_batch_embeds_warn_once_per_process(unreachable_daemon, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="memo.embedder_client"):
        for _ in range(5):
            embedder_client.embed(["uno"], state_dir=unreachable_daemon)

    notices = [r for r in caplog.records if "daemon unreachable" in r.getMessage()]
    assert len(notices) == 1, f"emitted {len(notices)} identical fallback notices"


def test_the_fallback_still_returns_vectors(unreachable_daemon) -> None:
    vectors = embedder_client.embed(["uno", "dos"], state_dir=unreachable_daemon)

    assert len(vectors) == 2
    assert all(len(v) == 4 for v in vectors)


def test_query_and_batch_share_the_notice(unreachable_daemon, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="memo.embedder_client"):
        embedder_client.embed(["uno"], state_dir=unreachable_daemon)
        embedder_client.embed_query("uno", state_dir=unreachable_daemon)

    notices = [r for r in caplog.records if "daemon unreachable" in r.getMessage()]
    assert len(notices) == 1, "the same advice was repeated for the query path"


def test_a_different_socket_still_warns(unreachable_daemon, tmp_path, caplog) -> None:
    """A second state dir is a genuinely different situation worth reporting."""
    other = tmp_path / "other-state"
    other.mkdir()

    with caplog.at_level(logging.WARNING, logger="memo.embedder_client"):
        embedder_client.embed(["uno"], state_dir=unreachable_daemon)
        embedder_client.embed(["uno"], state_dir=other)

    notices = [r for r in caplog.records if "daemon unreachable" in r.getMessage()]
    assert len(notices) == 2
