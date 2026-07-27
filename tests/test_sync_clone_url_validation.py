"""Regression: `memo sync clone`/`setup`/`bootstrap` must not reach git's
arbitrary-command remote helpers (`ext::`/`fd::`) via a caller-supplied URL —
the same RCE class already closed for `memo_repo_index`. See
`_validate_clone_url` in sync_git.py."""

from __future__ import annotations

import pytest

from memo.sync_git import _GIT_SAFE_ENV, SyncGitError, _validate_clone_url


@pytest.mark.parametrize(
    "url",
    [
        "ext::sh -c 'touch /tmp/pwned'",
        "ext::git-upload-pack",
        "fd::17/foo",
        "-oProxyCommand=evil",  # option injection
        "--upload-pack=evil",
        "",
        "   ",
        "sneaky::transport",
    ],
)
def test_rejects_dangerous_urls(url: str) -> None:
    with pytest.raises(SyncGitError):
        _validate_clone_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/jagoff/memo-sync.git",
        "http://example.com/x.git",
        "ssh://git@github.com/jagoff/memo-sync.git",
        "git://example.com/x.git",
        "git+ssh://git@host/x.git",
        "file:///Users/fer/repos/memo-sync",
        "git@github.com:jagoff/memo-sync.git",  # scp-like
        "https://[::1]/x.git",  # IPv6 literal must not trip the `::` guard
    ],
)
def test_accepts_legitimate_urls(url: str) -> None:
    _validate_clone_url(url)  # must not raise


def test_git_safe_env_disables_transport_helpers() -> None:
    allowed = _GIT_SAFE_ENV["GIT_ALLOW_PROTOCOL"].split(":")
    assert "ext" not in allowed and "fd" not in allowed
    assert "https" in allowed and "file" in allowed
    assert _GIT_SAFE_ENV["GIT_TERMINAL_PROMPT"] == "0"
