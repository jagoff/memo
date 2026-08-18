
def test_latest_remote_tag_refuses_unsafe_repo_urls() -> None:
    """MEMO_AUTO_UPDATE_REPO flows into `git`, so restrict the transport."""
    from memo.runtime.autoupdate import _is_safe_repo_url

    assert _is_safe_repo_url("https://github.com/jagoff/memo.git") is True
    assert _is_safe_repo_url("ext::sh -c 'touch /tmp/pwn'") is False
    assert _is_safe_repo_url("file:///tmp/evil") is False
    assert _is_safe_repo_url("--upload-pack=touch /tmp/pwn") is False
    assert _is_safe_repo_url("") is False
