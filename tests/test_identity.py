from __future__ import annotations

import pytest

from memo.config import Config
from memo.errors import IdentityConflictError
from memo.identity import (
    GLOBAL_NAMESPACE,
    UNSCOPED_NAMESPACE,
    Identity,
    bucket_for_namespace,
    canonical_topic_key,
    current,
    identity_keys,
    namespace_for_index,
    namespace_for_write,
    normalized_content,
    normalized_content_hash,
    normalized_title,
)

_SESSION_ENV_VARS = ("MEMO_SESSION_ID", "CLAUDE_SESSION_ID", "CLAUDE_CODE_SESSION_ID")


def _clear_session_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _SESSION_ENV_VARS:
        monkeypatch.delenv(key, raising=False)


def test_machine_id_is_device_id_and_stable(tmp_cfg: Config) -> None:
    ident = current(tmp_cfg)
    assert isinstance(ident, Identity)
    assert ident.machine_id == tmp_cfg.device_id
    assert current(tmp_cfg).machine_id == ident.machine_id


def test_hostname_nonempty_and_label_includes_it(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_session_env(monkeypatch)
    ident = current(tmp_cfg)
    assert ident.hostname
    assert ident.hostname in ident.label
    assert ident.session_id is None
    assert ident.label == ident.hostname


def test_session_id_from_env_in_label(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_session_env(monkeypatch)
    monkeypatch.setenv("MEMO_SESSION_ID", "abcdef0123456789")
    ident = current(tmp_cfg)
    assert ident.session_id == "abcdef0123456789"
    assert ident.label == f"{ident.hostname}·abcdef01"


def test_as_dict_has_expected_keys(tmp_cfg: Config) -> None:
    assert set(current(tmp_cfg).as_dict()) == {
        "machine_id",
        "hostname",
        "session_id",
        "terminal",
        "label",
    }


def test_canonical_topic_and_title_use_nfkc_whitespace_and_casefold() -> None:
    assert canonical_topic_key("  \uff21lpha\t STRASSE  ") == "alpha strasse"
    assert normalized_title("  Café\n  PLAN  ") == "café plan"
    assert canonical_topic_key("  \n ") is None


def test_content_normalization_preserves_internal_whitespace() -> None:
    assert normalized_content("  hello  world \r\nnext\t \r\n") == "hello  world\nnext"
    assert normalized_content("a  b") != normalized_content("a b")
    digest = normalized_content_hash("hello")
    assert len(digest) == 64
    assert digest == normalized_content_hash(" hello\r\n")


@pytest.mark.parametrize(
    ("tags", "auto_project", "expected"),
    [
        (["project:Memo"], True, "project:memo"),
        (["project:memo", "project:MEMO"], True, "project:memo"),
        ([], False, GLOBAL_NAMESPACE),
        ([], True, UNSCOPED_NAMESPACE),
    ],
)
def test_namespace_for_write(tags: list[str], auto_project: bool, expected: str) -> None:
    assert namespace_for_write(tags, auto_project=auto_project) == expected


@pytest.mark.parametrize("tags", [["project:a", "project:b"], ["project:"]])
def test_namespace_for_write_rejects_ambiguous_tags(tags: list[str]) -> None:
    with pytest.raises(IdentityConflictError) as exc:
        namespace_for_write(tags, auto_project=True)
    assert exc.value.kind == "ambiguous_namespace"


def test_namespace_for_index_preserves_historical_ambiguity() -> None:
    assert namespace_for_index([], path="_global/2026/x.md") == GLOBAL_NAMESPACE
    assert namespace_for_index([], path="_unscoped/2026/x.md") == UNSCOPED_NAMESPACE
    assert namespace_for_index([], path="2026/x.md") == GLOBAL_NAMESPACE
    assert namespace_for_index(["project:Memo"], path="x.md") == "project:memo"
    assert namespace_for_index(["project:a", "project:b"], path="x.md") is None


def test_bucket_for_namespace_keeps_archive_names_safe() -> None:
    assert bucket_for_namespace("project:memo") == "memo"
    assert bucket_for_namespace("project:inactive") == "_inactive"
    assert bucket_for_namespace(GLOBAL_NAMESPACE) == GLOBAL_NAMESPACE
    assert bucket_for_namespace(UNSCOPED_NAMESPACE) == UNSCOPED_NAMESPACE


@pytest.mark.parametrize("namespace", ["memo", "project:"])
def test_bucket_for_namespace_rejects_invalid_namespaces(namespace: str) -> None:
    with pytest.raises(ValueError):
        bucket_for_namespace(namespace)


def test_identity_keys_are_composed_from_one_policy() -> None:
    keys = identity_keys(
        title=" Plan ",
        content="Body\r\n",
        tags=["project:Memo"],
        topic_key=" Release  Plan ",
        auto_project=True,
    )
    assert keys.namespace == "project:memo"
    assert keys.topic_key == "release plan"
    assert keys.normalized_title == "plan"
    assert keys.normalized_content_hash == normalized_content_hash("Body")


def test_identity_conflict_message_never_contains_body() -> None:
    exc = IdentityConflictError(
        kind="topic_ambiguous",
        incoming={"namespace": "project:memo"},
        conflicts=[{"id": "abcdef012345", "body": "do-not-render"}],
    )
    assert "do-not-render" not in str(exc)
    assert "abcdef01" in str(exc)
