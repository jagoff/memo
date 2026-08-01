from pathlib import Path

from memo.chat.contacts_alias import build_index, resolve_jid

_MAMA_JID = "74191281901822@lid"


def _write(dir_path: Path, name: str, body: str) -> None:
    (dir_path / name).write_text(body, encoding="utf-8")


def test_build_index_includes_stem_apodo_fullname_and_kinship_triggers(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "Mama.md",
        "\n".join(
            [
                "- **wa_jid**: " + _MAMA_JID,
                "- **Apodo**: Ma",
                "- **Apellido / nombre completo**: Monica Ferrari",
                "- **Relación**: Mamá",
            ]
        ),
    )

    index = build_index(tmp_path)

    assert index["mama"] == _MAMA_JID  # filename stem
    assert index["ma"] == _MAMA_JID  # Apodo
    assert index["monica"] == _MAMA_JID  # full-name token
    assert index["ferrari"] == _MAMA_JID  # full-name token
    assert index["madre"] == _MAMA_JID  # kinship trigger
    assert index["mami"] == _MAMA_JID  # kinship trigger


def test_ambiguous_trigger_shared_by_two_jids_is_absent(tmp_path: Path) -> None:
    _write(tmp_path, "Vale.md", "- **wa_jid**: jid-one@lid\n- **Apodo**: Vale\n")
    _write(tmp_path, "Valeria.md", "- **wa_jid**: jid-two@lid\n- **Apodo**: Vale\n")

    index = build_index(tmp_path)

    assert "vale" not in index
    # unambiguous stems from each note still survive
    assert index["valeria"] == "jid-two@lid"


def test_resolve_jid_matches_kinship_trigger_in_free_text_query(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "Mama.md",
        "\n".join(
            [
                "- **wa_jid**: " + _MAMA_JID,
                "- **Relación**: Mamá",
            ]
        ),
    )
    index = build_index(tmp_path)

    assert resolve_jid("qué me dijo mi mamá?", index) == _MAMA_JID


def test_resolve_jid_returns_none_when_query_has_no_contact(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "Mama.md",
        "- **wa_jid**: " + _MAMA_JID + "\n- **Relación**: Mamá\n",
    )
    index = build_index(tmp_path)

    assert resolve_jid("cuál es el clima hoy?", index) is None


def test_note_without_wa_jid_is_ignored(tmp_path: Path) -> None:
    _write(tmp_path, "SinJid.md", "- **Apodo**: Fantasma\n")

    index = build_index(tmp_path)

    assert "fantasma" not in index
    assert "sinjid" not in index


def test_jid_label_is_used_as_fallback_when_wa_jid_absent(tmp_path: Path) -> None:
    _write(tmp_path, "Amigo.md", "- **jid**: fallback-jid@lid\n")

    index = build_index(tmp_path)

    assert index["amigo"] == "fallback-jid@lid"


def test_missing_contacts_dir_returns_empty_index(tmp_path: Path) -> None:
    assert build_index(tmp_path / "does-not-exist") == {}


def test_dot_and_underscore_prefixed_notes_are_excluded(tmp_path: Path) -> None:
    _write(tmp_path, ".Hidden.md", "- **wa_jid**: hidden-jid@lid\n")
    _write(tmp_path, "_Template.md", "- **wa_jid**: template-jid@lid\n")

    index = build_index(tmp_path)

    assert "hidden" not in index
    assert "template" not in index
