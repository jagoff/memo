"""Per-folder vault abstracts dream pass (K4)."""

from __future__ import annotations

from memo import dream_folder_abstracts
from memo.dream_folder_abstracts import (
    _folder_of,
    collect_folders,
    members_hash,
    run_folder_abstracts,
)


def test_folder_of_strips_chunk_suffix_and_roots():
    assert _folder_of("vault/Recetas/pan.md") == "vault/Recetas"
    assert _folder_of("vault/Recetas/pan.md#chunk-3") == "vault/Recetas"
    assert _folder_of("nota.md") == ""


def test_members_hash_order_independent_and_folder_sensitive():
    assert members_hash("f", ["a", "b"]) == members_hash("f", ["b", "a"])
    assert members_hash("f", ["a"]) != members_hash("g", ["a"])


def test_collect_folders_groups_reference_only():
    class _Store:
        def list_recent(self, limit):
            return [
                {"id": "r1", "type": "reference", "path": "V/Recetas/a.md", "title": "A"},
                {"id": "r2", "type": "reference", "path": "V/Recetas/b.md", "title": "B"},
                {
                    "id": "r3",
                    "type": "reference",
                    "path": "V/Recetas/b.md#chunk-1",
                    "title": "B §1",
                },
                {"id": "d1", "type": "decision", "path": "x.md", "title": "D"},
            ]

    class _Mem:
        store = _Store()

    groups = collect_folders(_Mem(), min_members=2)
    assert len(groups) == 1
    assert groups[0]["folder"] == "V/Recetas"
    assert groups[0]["ids"] == ["r1", "r2"]  # chunk + durable rows excluded


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MEMO_DREAM_FOLDER_ABSTRACTS_ENABLED", raising=False)
    assert run_folder_abstracts(None, None)["status"] == "disabled"


def _seed_reference(mem, n=5):
    # Reference content must clear the >=60-char reference-noise gate
    # (write_ops.is_reference_noise) or save() refuses to index it.
    return [
        mem.save(
            content=f"Receta de pan numero {i} con harina, agua, sal y levadura fresca.",
            title=f"pan {i}",
            type_="reference",
        ).id
        for i in range(n)
    ]


def test_run_saves_then_skips_unchanged(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_FOLDER_ABSTRACTS_ENABLED", "1")
    monkeypatch.setattr(
        dream_folder_abstracts,
        "_llm_abstract_folder",
        lambda mem, g: {"title": "Recetas", "body": "Recetario de pan y masas."},
    )
    _seed_reference(mock_memory)

    res1 = run_folder_abstracts(None, mock_memory, min_members=2)
    assert res1["status"] == "done"
    assert any(a["status"] == "saved" for a in res1["abstracts"])

    res2 = run_folder_abstracts(None, mock_memory, min_members=2)
    assert all(a["status"] == "skip_unchanged" for a in res2["abstracts"])


def test_run_updates_in_place_on_membership_change(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_FOLDER_ABSTRACTS_ENABLED", "1")
    monkeypatch.setattr(
        dream_folder_abstracts,
        "_llm_abstract_folder",
        lambda mem, g: {"title": "Recetas", "body": f"Recetario ({len(g['ids'])} docs)."},
    )
    _seed_reference(mock_memory)
    run_folder_abstracts(None, mock_memory, min_members=2)

    mock_memory.save(
        content="Receta nueva de pan de campo con masa madre y harina integral.",
        title="pan nuevo",
        type_="reference",
    )
    res = run_folder_abstracts(None, mock_memory, min_members=2)
    assert any(a["status"] == "updated" for a in res["abstracts"])
    row = mock_memory.store._conn.execute(
        "SELECT COUNT(*) AS n FROM meta WHERE type='synthesis' "
        "AND json_extract(extra_json, '$.synthesis_kind') = 'folder_abstract'"
    ).fetchone()
    assert row["n"] == 1  # updated in place — exactly ONE abstract per folder


def test_dry_run_saves_nothing(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_DREAM_FOLDER_ABSTRACTS_ENABLED", "1")
    _seed_reference(mock_memory)
    res = run_folder_abstracts(None, mock_memory, min_members=2, dry_run=True)
    assert all(a["status"] == "would_save" for a in res["abstracts"])
    row = mock_memory.store._conn.execute(
        "SELECT COUNT(*) AS n FROM meta WHERE type='synthesis'"
    ).fetchone()
    assert row["n"] == 0


def test_cli_folder_abstracts_disabled_smoke(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from memo.cli_dream import dream_cmd

    monkeypatch.setenv("MEMO_NONINTERACTIVE", "1")
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("MEMO_DREAM_FOLDER_ABSTRACTS_ENABLED", raising=False)
    res = CliRunner().invoke(dream_cmd, ["folder-abstracts", "--json"])
    assert res.exit_code == 0, res.output
    assert '"disabled"' in res.output
