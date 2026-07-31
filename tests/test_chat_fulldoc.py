from memo.chat.fulldoc import dominant_doc_group, resolve_fulldoc


def _chunk(n: int, total: int, path: str = "notes/x.md") -> dict:
    return {
        "source": "memory",
        "id": f"m{n}",
        "type": "note",
        "score": 1.0,
        "title": f"Doc X (§{n}/{total})",
        "snippet": f"parte {n}",
        "path": path,
    }


def test_dominant_requires_share_and_chunks() -> None:
    group = dominant_doc_group(
        [
            _chunk(1, 2),
            _chunk(2, 2),
            {
                "source": "vault",
                "id": "v",
                "title": "otro",
                "score": 0.1,
                "snippet": "y",
                "path": "a.md",
            },
        ]
    )
    assert group is not None and len(group) == 2
    assert dominant_doc_group([_chunk(1, 2)]) is None  # <2 chunks
    others = [
        {
            "source": "vault",
            "id": f"v{i}",
            "title": f"t{i}",
            "score": 0.1,
            "snippet": "y",
            "path": f"{i}.md",
        }
        for i in range(4)
    ]
    assert dominant_doc_group([_chunk(1, 2), _chunk(2, 2), *others]) is None  # share 2/6 < 0.6


def test_resolve_memory_note_requires_all_chunks() -> None:
    class _Mem:
        def repo_get_file(self, repo, path, *, start=None, end=None):
            raise AssertionError("no debe llamarse para notas de memoria")

    doc = resolve_fulldoc(_Mem(), [_chunk(2, 2), _chunk(1, 2)])
    assert doc == {"title": "doc x", "text": "parte 1\n\nparte 2", "fulldoc_source": "memory"}
    assert resolve_fulldoc(_Mem(), [_chunk(1, 3), _chunk(2, 3)]) is None  # falta §3


def test_resolve_vault_uses_repo_get_file() -> None:
    class _Mem:
        def repo_get_file(self, repo, path, *, start=None, end=None):
            assert (repo, path) == ("vault", "docs/plan.md")
            return {"text": "contenido completo"}

    members = [
        {
            "source": "vault",
            "id": "v1",
            "title": "plan (§1/2)",
            "score": 1.0,
            "snippet": "a",
            "path": "docs/plan.md",
            "repo_name": "vault",
        },
        {
            "source": "vault",
            "id": "v2",
            "title": "plan (§2/2)",
            "score": 0.9,
            "snippet": "b",
            "path": "docs/plan.md",
            "repo_name": "vault",
        },
    ]
    doc = resolve_fulldoc(_Mem(), members)
    assert (
        doc is not None and doc["text"] == "contenido completo" and doc["fulldoc_source"] == "repo"
    )
