from __future__ import annotations

from dataclasses import dataclass

from click.testing import CliRunner

from memo.cli import cli


def _env(tmp_cfg) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        "MEMO_VAULT_PATH": str(tmp_cfg.vault_path),
        "MEMO_AUTO_PROJECT_TAG": "0",
    }


@dataclass
class _Rec:
    id: str
    body: str


class _FakeCrossref:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.indexed: list[str] = []

    def reset(self) -> None:
        self.reset_calls += 1
        self.indexed.clear()

    def index_source(self, source_id: str, body: str) -> None:
        self.indexed.append(source_id)


class _FakeStore:
    def __init__(self, records: list[_Rec]) -> None:
        self._records = records

    def all_ids(self) -> list[str]:
        return [r.id for r in self._records]


class _FakeMemory:
    def __init__(self, records: list[_Rec]) -> None:
        self._records = records
        self._by_id = {r.id: r for r in records}
        self.crossref = _FakeCrossref()
        self.store = _FakeStore(records)

    def get(self, id_: str) -> _Rec | None:
        return self._by_id.get(id_)


def _corpus(n: int, *, bodyless: int = 0) -> list[_Rec]:
    return [
        _Rec(id=f"{i:032x}", body="" if i < bodyless else f"body {i} [[target]]") for i in range(n)
    ]


def test_links_reindex_covers_corpus_past_the_old_10k_cap(tmp_cfg, monkeypatch) -> None:
    records = _corpus(10_050)
    memory = _FakeMemory(records)
    monkeypatch.setattr("memo.cli_links._get_memory", lambda _cfg: memory)

    result = CliRunner().invoke(cli, ["links", "reindex", "--yes"], env=_env(tmp_cfg))

    assert result.exit_code == 0, result.output
    # reset() wipes every edge, so anything the rebuild misses loses its
    # crossrefs permanently.
    assert memory.crossref.reset_calls == 1
    assert memory.crossref.indexed == [r.id for r in records]
    assert "10050" in result.output
    assert "10000" not in result.output


def test_links_reindex_reports_bodyless_memories_separately(tmp_cfg, monkeypatch) -> None:
    records = _corpus(10, bodyless=3)
    memory = _FakeMemory(records)
    monkeypatch.setattr("memo.cli_links._get_memory", lambda _cfg: memory)

    result = CliRunner().invoke(cli, ["links", "reindex", "--yes"], env=_env(tmp_cfg))

    assert result.exit_code == 0, result.output
    assert memory.crossref.indexed == [r.id for r in records[3:]]
    assert "Reindexed 7 of 10 memories" in result.output
    assert "3" in result.output
