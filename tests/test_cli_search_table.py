"""`memo search`'s default (non-`--json`) output: the results table extracted
into `_render_search_table`, plus the `--explain` reason table. `--json`
output is covered elsewhere (`test_cli_parity.py`); this covers the rich-table
rendering path that a JSON-only test never exercises.
"""

from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from memo.cli_search import _render_search_table, search
from memo.config import Config
from memo.memory.record import MemoryRecord


def _hit(id_: str, title: str, score: float | None = 0.9) -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        path=f"2026/08/{id_}.md",
        title=title,
        type="note",
        tags=["alpha", "beta"],
        created="2026-08-01T00:00:00",
        updated="2026-08-01T00:00:00",
        body="body text",
        score=score,
    )


def _env(tmp_cfg: Config) -> dict[str, str]:
    return {
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        "MEMO_NONINTERACTIVE": "1",
    }


def _install_fake_mem(monkeypatch, hits: list[MemoryRecord], *, trace=None):
    fake = SimpleNamespace()

    def _search_with_trace(query, **kw):
        return {"hits": hits, "trace": trace or []}

    fake.search = lambda query, **kw: hits
    fake.search_with_trace = _search_with_trace
    monkeypatch.setattr("memo.cli_search._get_memory", lambda cfg: fake)
    monkeypatch.setattr("memo.cli_search.log_cli_consult", lambda *a, **k: None)
    return fake


def test_render_search_table_prints_score_type_title_tags(capsys):
    hits = [_hit("id1", "First hit"), _hit("id2", "Second hit", score=None)]
    hit_dicts = [h.to_dict() for h in hits]
    _render_search_table(hits, hit_dicts, explain=False, trace=None)
    out = capsys.readouterr().out
    assert "First hit" in out
    assert "Second hit" in out
    assert "0.900" in out
    # A None score renders the em-dash placeholder, not a crash.
    assert "—" in out


def test_render_search_table_explain_prints_reason_table_and_trace(capsys):
    hits = [_hit("id1", "First hit")]
    hit_dicts = [h.to_dict() for h in hits]
    hit_dicts[0]["explain"] = {"why": ["matched tag alpha", "recent"]}
    trace = [{"stage": "candidate_generation"}, {"stage": "rerank"}]
    _render_search_table(hits, hit_dicts, explain=True, trace=trace)
    out = capsys.readouterr().out
    assert "Why these ranked" in out
    assert "matched tag alpha" in out
    assert "search trace stages: candidate_generation, rerank" in out


def test_render_search_table_explain_skips_reason_table_when_no_hit_has_why(capsys):
    hits = [_hit("id1", "First hit")]
    hit_dicts = [h.to_dict() for h in hits]  # no "explain" key on any hit
    _render_search_table(hits, hit_dicts, explain=True, trace=[{"stage": "final"}])
    out = capsys.readouterr().out
    assert "Why these ranked" not in out
    assert "search trace stages: final" in out


def test_search_cli_prints_no_results_message_for_empty_hits(tmp_cfg: Config, monkeypatch):
    _install_fake_mem(monkeypatch, [])
    res = CliRunner().invoke(search, ["some query"], env=_env(tmp_cfg))
    assert res.exit_code == 0, res.output
    assert "no results" in res.output


def test_search_cli_renders_table_for_real_hits(tmp_cfg: Config, monkeypatch):
    _install_fake_mem(monkeypatch, [_hit("id1", "A memory about search")])
    res = CliRunner().invoke(search, ["some query"], env=_env(tmp_cfg))
    assert res.exit_code == 0, res.output
    assert "A memory about search" in res.output


def test_search_cli_explain_renders_reason_table(tmp_cfg: Config, monkeypatch):
    hit = _hit("id1", "A memory about search")
    _install_fake_mem(
        monkeypatch,
        [hit],
        trace=[{"stage": "candidate_generation"}, {"stage": "final"}],
    )
    res = CliRunner().invoke(search, ["some query", "--explain"], env=_env(tmp_cfg))
    assert res.exit_code == 0, res.output
    assert "A memory about search" in res.output
    assert "search trace stages:" in res.output
