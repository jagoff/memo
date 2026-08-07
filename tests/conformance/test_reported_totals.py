"""A total is a claim about the corpus. A page size presented as a total makes
every derived metric (growth rate, panels) quietly wrong.

The brief asserted `payload["total"]` for both surfaces, but that key name
was never checked against the code. The real shapes:
  - `memo analytics summary --json` dumps `CorpusMetrics.__dict__`, whose
    corpus-count field is `total_memories`.
  - `memo stats --json` nests corpus figures under `payload["corpus"]["total"]`.
Both already read from `store.count()` (a real `SELECT COUNT(*) FROM meta`),
not from the length of a capped page -- this test pins that down so a
regression back to `len(list(...limit=N))` fails loudly instead of silently.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from memo import web_build
from memo.cli import cli

pytestmark = pytest.mark.conformance


def _env(cfg) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),
        "MEMO_STATE_DIR": str(cfg.state_dir),
        "MEMO_EMBEDDER_DIMS": "64",
    }


def test_analytics_summary_reports_the_real_total(big_corpus, corpus_size) -> None:
    result = CliRunner().invoke(cli, ["analytics", "summary", "--json"], env=_env(big_corpus))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["total_memories"] == corpus_size


def test_stats_reports_the_real_total(big_corpus, corpus_size) -> None:
    result = CliRunner().invoke(cli, ["stats", "--json"], env=_env(big_corpus))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["corpus"]["total"] == corpus_size


def test_dashboard_corpus_pillar_reports_the_real_total(big_corpus, corpus_size) -> None:
    # include_projection=False skips the vector-read + PCA step (expensive,
    # irrelevant to the total) -- the same cheap path the live-refresh
    # endpoint uses. Call collect_data() directly: it's the single function
    # that gathers everything `memo dashboard` renders, so this covers the
    # real data path without standing up an HTTP server.
    data = web_build.collect_data(big_corpus, include_projection=False)
    corpus_pillar = next((p for p in data["pillars"] if p["label"] == "Corpus"), None)
    assert corpus_pillar is not None, data["pillars"]
    assert corpus_pillar["summary"] == f"{corpus_size:,} memories"
