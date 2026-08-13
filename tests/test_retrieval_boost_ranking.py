"""Ranking invariants for the curatorial retrieval boost.

`boost_for` multiplies a score that upstream stages already bounded to 1.0
(`_rerank` fuses `alpha * P(yes) + (1 - alpha) * rrf_bonus`, both in [0, 1] —
`rerank_ops.py:445`), and it runs at stage 14 of 22 (`search_ops.py:345`), six
stages after that bound. A multiplier reaching 12x sitting downstream of a term
bounded at 1.0 does not adjust the ranking, it replaces it. These tests pin the
two properties that stop it:

1. **Bounded displacement.** `_MAX_BOOST` is exactly the score ratio at which a
   candidate becomes immune to metadata alone. At 12x that was vacuous.
2. **No double-counting.** memo derives a record's title from its own body
   (`record.py:653`) and its filename from that title (`write_ops.py:1704`), so
   for a self-titled record "the filename matches" and "the title matches" are
   one signal counted twice — and that signal is *entity mention*, not *answer
   relevance*.

The control shape is the live 2026-08-05 regression, query "por que se deprecó
synapse" (`docs/SPECS/2026-08-04-memo-audit-design.md`).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from memo.memory.record import MemoryRecord, _slugify
from memo.memory.search_scoring_ops import _SearchScoringMixin
from memo.retrieval_boost import _MAX_BOOST, boost_for

_QUERY = "por que se deprecó synapse"


class _Harness(_SearchScoringMixin):
    pass


def _memo_authored(id_: str, title: str, score: float) -> MemoryRecord:
    """A memo-authored record: the path is the slugified title under a date
    prefix, exactly what `_build_rel_path` produces (`write_ops.py:1704`), and
    the title is derived from the body (`record.py:653`). Both metadata fields
    therefore restate the same text the body already carries."""
    return MemoryRecord(
        id=id_,
        path=f"2026-07-30-{_slugify(title)[:80]}.md",
        title=title,
        type="note",
        tags=[],
        created="2026-07-30T00:00:00+00:00",
        updated="2026-07-30T00:00:00+00:00",
        body=title,
        extra={},
        score=score,
    )


def test_entity_mention_titles_do_not_outrank_the_answering_record() -> None:
    """The live control shape, with the real fused scores.

    The 2026-08-04 audit scored these five candidates against the cross-encoder
    and recorded P(yes) of 0.990 for the record that answers the question and
    0.013 for the top-ranked one. Reconstructing the fused base through
    `alpha * ce + (1 - alpha) * rrf_bonus` (alpha=0.7, `config.py:631`) gives the
    scores below: the answer entered this stage at 0.768 and the junk at 0.309,
    a 2.5x lead. `retrieval_boost` delivered 0.461 vs 1.411 — an inversion of a
    2.5x spread, which is only possible because the multiplier outweighs the
    entire score range it is applied to.
    """
    answer = _memo_authored(
        "answer", "memo absorbió handoffs y continuidad, synapse quedó sin razón", 0.768
    )
    mentions = [
        _memo_authored("bug-boost-plano", "boost de título en synapse se deprecó plano", 0.309),
        _memo_authored("note-mypy", "fix de mypy en synapse tras deprecó", 0.346),
        _memo_authored("fact-deprecado", "synapse se deprecó", 0.503),
    ]

    harness = _Harness()
    harness.store = MagicMock()
    harness.store.get_health_batch.return_value = {}

    out = harness._apply_retrieval_boost(_QUERY, [answer, *mentions])

    # Guard against the vacuous pass: `_apply_retrieval_boost` swallows every
    # exception and returns its input unchanged, so a fixture that never reached
    # `boost_for` would "rank correctly" without having been scored at all.
    assert [r.score for r in out] != [0.768, 0.309, 0.346, 0.503]
    assert out[0].id == "answer", [(r.id, r.score) for r in out]


def test_title_does_not_recount_terms_the_filename_already_matched() -> None:
    """A self-titled record's filename and title carry the same terms. The old
    `title != filename` guard never caught it, because the filename is a *slug*
    of the title and so is never string-equal to it."""
    title = "synapse se deprecó"
    path = f"2026-07-30-{_slugify(title)}.md"

    with_title = boost_for(query=_QUERY, filename=path, title=title)
    filename_only = boost_for(query=_QUERY, filename=path)

    assert with_title == pytest.approx(filename_only)
    # And it stays *graded*: pinned below the cap, so a record with genuinely
    # independent metadata can still out-boost this one.
    assert with_title < _MAX_BOOST


def test_title_still_scores_terms_the_filename_missed() -> None:
    """De-duplication must not silence the title: a term the filename does not
    carry is new evidence and still earns its boost."""
    both = boost_for(query="aws legacy", filename="login-legacy.md", title="AWS Legacy Access")
    filename_only = boost_for(query="aws legacy", filename="login-legacy.md")

    assert both > filename_only


def test_boost_cannot_bury_a_substantially_stronger_body_match() -> None:
    """`_MAX_BOOST` is the displacement guarantee: a candidate scoring more than
    `_MAX_BOOST`x another cannot be overtaken by curatorial metadata."""
    metadata_perfect = boost_for(
        query="aws legacy",
        filename="aws-legacy.md",
        title="AWS Legacy",
        headings=["AWS Legacy procedure"],
        tags=["#aws", "#legacy"],
    )

    assert _MAX_BOOST <= 1.5
    assert metadata_perfect <= _MAX_BOOST

    # The control query's own spread: 0.768 vs 0.309, which the historical 12x
    # cap did not protect.
    assert 0.309 * metadata_perfect < 0.768
