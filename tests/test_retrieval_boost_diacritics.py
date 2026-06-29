"""Diacritic-insensitive retrieval boost.

FTS5 is built with ``unicode61 remove_diacritics 2`` (store/schema.py), so a
record matches an accented Spanish query diacritic-insensitively. The curatorial
boost must fold accents the same way, or an accented query loses the 4-10x
filename/title boost on the very note that answers it.
"""

from __future__ import annotations

from memo.retrieval_boost import _fold_diacritics, boost_for, query_terms


def test_fold_diacritics_strips_accents() -> None:
    assert _fold_diacritics("decisión") == "decision"
    assert _fold_diacritics("español") == "espanol"


def test_query_terms_are_diacritic_folded() -> None:
    terms = query_terms("decisión español")
    assert "decision" in terms and "espanol" in terms


def test_accented_query_earns_filename_boost_on_unaccented_note() -> None:
    accented = boost_for(query="español", filename="espanol.md")
    plain = boost_for(query="espanol", filename="espanol.md")
    assert accented > 1.0
    assert accented == plain


def test_accented_query_earns_title_boost_on_unaccented_note() -> None:
    accented = boost_for(query="decisión markdown", title="Decision Markdown")
    plain = boost_for(query="decision markdown", title="Decision Markdown")
    assert accented > 1.0
    assert accented == plain
