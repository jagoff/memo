"""Tests for filename/title/heading/tag retrieval boost."""

from __future__ import annotations

import pytest

from memo.retrieval_boost import _MAX_BOOST, boost_for, query_terms

# ---------- query_terms ----------


def test_query_terms_drops_stopwords() -> None:
    assert query_terms("Cuales son los pasos para correr goku") == [
        "son",
        "pasos",
        "correr",
        "goku",
    ]


def test_query_terms_min_length_3() -> None:
    assert query_terms("aa bb cc abc def ghi") == ["abc", "def", "ghi"]


def test_query_terms_empty_input() -> None:
    assert query_terms("") == []


def test_query_terms_handles_punctuation() -> None:
    # Stopwords like "Cómo" lose accent in lowercase; "como" is in stoplist.
    assert "como" not in query_terms("Cómo, instalar la app?")


# ---------- boost_for ----------


def test_no_query_terms_returns_1() -> None:
    # Only stopwords → no boost.
    assert boost_for(query="los los", filename="anything.md") == 1.0


def test_filename_exact_match_is_the_strongest_single_signal() -> None:
    b = boost_for(query="correr goku", filename="02-Areas/Correr Goku.md")
    assert b == pytest.approx(1.30)


def test_filename_half_match_scores_below_an_exact_one() -> None:
    b = boost_for(query="correr goku otra cosa", filename="Correr Goku.md")
    # 2/3 terms in filename → >= 0.5 → the mid tier
    assert b == pytest.approx(1.15)
    assert b < boost_for(query="correr goku", filename="Correr Goku.md")


def test_filename_no_match_returns_1() -> None:
    b = boost_for(query="aws legacy", filename="random-notes.md")
    assert b == 1.0


def test_title_match_adds_boost() -> None:
    filename_only = boost_for(query="aws legacy", filename="login-legacy.md")
    b = boost_for(
        query="aws legacy",
        filename="login-legacy.md",
        title="AWS Legacy Access Procedure",
    )
    # filename has "legacy" (1/2 → mid tier); the title newly covers "aws", the
    # one term the filename missed, so it compounds.
    assert b > filename_only
    assert b == pytest.approx(1.15 * 1.08)


def test_tag_match_adds_boost() -> None:
    b_no_tag = boost_for(query="goku", filename="other.md")
    b_with_tag = boost_for(query="goku", filename="other.md", tags=["#goku"])
    assert b_with_tag > b_no_tag
    assert b_with_tag == pytest.approx(1.08)


def test_heading_match_adds_boost() -> None:
    b = boost_for(
        query="aws legacy",
        filename="other.md",
        headings=["## AWS Legacy access flow", "## Other section"],
    )
    assert b == pytest.approx(1.10)


def test_total_boost_is_hard_capped() -> None:
    b = boost_for(
        query="aws legacy",
        filename="aws-legacy.md",
        title="AWS Legacy",
        headings=["AWS Legacy procedure"],
        tags=["#aws", "#legacy"],
    )
    # filename×1.30 · heading×1.10 · tag×1.08 = 1.544 → capped. The title adds
    # nothing: the filename already matched both of its terms.
    assert b == pytest.approx(_MAX_BOOST)


def test_title_scales_with_overlap() -> None:
    # Near-exact frontmatter title (distinct from filename) must out-boost a
    # half-match — the asymmetry that let a terse correct note get blended.
    near = boost_for(query="deploy lambda", title="Deploy nuevas lambda")
    half = boost_for(query="deploy lambda aws prod", title="Deploy nuevas lambda")
    assert near == pytest.approx(1.20)
    assert near > half


def test_filename_with_extension_handled() -> None:
    # `.stem` strips extension; "correr goku" should match.
    b1 = boost_for(query="correr goku", filename="Correr Goku.md")
    b2 = boost_for(query="correr goku", filename="path/to/Correr Goku.MD")
    assert b1 == pytest.approx(1.30)
    assert b2 == pytest.approx(1.30)


def test_filename_partial_match_is_the_weakest_tier() -> None:
    # 1/3 of terms in filename → <0.5 but >0 → the weakest filename tier
    b = boost_for(query="aws legacy login", filename="aws-only.md")
    assert b == pytest.approx(1.06)
