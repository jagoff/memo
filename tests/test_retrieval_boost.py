"""Tests for filename/title/heading/tag retrieval boost."""

from __future__ import annotations

from memo.retrieval_boost import boost_for, query_terms

# ---------- query_terms ----------


def test_query_terms_drops_stopwords() -> None:
    assert query_terms("Cuales son los pasos para correr goku") == [
        "son", "pasos", "correr", "goku",
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


def test_filename_exact_match_boost_4x() -> None:
    b = boost_for(query="correr goku", filename="02-Areas/Correr Goku.md")
    assert b >= 4.0


def test_filename_half_match_boost_2x() -> None:
    b = boost_for(query="correr goku otra cosa", filename="Correr Goku.md")
    # 2/3 terms in filename → >= 0.5 → x2
    assert 1.5 < b <= 3.0


def test_filename_no_match_returns_1() -> None:
    b = boost_for(query="aws legacy", filename="random-notes.md")
    assert b == 1.0


def test_title_match_adds_boost() -> None:
    b = boost_for(
        query="aws legacy",
        filename="login-legacy.md",
        title="AWS Legacy Access Procedure",
    )
    # filename has "legacy" (1/2 = 0.5 → x2), title has "aws legacy" (2/2 → x1.5)
    assert b >= 2.0 * 1.5 * 0.99


def test_tag_match_adds_boost() -> None:
    b_no_tag = boost_for(query="goku", filename="other.md")
    b_with_tag = boost_for(query="goku", filename="other.md", tags=["#goku"])
    assert b_with_tag > b_no_tag
    assert b_with_tag >= 1.4


def test_heading_match_adds_boost() -> None:
    b = boost_for(
        query="aws legacy",
        filename="other.md",
        headings=["## AWS Legacy access flow", "## Other section"],
    )
    assert b >= 1.25


def test_total_boost_cap_around_10() -> None:
    b = boost_for(
        query="aws legacy",
        filename="aws-legacy.md",
        title="AWS Legacy",
        headings=["AWS Legacy procedure"],
        tags=["#aws", "#legacy"],
    )
    # 4 (filename exact) × 1.5 (title) × 1.25 (heading) × 1.4 (tag) = 10.5
    assert 10.0 < b <= 11.0


def test_filename_with_extension_handled() -> None:
    # `.stem` strips extension; "correr goku" should match.
    b1 = boost_for(query="correr goku", filename="Correr Goku.md")
    b2 = boost_for(query="correr goku", filename="path/to/Correr Goku.MD")
    assert b1 >= 4.0
    assert b2 >= 4.0


def test_filename_partial_match_boost_1_3x() -> None:
    # 1/3 of terms in filename → <0.5 but >0 → x1.3
    b = boost_for(query="aws legacy login", filename="aws-only.md")
    assert 1.2 < b < 1.4
