"""Unit tests for _normalize_relative_dates in consolidate_ops."""

from __future__ import annotations

import datetime

from memo.memory.consolidate_ops import _normalize_relative_dates

REF = datetime.date(2026, 6, 18)


def test_ayer_replaced():
    result = _normalize_relative_dates("ayer decidimos usar sqlite", REF)
    assert "2026-06-17" in result
    assert "ayer" in result  # kept, with ISO appended


def test_hoy_replaced():
    result = _normalize_relative_dates("hoy cerramos el PR", REF)
    assert "2026-06-18" in result


def test_anteayer_replaced():
    result = _normalize_relative_dates("anteayer el build falló", REF)
    assert "2026-06-16" in result


def test_yesterday_english():
    result = _normalize_relative_dates("yesterday we merged the fix", REF)
    assert "2026-06-17" in result


def test_today_english():
    result = _normalize_relative_dates("today the deploy happened", REF)
    assert "2026-06-18" in result


def test_la_semana_pasada():
    result = _normalize_relative_dates("la semana pasada actualizamos deps", REF)
    assert "2026-06-11" in result


def test_last_week_english():
    result = _normalize_relative_dates("last week we upgraded deps", REF)
    assert "2026-06-11" in result


def test_el_mes_pasado():
    result = _normalize_relative_dates("el mes pasado migramos a uv", REF)
    assert "2026-05" in result


def test_last_month_english():
    result = _normalize_relative_dates("last month we migrated to uv", REF)
    assert "2026-05" in result


def test_hace_n_dias():
    result = _normalize_relative_dates("hace 3 días se rompió el test", REF)
    assert "2026-06-15" in result


def test_n_days_ago_english():
    result = _normalize_relative_dates("5 days ago the server went down", REF)
    assert "2026-06-13" in result


def test_no_match_returns_original():
    text = "nothing temporal here"
    assert _normalize_relative_dates(text, REF) == text


def test_never_raises_on_bad_input():
    result = _normalize_relative_dates("hace abc días de algo", REF)
    assert isinstance(result, str)
