from __future__ import annotations

import re

from memo.util import sha256_short, stable_hash, utc_now_iso

_ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


def test_utc_now_iso_has_z_suffix_and_no_offset() -> None:
    ts = utc_now_iso()
    assert _ISO_Z.match(ts), ts
    assert "+00:00" not in ts


def test_sha256_short_is_16_hex_and_deterministic() -> None:
    h = sha256_short("hello")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
    assert h == sha256_short("hello")
    assert h != sha256_short("world")


def test_sha256_short_handles_non_utf8_gracefully() -> None:
    # errors="replace" means a surrogate doesn't blow up.
    assert len(sha256_short("\ud800broken")) == 16


def test_stable_hash_is_order_independent() -> None:
    a = stable_hash({"x": 1, "y": 2})
    b = stable_hash({"y": 2, "x": 1})
    assert a == b
    assert len(a) == 64  # full sha256 hex


def test_stable_hash_coerces_non_json_types() -> None:
    from datetime import date

    # default=str — a date object must not raise.
    assert len(stable_hash({"d": date(2026, 5, 29)})) == 64


def test_stable_hash_distinguishes_values() -> None:
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
