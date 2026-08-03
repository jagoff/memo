from __future__ import annotations

import importlib.util
import io
import re
from types import SimpleNamespace

from memo import cli_viz
from memo.cli_dashboard import _make_handler

_ATTACK = '</script><script id="memo-xss">globalThis.__memo_xss=1</script>'


def test_map_renderer_html_escapes_json_and_hardens_scripts() -> None:
    render = getattr(cli_viz, "_render_map_html", None)
    assert callable(render), "map needs one security-aware renderer"

    html = render(
        {
            "xs": [0.0],
            "ys": [0.0],
            "ids": ["deadbeef"],
            "titles": [_ATTACK],
            "types": [_ATTACK],
            "tags": [_ATTACK],
            "created": ["2026-01-01"],
            "updated": ["2026-01-01"],
            "frames": [],
            "method": "PCA",
            "n": 1,
            "type_colors": {"note": "#fff"},
        },
        nonce="fixed-test-nonce",
    )

    assert _ATTACK not in html
    assert r"\u003c/script\u003e" in html
    assert "https://cdn.plot.ly" not in html
    assert "Plotly." not in html
    assert "getContext('2d')" in html
    assert "addEventListener('mousemove'" in html
    assert "document.execCommand('copy')" in html
    assert "Content-Security-Policy" in html
    assert "script-src 'nonce-fixed-test-nonce'" in html
    assert '<script nonce="fixed-test-nonce">' in html
    assert '<script src="' not in html


def test_dashboard_builder_is_a_real_packaged_module() -> None:
    assert importlib.util.find_spec("memo.web_build") is not None


def test_dashboard_renderer_html_escapes_json_and_hardens_dynamic_html() -> None:
    from memo import web_build

    html = web_build._render_html(
        {
            "generated_at": "2026-01-01T00:00:00Z",
            "memo_version": "test",
            "gerencial": {"funnel": [{"label": _ATTACK, "sub": _ATTACK, "value": 1}]},
            "verdict": {},
            "usefulness": {
                "consumers": [
                    {
                        "consumer": _ATTACK,
                        "consults": 1,
                        "grounded_rate": None,
                        "hit_rate": 0,
                        "last_seen": None,
                    }
                ],
                "silent": [_ATTACK],
            },
            "gaps": [{"prompt": _ATTACK, "reasons": [_ATTACK], "count": 1}],
            "pillars": [],
            "sync": {},
        },
        nonce="fixed-test-nonce",
    )

    assert _ATTACK not in html
    assert r"\u003c/script\u003e" in html
    assert "https://cdn.plot.ly" not in html
    assert "Plotly." not in html
    assert "renderBarChart" in html
    assert "createElementNS" in html
    assert "script-src 'nonce-fixed-test-nonce'" in html
    assert '<script src="' not in html
    assert len(re.findall(r'<script[^>]+nonce="fixed-test-nonce"', html)) == 2
    assert "esc(s.label)" in html
    assert "esc(c.consumer)" in html
    assert "silent.map(esc).join" in html
    assert "${esc(n)}&times;" in html


def test_dashboard_http_responses_send_csp_and_browser_hardening(tmp_cfg) -> None:
    handler = _make_handler(
        object(),
        tmp_cfg,
        "<html></html>",
        5,
        csp="test-csp",
        capability_token="test-token",
    )
    headers: list[tuple[str, str]] = []
    fake = SimpleNamespace(
        send_response=lambda _status: None,
        send_header=lambda name, value: headers.append((name.lower(), value)),
        end_headers=lambda: None,
        wfile=io.BytesIO(),
    )

    handler._send(fake, 200, b"<html></html>", "text/html; charset=utf-8")

    assert ("content-security-policy", "test-csp") in headers
    assert ("x-content-type-options", "nosniff") in headers
    assert ("referrer-policy", "no-referrer") in headers
    assert ("x-frame-options", "DENY") in headers
    assert ("cache-control", "no-store") in headers
