"""Security primitives for generated HTML that embeds memo-owned data.

Generated views use memo's small native Canvas/SVG renderers. They therefore
remain fully offline and need no third-party script exception in their CSP.
"""

from __future__ import annotations

import json
import secrets
from typing import Any


def html_safe_json(
    value: Any,
    *,
    ensure_ascii: bool = False,
    default: Any = None,
) -> str:
    """Serialize JSON so it cannot terminate its surrounding HTML script node."""
    payload = json.dumps(value, ensure_ascii=ensure_ascii, default=default)
    return (
        payload.replace("&", r"\u0026")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("\u2028", r"\u2028")
        .replace("\u2029", r"\u2029")
    )


def new_csp_nonce() -> str:
    """Return an unpredictable, HTML-attribute-safe CSP nonce."""
    return secrets.token_urlsafe(18)


def content_security_policy(nonce: str, *, allow_local_fetch: bool) -> str:
    """CSP for generated pages; only the live dashboard may fetch local JSON."""
    connect_src = "'self'" if allow_local_fetch else "'none'"
    return "; ".join(
        (
            "default-src 'none'",
            f"script-src 'nonce-{nonce}'",
            "style-src 'unsafe-inline'",
            "img-src data:",
            f"connect-src {connect_src}",
            "font-src 'none'",
            "object-src 'none'",
            "base-uri 'none'",
            "form-action 'none'",
        )
    )
