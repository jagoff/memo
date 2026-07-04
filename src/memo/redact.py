"""Secret redaction + <private> span stripping (privacy layer).

Pure, stdlib-only, no MLX, no flag reads — callers gate with
``flag_bool("MEMO_REDACT_SECRETS")`` / ``flag_bool("MEMO_REDACT_ENTROPY")`` /
``flag_bool("MEMO_PRIVATE_MARKERS")`` so this module stays unit-testable
without touching the environment.

Two tiers:

- **Pattern tier** (near-zero false positives): provider-prefixed API keys
  (AWS ``AKIA``/``ASIA``, GitHub ``ghp_``/``github_pat_``…, OpenAI ``sk-``,
  Anthropic ``sk-ant-``, Slack ``xox*``, GCP ``AIza``) and PEM private-key
  blocks. Matches are masked to ``****<last4>`` so a leaked key stays
  identifiable without being usable.
- **Entropy tier** (opt-in, false-positive-prone): long mixed-class tokens
  with Shannon entropy >= 4.2 bits/char. Pure-hex strings are ALWAYS exempt
  (git SHAs and memo ids are hex and pepper every corpus).

``_PEM_RE`` needs the BEGIN and END markers in the SAME input string —
callers with line-oriented input (e.g. the sync gate's staged diff) must
join lines before scanning; token patterns are single-line so joining is
always safe.

Used by capture extraction (capture.py), vault ingest (cli_ingest.py) and
the pre-push sync gate (sync_git.py). Never imported on the recall-hook path.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# The anthropic/openai prefix overlap is resolved by the negative lookahead,
# so pattern order is not load-bearing.
_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,255}\b")),
    ("github-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("openai-key", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_-]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("gcp-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
)

_PEM_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)

_ENTROPY_CANDIDATE_RE = re.compile(r"\b[A-Za-z0-9+/_=-]{32,}\b")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_ENTROPY_MIN_BITS = 4.2

_PRIVATE_SPAN_RE = re.compile(r"<private>.*?</private>", re.DOTALL | re.IGNORECASE)
_PRIVATE_OPEN_RE = re.compile(r"<private>.*\Z", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class RedactionResult:
    """Masked text + the pattern kinds that fired (empty tuple = untouched)."""

    text: str
    found: tuple[str, ...]


def _mask(token: str) -> str:
    return "****" + token[-4:]


def _shannon_bits_per_char(s: str) -> float:
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _is_high_entropy(token: str) -> bool:
    if _HEX_RE.match(token):
        return False  # git SHAs / memo ids — never mask
    has_lower = any(c.islower() for c in token)
    has_upper = any(c.isupper() for c in token)
    has_digit = any(c.isdigit() for c in token)
    if not (has_lower and has_upper and has_digit):
        return False
    return _shannon_bits_per_char(token) >= _ENTROPY_MIN_BITS


def redact_secrets(text: str, *, entropy: bool = False) -> RedactionResult:
    """Mask every secret in `text` to ``****<last4>``; PEM blocks become
    ``****[private-key]``. Returns rewritten text + the kinds found."""
    if not text:
        return RedactionResult(text, ())
    found: list[str] = []
    out = text
    for kind, pat in _TOKEN_PATTERNS:
        if pat.search(out):
            found.append(kind)
            out = pat.sub(lambda m: _mask(m.group(0)), out)
    if _PEM_RE.search(out):
        found.append("private-key")
        out = _PEM_RE.sub("****[private-key]", out)
    if entropy:

        def _sub(m: re.Match[str]) -> str:
            tok = m.group(0)
            if _is_high_entropy(tok):
                if "high-entropy" not in found:
                    found.append("high-entropy")
                return _mask(tok)
            return tok

        out = _ENTROPY_CANDIDATE_RE.sub(_sub, out)
    return RedactionResult(out, tuple(found))


def scan_secrets(text: str, *, entropy: bool = False) -> list[tuple[str, str]]:
    """Report ``(kind, "****last4")`` per match WITHOUT rewriting — the sync
    gate wants findings for a block message, not a masked copy.

    ``text`` must contain multi-line secrets (PEM blocks) as one string —
    see the module docstring; join line-oriented input before calling."""
    if not text:
        return []
    findings: list[tuple[str, str]] = []
    for kind, pat in _TOKEN_PATTERNS:
        findings.extend((kind, _mask(m.group(0))) for m in pat.finditer(text))
    findings.extend(("private-key", "****[private-key]") for _ in _PEM_RE.finditer(text))
    if entropy:
        findings.extend(
            ("high-entropy", _mask(m.group(0)))
            for m in _ENTROPY_CANDIDATE_RE.finditer(text)
            if _is_high_entropy(m.group(0))
        )
    return findings


def strip_private_spans(text: str) -> str:
    """Drop ``<private>…</private>`` spans; an unclosed ``<private>`` drops
    everything to end-of-text (fail-closed: better to lose a capture than
    leak private content into a persisted memory)."""
    if not text or "<private>" not in text.lower():
        return text
    out = _PRIVATE_SPAN_RE.sub("", text)
    out = _PRIVATE_OPEN_RE.sub("", out)
    return out.strip()
