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
from typing import Any

from memo.errors import ValidationError

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
    r"-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----.*?-----END [A-Z ]{0,40}PRIVATE KEY-----",
    re.DOTALL,
)

_ENTROPY_CANDIDATE_RE = re.compile(r"\b[A-Za-z0-9+/_=-]{32,}\b")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_ENTROPY_MIN_BITS = 4.2

# Fixed-width literal markers only (no ``.*`` — linear, ReDoS-free). Span
# logic lives in ``strip_private_spans`` as a linear scan over these matches.
_PRIVATE_MARK_OPEN_RE = re.compile(r"<private>", re.IGNORECASE)
_PRIVATE_MARK_CLOSE_RE = re.compile(r"</private>", re.IGNORECASE)


@dataclass(frozen=True)
class RedactionResult:
    """Masked text + the pattern kinds that fired (empty tuple = untouched)."""

    text: str
    found: tuple[str, ...]


@dataclass(frozen=True)
class SanitizedMemoryInput:
    """Complete caller-controlled record after persistence sanitization."""

    content: str
    title: str | None
    tags: list[str]
    topic_key: str | None
    normalized_hash: str | None
    extra: dict[str, Any]
    changed: bool


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
    if not text or _PRIVATE_MARK_OPEN_RE.search(text) is None:
        return text
    # Linear scan (indices into the original string — no lowercased copy, so
    # no Unicode length-drift): keep text before each ``<private>``, skip to
    # the next ``</private>``; an unclosed open drops everything to EOT.
    parts: list[str] = []
    i = 0
    while True:
        m_open = _PRIVATE_MARK_OPEN_RE.search(text, i)
        if m_open is None:
            parts.append(text[i:])
            break
        parts.append(text[i : m_open.start()])
        m_close = _PRIVATE_MARK_CLOSE_RE.search(text, m_open.end())
        if m_close is None:
            break
        i = m_close.end()
    return "".join(parts).strip()


def sanitize_persisted_text(text: str, *, entropy: bool = False) -> RedactionResult:
    """Mandatory final text sanitizer used by every persistence path."""
    private_stripped = strip_private_spans(text)
    redacted = redact_secrets(private_stripped, entropy=entropy)
    found = list(redacted.found)
    if private_stripped != text:
        found.insert(0, "private-span")
    return RedactionResult(redacted.text, tuple(dict.fromkeys(found)))


def _sanitize_value(value: Any, *, entropy: bool) -> tuple[Any, bool]:
    if isinstance(value, str):
        result = sanitize_persisted_text(value, entropy=entropy)
        return result.text, result.text != value
    if isinstance(value, list):
        changed = False
        out: list[Any] = []
        for item in value:
            clean, item_changed = _sanitize_value(item, entropy=entropy)
            out.append(clean)
            changed = changed or item_changed
        return out, changed
    if isinstance(value, tuple):
        changed = False
        out_items: list[Any] = []
        for item in value:
            clean, item_changed = _sanitize_value(item, entropy=entropy)
            out_items.append(clean)
            changed = changed or item_changed
        return tuple(out_items), changed
    if isinstance(value, dict):
        changed = False
        out_dict: dict[Any, Any] = {}
        for key, item in value.items():
            clean_key, key_changed = _sanitize_value(key, entropy=entropy)
            if isinstance(clean_key, str) and not clean_key.strip():
                raise ValidationError("sanitized metadata contains an empty key")
            if clean_key in out_dict:
                raise ValidationError("sanitized metadata contains colliding keys")
            clean_item, item_changed = _sanitize_value(item, entropy=entropy)
            out_dict[clean_key] = clean_item
            changed = changed or key_changed or item_changed
        return out_dict, changed
    return value, False


def sanitize_memory_input(
    *,
    content: str,
    title: str | None = None,
    tags: list[str] | None = None,
    topic_key: str | None = None,
    normalized_hash: str | None = None,
    extra: dict[str, Any] | None = None,
    entropy: bool = False,
    allow_empty_content: bool = False,
) -> SanitizedMemoryInput:
    """Sanitize every caller-controlled value before it can be persisted.

    Pattern masking and private-span stripping are unconditional. Entropy
    scanning is the only optional tier because it has a higher false-positive
    rate. The function is pure and never logs source values.
    """
    clean_content = sanitize_persisted_text(content, entropy=entropy).text
    if not clean_content.strip() and not allow_empty_content:
        raise ValidationError("memory content is empty after privacy sanitization")

    clean_title: str | None = None
    if title is not None:
        candidate = sanitize_persisted_text(title, entropy=entropy).text.strip()
        clean_title = candidate or None

    clean_tags: list[str] = []
    for tag in tags or []:
        candidate = sanitize_persisted_text(str(tag), entropy=entropy).text.strip()
        if candidate:
            clean_tags.append(candidate)

    clean_topic: str | None = None
    if topic_key is not None:
        candidate = sanitize_persisted_text(topic_key, entropy=entropy).text.strip()
        if not candidate:
            raise ValidationError("topic_key is empty after privacy sanitization")
        clean_topic = candidate

    clean_legacy_hash: str | None = None
    if normalized_hash is not None:
        clean_legacy_hash = sanitize_persisted_text(normalized_hash, entropy=entropy).text.strip()
        clean_legacy_hash = clean_legacy_hash or None

    clean_extra_raw, extra_changed = _sanitize_value(dict(extra or {}), entropy=entropy)
    clean_extra = dict(clean_extra_raw)
    changed = (
        clean_content != content
        or clean_title != title
        or clean_tags != list(tags or [])
        or clean_topic != topic_key
        or clean_legacy_hash != normalized_hash
        or extra_changed
    )
    if changed and not any(tag.casefold() == "_redacted" for tag in clean_tags):
        clean_tags.append("_redacted")

    return SanitizedMemoryInput(
        content=clean_content,
        title=clean_title,
        tags=clean_tags,
        topic_key=clean_topic,
        normalized_hash=clean_legacy_hash,
        extra=clean_extra,
        changed=changed,
    )
