"""Resolve a person named in a query to their WhatsApp ``wa_jid`` via the
Obsidian ``Contacts/`` notes — the user's hand-written, authoritative contact
book. Pure: no cache, no network, no I/O beyond reading ``.md`` files in the
given directory.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

# Relación/Relation field value (folded — accents stripped) → kinship trigger
# words a user would type. Ambiguity (a trigger shared by 2+ jids) is dropped
# at build time in build_index().
_KINSHIP: dict[str, set[str]] = {
    "mama": {"mama", "madre", "mami"},
    "madre": {"mama", "madre", "mami"},
    "papa": {"papa", "padre", "papi"},
    "padre": {"papa", "padre", "papi"},
    "hermano": {"hermano", "hermana"},
    "hermana": {"hermano", "hermana"},
    "hijo": {"hijo", "hija"},
    "hija": {"hijo", "hija"},
    "esposa": {"esposa", "mujer"},
    "mujer": {"esposa", "mujer"},
    "esposo": {"esposo", "marido"},
    "marido": {"esposo", "marido"},
}

# `- **Label**: value` on a single line. Distinct character classes separate
# each quantifier group ([ \t]* around literal \*\*/: markers), so there is no
# adjacent-quantifier ambiguity for a regex engine to backtrack over.
_WA_JID_RE = re.compile(r"(?im)^-[ \t]*\*\*[ \t]*wa[_\s-]?jid[ \t]*\*\*[ \t]*:[ \t]*(.*)$")
_JID_FALLBACK_RE = re.compile(r"(?im)^-[ \t]*\*\*[ \t]*jid[ \t]*\*\*[ \t]*:[ \t]*(.*)$")
_APODO_RE = re.compile(r"(?im)^-[ \t]*\*\*[ \t]*Apodo[ \t]*\*\*[ \t]*:[ \t]*(.*)$")
_FULL_NAME_RE = re.compile(
    r"(?im)^-[ \t]*\*\*[ \t]*(?:Apellido[ \t]*/[ \t]*nombre[ \t]+completo|Full[ \t]*name)"
    r"[ \t]*\*\*[ \t]*:[ \t]*(.*)$"
)
_RELATION_RE = re.compile(
    r"(?im)^-[ \t]*\*\*[ \t]*(?:Relaci[oó]n|Relation)[ \t]*\*\*[ \t]*:[ \t]*(.*)$"
)

_FULL_NAME_TOKEN_RE = re.compile(r"[a-záéíóúñ]{3,}")


def _fold(text: str) -> str:
    """Lowercase + strip diacritics for accent-insensitive matching."""
    normalized = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).strip()


def _extract(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def build_index(contacts_dir: Path) -> dict[str, str]:
    """Map trigger word (folded) → wa_jid from Contacts notes in ``contacts_dir``.

    A trigger shared by 2+ distinct jids is ambiguous and dropped.
    """
    trigger_jids: dict[str, set[str]] = {}

    def _add(word: str, jid: str) -> None:
        folded = _fold(word)
        if len(folded) < 2:
            return
        trigger_jids.setdefault(folded, set()).add(jid)

    if not contacts_dir.is_dir():
        return {}

    for path in sorted(contacts_dir.glob("*.md")):
        if path.name.startswith((".", "_")):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue

        jid = _extract(_WA_JID_RE, text) or _extract(_JID_FALLBACK_RE, text)
        if not jid:
            continue

        _add(path.stem, jid)

        apodo = _extract(_APODO_RE, text)
        if apodo:
            _add(apodo, jid)

        full_name = _extract(_FULL_NAME_RE, text)
        for token in _FULL_NAME_TOKEN_RE.findall(full_name.lower()):
            _add(token, jid)

        relation = _extract(_RELATION_RE, text)
        for trigger in _KINSHIP.get(_fold(relation), set()):
            _add(trigger, jid)

    return {trigger: next(iter(jids)) for trigger, jids in trigger_jids.items() if len(jids) == 1}


def resolve_jid(query: str, index: dict[str, str]) -> str | None:
    """Resolve a person named in ``query`` to a single wa_jid via ``index``.

    Word-boundary, accent-insensitive match of query text against the trigger
    index. Returns the jid only when exactly one distinct contact matches;
    ``None`` on no match or ambiguity.
    """
    if not index:
        return None
    folded = _fold(query)
    matched: set[str] = set()
    for trigger, jid in index.items():
        if re.search(rf"\b{re.escape(trigger)}\b", folded):
            matched.add(jid)
    if len(matched) == 1:
        return next(iter(matched))
    return None
