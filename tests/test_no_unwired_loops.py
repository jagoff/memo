"""Wiring gate — fails if an unwired/half-wired loop reappears.

Standing enforcement of the "close every loop" invariant (mirrors the
completeness pattern of ``test_dream_flags``): a change cannot merge if it
introduces a dead flag (declared, never read) or a half-wired marker in a
reachable source path. Zero-tolerance — no baseline file, no ratchet.

Two properties:
1. Every registered ``MEMO_*`` flag has at least one reader (a ``flag_*()`` /
   accessor / ``os.environ`` reference on a line other than its own ``_spec``
   declaration — so same-file accessor wrappers like ``flag_crusher_*`` count).
2. No half-wired markers ("not yet implemented", "return placeholder data", …)
   in ``src/memo`` production code.
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src" / "memo"

# Half-wired phrases that mean "a loop is left open" — precise, not the bare
# word "placeholder" (which legitimately appears in SQL bind-param helpers and
# HTML attributes). Extend deliberately; do not weaken.
_HALFWIRED = re.compile(
    r"not yet implemented"
    r"|not yet wired"
    r"|return placeholder data"
    r"|placeholder for now"
    r"|for now, skip"
    r"|for now, return placeholder",
    re.IGNORECASE,
)


def _src_files() -> list[Path]:
    return [p for p in _SRC.rglob("*.py")]


_REPO = _SRC.parent.parent


def _declared_flags() -> dict[str, Path]:
    """Every MEMO_* flag declared via _spec(...) in flags_*.py → its file."""
    decl: dict[str, Path] = {}
    for f in _SRC.glob("flags_*.py"):
        for m in re.finditer(r'_spec\(\s*"(MEMO_[A-Z0-9_]+)"', f.read_text(encoding="utf-8")):
            decl[m.group(1)] = f
    return decl


def _reader_texts() -> list[str]:
    """Files that legitimately CONSUME a flag: memo's Python source plus the
    non-Python consumers (the bash statusline / shell assets read some flags
    directly from the env). Excludes tests/ and docs/ — a flag read only by a
    test is still dead in production."""
    consumer_suffixes = {".py", ".sh", ".bash", ".plist", ".json", ".template"}
    texts: list[str] = []
    # memo's own source, incl. bundled assets under src/memo/agent_assets/.
    for p in _SRC.rglob("*"):
        if p.is_file() and p.suffix in consumer_suffixes:
            texts.append(p.read_text(encoding="utf-8", errors="ignore"))
    # Repo-root consumer dirs (installed statusline / hooks / launchd assets).
    for sub in ("statusline", "hooks", "launchd", "scripts"):
        d = _REPO / sub
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix in consumer_suffixes:
                texts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return texts


def test_no_dead_flags() -> None:
    """A registered flag with zero readers is a dead loop — declared behavior
    the user can set that does nothing. Every flag must be read somewhere."""
    decl = _declared_flags()
    # Build a reader index once: name -> total occurrences across consumer files
    # (src + bash/plist assets), and occurrences that are its own `_spec("NAME"`
    # declaration. A reader is any occurrence beyond the declaration itself.
    texts = _reader_texts()
    dead: list[str] = []
    for name in sorted(decl):
        total = 0
        decl_hits = 0
        spec_re = re.compile(r'_spec\(\s*"' + re.escape(name) + r'"')
        for txt in texts:
            total += txt.count(name)
            decl_hits += len(spec_re.findall(txt))
        if total - decl_hits <= 0:
            dead.append(name)
    assert not dead, (
        "Dead flags (declared but never read — wire a reader or delete the spec): "
        + ", ".join(dead)
    )


def test_no_halfwired_markers() -> None:
    """No 'not yet implemented' / 'return placeholder data' style markers in
    production source — every such loop must be closed (wired or deleted)."""
    hits: list[str] = []
    for p in _src_files():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if _HALFWIRED.search(line):
                hits.append(f"{p.relative_to(_SRC.parent.parent)}:{i}: {line.strip()[:80]}")
    assert not hits, "Half-wired markers found (close the loop — wire or delete):\n" + "\n".join(
        hits
    )
