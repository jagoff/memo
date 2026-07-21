"""Cheap proactive surfaces must not drag in `suggester`'s LLM chain (I4
review fix).

`memo.proactive.surfaces` is on the digest/briefing hot path. Before the fix,
`memo/proactive/__init__.py` did an eager
`from .suggester import ProactiveSuggester, ...` at package load, and
`suggester.py` imports `memo.llm` at module top — so importing
`memo.proactive.surfaces` (which only needs the package `__init__` to run,
not `suggester`) paid the LLM-graph import cost anyway, even though
`surfaces.py` never touches `suggester`. The re-export is now lazy via
`__getattr__`.

Note: this test asserts `memo.proactive.suggester` (NOT `memo.llm`) stays out
of `sys.modules`. `memo.llm` itself is unavoidably already loaded the instant
`import memo` runs — `memo/__init__.py` eagerly imports `Memory`, and both
`memo/memory/__init__.py` and `memo/memory/facade.py` import `memo.llm` at
module top — which happens before `memo.proactive` is ever reached and is a
separate, pre-existing, out-of-scope issue. `memo.proactive.suggester` is the
correct, fix-scoped signal: verified (by diffing against the pre-fix
`__init__.py`) to be present in `sys.modules` after `import
memo.proactive.surfaces` before this fix, and absent after.

Run in a subprocess: other tests in the same pytest session may have already
imported these modules, which would make an in-process `sys.modules` check a
false negative.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_importing_proactive_surfaces_does_not_load_suggester() -> None:
    code = (
        "import sys\n"
        "import memo.proactive.surfaces\n"
        "assert 'memo.proactive.suggester' not in sys.modules, "
        "sorted(m for m in sys.modules if m.startswith('memo.proactive'))\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_proactive_suggester_still_importable_via_lazy_getattr() -> None:
    from memo.proactive import ProactiveSuggester, Suggestion, SuggestionFeedback

    assert ProactiveSuggester.__name__ == "ProactiveSuggester"
    assert Suggestion.__name__ == "Suggestion"
    assert SuggestionFeedback.__name__ == "SuggestionFeedback"


def test_unknown_attribute_still_raises_attribute_error() -> None:
    import memo.proactive as proactive_pkg

    with pytest.raises(AttributeError):
        _ = proactive_pkg.definitely_not_a_real_export
