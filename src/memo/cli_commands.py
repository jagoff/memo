"""CLI commands registry - refactored from monolithic cli.py.

Este módulo organiza los comandos CLI en grupos lógicos:
- Memory: save, search, list, get, delete, update
- Repo: repo-index, repo-embed, repo-search, repo-list
- Session: session list, session get, session autosave
- Temporal: temporal-contradictions, temporal-timeline, temporal-stale
- Graph: graph-path, graph-neighbors, graph-communities
- Utilities: stats, doctor, reindex, briefing

Cada grupo puede ser refactorizado a su propio módulo si crece.
"""

from __future__ import annotations

# Placeholder para refactoring futuro. Mantiene estructura actual funcional.
# TODO: Separar en:
# - cli/memory.py (save, search, list, get, delete, update)
# - cli/repo.py (repo-index, repo-embed, repo-search, repo-list)
# - cli/session.py (session list, session get, session autosave)
# - cli/temporal.py (temporal-contradictions, temporal-timeline, temporal-stale)
# - cli/graph.py (graph-path, graph-neighbors, graph-communities)
# - cli/utils.py (stats, doctor, reindex, briefing)

__all__ = []
