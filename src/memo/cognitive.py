"""EXPERIMENTAL — not covered by the test suite, not exposed via MCP. API may change without notice.

Memoria con Estado Mental del Usuario (Cognitive State Model).

Modela el estado mental, objetivos y contexto del usuario para proporcionar
información relevante en el momento justo.

## Gamechanger

- Entiende QUÉ estás tratando de lograr, no solo QUÉ has guardado
- Adapta las respuestas según tu estado mental actual (enfatizado, relajado, buscando)
- Proporciona información proactiva basada en tus objetivos actuales
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class MentalState(Enum):
    """Estados mentales del usuario."""
    FOCUSED = "focused"
    RELAXED = "relaxed"
    STRESSED = "stressed"
    EXPLORING = "exploring"
    PROBLEM_SOLVING = "problem_solving"
    LEARNING = "learning"


class ContextType(Enum):
    """Tipos de contexto del usuario."""
    WORK = "work"
    PERSONAL = "personal"
    RESEARCH = "research"
    CREATIVE = "creative"
    ROUTINE = "routine"


@dataclass
class CognitiveState:
    """Estado cognitivo actual del usuario."""
    timestamp: str
    mental_state: str
    context_type: str
    current_goal: str | None
    focus_area: str | None
    energy_level: int  # 0-100
    stress_level: int  # 0-100


@dataclass
class ContextualSuggestion:
    """Sugerencia contextual basada en estado mental."""
    suggestion_id: str
    memoria_id: str
    relevance_reason: str
    confidence: float
    suggested_at: str


class CognitiveStateTracker:
    """Rastrea el estado cognitivo del usuario.

    Args:
        state_dir: Directorio para almacenar el estado.
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.state_file = state_dir / "cognitive_state.json"
        self.history_file = state_dir / "cognitive_history.json"
        self._current_state: CognitiveState | None = None
        self._history: list[CognitiveState] = []
        self._load()

    def _load(self) -> None:
        """Carga el estado desde disco."""
        if self.state_file.is_file():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                self._current_state = CognitiveState(**data)
            except Exception:
                self._current_state = None

        if self.history_file.is_file():
            try:
                data = json.loads(self.history_file.read_text(encoding="utf-8"))
                self._history = [CognitiveState(**h) for h in data]
            except Exception:
                self._history = []

    def _save(self) -> None:
        """Guarda el estado a disco."""
        try:
            if self._current_state:
                self.state_file.write_text(
                    json.dumps(self._current_state.__dict__, indent=2),
                    encoding="utf-8",
                )

            self.history_file.write_text(
                json.dumps([h.__dict__ for h in self._history], indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def update_state(
        self,
        mental_state: str,
        context_type: str,
        current_goal: str | None = None,
        focus_area: str | None = None,
        energy_level: int = 50,
        stress_level: int = 30,
    ) -> CognitiveState:
        """Actualiza el estado cognitivo actual.

        Args:
            mental_state: Estado mental actual.
            context_type: Tipo de contexto.
            current_goal: Objetivo actual.
            focus_area: Área de enfoque.
            energy_level: Nivel de energía (0-100).
            stress_level: Nivel de stress (0-100).

        Returns:
            CognitiveState actualizado.
        """
        self._current_state = CognitiveState(
            timestamp=datetime.now(UTC).isoformat(),
            mental_state=mental_state,
            context_type=context_type,
            current_goal=current_goal,
            focus_area=focus_area,
            energy_level=energy_level,
            stress_level=stress_level,
        )
        self._history.append(self._current_state)

        # Keep only last 100 entries
        if len(self._history) > 100:
            self._history = self._history[-100:]

        self._save()
        return self._current_state

    def get_current_state(self) -> CognitiveState | None:
        """Obtiene el estado cognitivo actual.

        Returns:
            CognitiveState actual o None.
        """
        return self._current_state

    def get_history(self, limit: int = 10) -> list[CognitiveState]:
        """Obtiene el historial de estados.

        Args:
            limit: Máximo de resultados.

        Returns:
            Lista de CognitiveState.
        """
        return self._history[-limit:]


class ContextAwareRetrieval:
    """Recuperación basada en estado mental y contexto.

    Args:
        tracker: CognitiveStateTracker.
    """

    def __init__(self, tracker: CognitiveStateTracker) -> None:
        self.tracker = tracker

    def retrieve_with_context(
        self,
        query: str,
        search_func: Any,
        limit: int = 10,
    ) -> list[Any]:
        """Recupera información considerando el estado mental.

        Args:
            query: Query de búsqueda.
            search_func: Función de búsqueda.
            limit: Máximo de resultados.

        Returns:
            Lista de resultados adaptados al contexto.
        """
        state = self.tracker.get_current_state()

        # Si no hay estado, usar búsqueda normal
        if not state:
            return search_func(query, limit=limit)

        # Adaptar la query según el estado mental
        adapted_query = self._adapt_query_to_state(query, state)

        # Realizar búsqueda
        results = search_func(adapted_query, limit=limit)

        # Re-rank según contexto
        ranked = self._rerank_by_context(results, state)

        return ranked[:limit]

    def _adapt_query_to_state(self, query: str, state: CognitiveState) -> str:
        """Adapta la query según el estado mental.

        Args:
            query: Query original.
            state: Estado cognitivo.

        Returns:
            Query adaptada.
        """
        # Si el usuario está resolviendo problemas, enfocar en soluciones
        if state.mental_state == MentalState.PROBLEM_SOLVING.value:
            return f"{query} solution approach fix"

        # Si está aprendiendo, enfocar en fundamentos
        if state.mental_state == MentalState.LEARNING.value:
            return f"{query} basics tutorial explanation"

        # Si está explorando, enfocar en descubrimiento
        if state.mental_state == MentalState.EXPLORING.value:
            return f"{query} related similar alternative"

        return query

    def _rerank_by_context(self, results: list[Any], state: CognitiveState) -> list[Any]:
        """Re-rank resultados según contexto.

        Args:
            results: Resultados de búsqueda.
            state: Estado cognitivo.

        Returns:
            Lista re-rankeada.
        """
        # En una implementación real, esto usaría un reranker
        # Por ahora, devolvemos los resultados sin cambios
        return results


class ProactiveGuidance:
    """Guía proactiva basada en objetivos y estado mental.

    Args:
        tracker: CognitiveStateTracker.
    """

    def __init__(self, tracker: CognitiveStateTracker) -> None:
        self.tracker = tracker
        self._suggestions: list[ContextualSuggestion] = []

    def generate_suggestions(
        self,
        search_func: Any,
        limit: int = 5,
    ) -> list[ContextualSuggestion]:
        """Genera sugerencias proactivas basadas en estado mental.

        Args:
            search_func: Función de búsqueda.
            limit: Máximo de sugerencias.

        Returns:
            Lista de ContextualSuggestion.
        """
        state = self.tracker.get_current_state()

        if not state or not state.focus_area:
            return []

        # Buscar memorias relacionadas al área de enfoque
        results = search_func(state.focus_area, limit=limit * 2)

        suggestions = []
        for r in results[:limit]:
            import uuid

            suggestion = ContextualSuggestion(
                suggestion_id=str(uuid.uuid4()),
                memoria_id=r.id if hasattr(r, "id") else str(r),
                relevance_reason=f"Related to your focus on {state.focus_area}",
                confidence=0.7,
                suggested_at=datetime.now(UTC).isoformat(),
            )
            suggestions.append(suggestion)

        self._suggestions.extend(suggestions)
        return suggestions

    def get_recent_suggestions(self, limit: int = 10) -> list[ContextualSuggestion]:
        """Obtiene sugerencias recientes.

        Args:
            limit: Máximo de resultados.

        Returns:
            Lista de ContextualSuggestion.
        """
        return self._suggestions[-limit:]


class CognitiveManager:
    """Gestiona funcionalidad cognitiva.

    Args:
        tracker: CognitiveStateTracker.
        retrieval: ContextAwareRetrieval.
        guidance: ProactiveGuidance.
    """

    def __init__(
        self,
        tracker: CognitiveStateTracker,
        retrieval: ContextAwareRetrieval,
        guidance: ProactiveGuidance,
    ) -> None:
        self.tracker = tracker
        self.retrieval = retrieval
        self.guidance = guidance

    def update_mental_state(
        self,
        mental_state: str,
        context_type: str,
        current_goal: str | None = None,
        focus_area: str | None = None,
        energy_level: int = 50,
        stress_level: int = 30,
    ) -> CognitiveState:
        """Actualiza el estado mental del usuario.

        Args:
            mental_state: Estado mental.
            context_type: Tipo de contexto.
            current_goal: Objetivo actual.
            focus_area: Área de enfoque.
            energy_level: Nivel de energía.
            stress_level: Nivel de stress.

        Returns:
            CognitiveState actualizado.
        """
        return self.tracker.update_state(
            mental_state=mental_state,
            context_type=context_type,
            current_goal=current_goal,
            focus_area=focus_area,
            energy_level=energy_level,
            stress_level=stress_level,
        )

    def get_mental_state(self) -> CognitiveState | None:
        """Obtiene el estado mental actual.

        Returns:
            CognitiveState actual o None.
        """
        return self.tracker.get_current_state()

    def retrieve_aware(
        self,
        query: str,
        search_func: Any,
        limit: int = 10,
    ) -> list[Any]:
        """Recupera información con conciencia del estado mental.

        Args:
            query: Query de búsqueda.
            search_func: Función de búsqueda.
            limit: Máximo de resultados.

        Returns:
            Lista de resultados adaptados al contexto.
        """
        return self.retrieval.retrieve_with_context(query, search_func, limit)

    def get_proactive_suggestions(
        self,
        search_func: Any,
        limit: int = 5,
    ) -> list[ContextualSuggestion]:
        """Obtiene sugerencias proactivas.

        Args:
            search_func: Función de búsqueda.
            limit: Máximo de sugerencias.

        Returns:
            Lista de ContextualSuggestion.
        """
        return self.guidance.generate_suggestions(search_func, limit)


__all__ = [
    "CognitiveManager",
    "CognitiveState",
    "CognitiveStateTracker",
    "ContextAwareRetrieval",
    "ContextType",
    "ContextualSuggestion",
    "MentalState",
    "ProactiveGuidance",
]

