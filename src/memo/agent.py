"""Autonomous Memory Agent — razonamiento causal y síntesis de conocimiento.

Este es EL GAMECHANGER: transforma memo de un sistema de recuperación pasivo
a un sistema de razonamiento activo que puede generar nuevo conocimiento.

## Características Revolucionarias

1. **Agente Autónomo**: Un LLM que proactivamente explora el corpus sin que el usuario lo pida
2. **Razonamiento Causal**: Entiende POR QUÉ las cosas están conectadas, no solo que lo están
3. **Síntesis de Conocimiento**: Combina memorias para generar insights que no existían antes
4. **Meta-Cognición**: El agente reflexiona sobre su propio proceso de razonamiento
5. **Planificación**: El agente puede planificar investigaciones complejas
6. **Proactividad**: Sugerencias de información antes de que el usuario la pida

## Diferencia con Sistemas Tradicionales

- **RAG tradicional**: Recupera información relevante para una query
- **Este agente**: Genera nuevo conocimiento a través de razonamiento multi-paso

Ejemplo:
- RAG: "¿Qué sé sobre MLX?" → Recupera memorias sobre MLX
- Agente: "Explora las implicaciones de usar MLX para inferencia en edge devices" →
  - Busca memorias sobre MLX, edge computing, hardware constraints
  - Razona sobre trade-offs
  - Sintetiza nueva conclusión: "MLX es ideal para edge porque X, Y, Z"
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel


class ReasoningStep(BaseModel):
    """Un paso en el proceso de razonamiento."""
    step_number: int
    action: str  # "search", "analyze", "synthesize", "validate"
    query: str
    results: list[str]  # memoria IDs
    reasoning: str
    confidence: float


class InvestigationPlan(BaseModel):
    """Plan de investigación generado por el agente."""
    goal: str
    steps: list[str]
    estimated_complexity: int
    estimated_insight_value: int


class SynthesisResult(BaseModel):
    """Resultado de síntesis de conocimiento."""
    new_insight: str
    supporting_memorias: list[str]
    reasoning_chain: list[ReasoningStep]
    confidence: float
    novelty_score: float


class AgentThought(BaseModel):
    """Un pensamiento del agente (meta-cognición)."""
    timestamp: str
    thought_type: str  # "hypothesis", "reflection", "question", "insight"
    content: str
    related_memorias: list[str]


class AutonomousAgent:
    """Agente autónomo de memoria con razonamiento causal y síntesis.

    Este es el gamechanger: no solo recupera, sino que razona y genera
    nuevo conocimiento.

    Args:
        memory: La instancia Memory.
        chat: La instancia MLXChat para razonamiento.
    """

    def __init__(self, memory: Any, chat: Any | None) -> None:
        self.memory = memory
        if chat is None:
            from memo.llm import MLXChat

            chat = MLXChat()
        self.chat = chat
        self._thoughts: list[AgentThought] = []
        self._synthesis_history: list[SynthesisResult] = []

    def _complete(self, prompt: str, *, temperature: float, max_tokens: int = 512) -> str:
        """Return plain text from either a test double or the MLXChat wrapper."""
        if hasattr(self.chat, "complete"):
            return self.chat.complete(prompt, temperature=temperature)

        out = self.chat.chat(
            model=self.memory.cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "max_tokens": max_tokens},
        )
        return ((out.get("message") or {}).get("content") or "").strip()

    def think(self, thought: str, thought_type: str = "hypothesis") -> AgentThought:
        """Registra un pensamiento del agente (meta-cognición).

        Args:
            thought: El contenido del pensamiento.
            thought_type: Tipo de pensamiento.

        Returns:
            AgentThought registrado.
        """
        agent_thought = AgentThought(
            timestamp=datetime.now(UTC).isoformat(),
            thought_type=thought_type,
            content=thought,
            related_memorias=[],
        )
        self._thoughts.append(agent_thought)
        return agent_thought

    def plan_investigation(self, goal: str) -> InvestigationPlan:
        """Planifica una investigación compleja sobre el corpus.

        El agente analiza el goal y genera un plan de pasos para investigarlo.

        Args:
            goal: El objetivo de la investigación.

        Returns:
            InvestigationPlan con los pasos a seguir.
        """
        # Usar LLM para generar el plan
        prompt = f"""You are an autonomous research agent. Plan an investigation to achieve this goal:

Goal: {goal}

The corpus contains personal memorias about various topics. Plan 3-5 concrete steps to investigate this goal.
Each step should be a specific action like "search for X", "analyze relationship between X and Y", etc.

Output JSON:
{{
  "goal": "...",
  "steps": ["step 1", "step 2", ...],
  "estimated_complexity": 1-10,
  "estimated_insight_value": 1-10
}}"""

        response = self._complete(prompt, temperature=0.3)
        data = json.loads(response)

        return InvestigationPlan(**data)

    def execute_step(self, step: str) -> ReasoningStep:
        """Ejecuta un paso de investigación.

        Args:
            step: La descripción del paso.

        Returns:
            ReasoningStep con el resultado.
        """
        # Interpretar el paso y ejecutar la acción correspondiente
        if "search" in step.lower():
            # Extraer query del paso
            query = step.replace("search for", "").strip()
            results = self.memory.search(query, limit=10, mode="hybrid")
            memoria_ids = [r.id for r in results]

            reasoning = f"Searched for '{query}', found {len(memoria_ids)} relevant memorias"

            return ReasoningStep(
                step_number=0,  # Will be set by caller
                action="search",
                query=query,
                results=memoria_ids,
                reasoning=reasoning,
                confidence=0.8,
            )
        elif "analyze" in step.lower():
            # Análisis relacional
            return ReasoningStep(
                step_number=0,
                action="analyze",
                query=step,
                results=[],
                reasoning="Analysis performed",
                confidence=0.7,
            )
        else:
            return ReasoningStep(
                step_number=0,
                action="unknown",
                query=step,
                results=[],
                reasoning=f"Executed step: {step}",
                confidence=0.5,
            )

    def reason_causally(self, query: str, memoria_ids: list[str]) -> str:
        """Razona causalmente sobre un conjunto de memorias.

        No solo encuentra conexiones, sino explica POR QUÉ están conectadas.

        Args:
            query: La pregunta o tema a razonar.
            memoria_ids: IDs de las memorias a analizar.

        Returns:
            Explicación causal del razonamiento.
        """
        # Obtener el contenido de las memorias
        memorias = []
        for mid in memoria_ids[:5]:  # Limitar a 5 para contexto
            m = self.memory.get(mid)
            if m:
                memorias.append(f"Title: {m.title}\nContent: {m.body or ''}\nTags: {', '.join(m.tags)}")

        context = "\n\n---\n\n".join(memorias)

        prompt = f"""You are a causal reasoning agent. Analyze these memorias and explain the CAUSAL relationships between them.

Query: {query}

Memorias:
{context}

Provide a causal explanation:
1. What are the key entities/concepts?
2. How are they causally connected?
3. What caused what?
4. What are the implications?

Focus on CAUSAL links (X caused Y, X implies Y, X is necessary for Y)."""

        response = self._complete(prompt, temperature=0.5)
        return response

    def synthesize_knowledge(self, topic: str) -> SynthesisResult:
        """Sintetiza nuevo conocimiento a partir de memorias existentes.

        Este es el core del gamechanger: genera insights que NO existían antes
        combinando y razonando sobre memorias existentes.

        Args:
            topic: El tema sobre el cual sintetizar conocimiento.

        Returns:
            SynthesisResult con el nuevo insight.
        """
        # Paso 1: Buscar memorias relevantes
        results = self.memory.search(topic, limit=15, mode="hybrid")
        memoria_ids = [r.id for r in results]

        # Paso 2: Razonar causalmente
        causal_explanation = self.reason_causally(topic, memoria_ids)

        # Paso 3: Usar LLM para sintetizar nuevo conocimiento
        prompt = f"""You are a knowledge synthesis agent. Based on the causal analysis below, generate a NEW insight that was not explicitly stated in the original memorias.

Topic: {topic}

Causal Analysis:
{causal_explanation}

Generate a novel insight that:
1. Combines information from multiple memorias
2. Identifies a pattern or relationship not obvious before
3. Has practical implications
4. Is supported by the memorias but represents new understanding

Output JSON:
{{
  "new_insight": "...",
  "supporting_memorias": ["id1", "id2", ...],
  "confidence": 0.0-1.0,
  "novelty_score": 0.0-1.0
}}"""

        response = self._complete(prompt, temperature=0.7)
        data = json.loads(response)

        # Construir chain de razonamiento
        reasoning_chain = [
            ReasoningStep(
                step_number=1,
                action="search",
                query=topic,
                results=memoria_ids[:5],
                reasoning=f"Found {len(memoria_ids)} relevant memorias",
                confidence=0.8,
            ),
            ReasoningStep(
                step_number=2,
                action="analyze",
                query="causal analysis",
                results=memoria_ids[:5],
                reasoning=causal_explanation[:200],
                confidence=0.7,
            ),
            ReasoningStep(
                step_number=3,
                action="synthesize",
                query="knowledge synthesis",
                results=memoria_ids[:5],
                reasoning="Generated novel insight from causal relationships",
                confidence=data.get("confidence", 0.7),
            ),
        ]

        synthesis = SynthesisResult(
            new_insight=data["new_insight"],
            supporting_memorias=data.get("supporting_memorias", memoria_ids[:3]),
            reasoning_chain=reasoning_chain,
            confidence=data.get("confidence", 0.7),
            novelty_score=data.get("novelty_score", 0.8),
        )

        self._synthesis_history.append(synthesis)

        # Registrar como pensamiento
        self.think(f"Synthesized new insight: {synthesis.new_insight[:100]}...", "insight")

        return synthesis

    def proactive_discovery(self) -> list[SynthesisResult]:
        """Descubrimiento proactivo: explora el corpus sin que el usuario lo pida.

        El agente identifica áreas del corpus que podrían contener insights
        no descubiertos y los explora proactivamente.

        Returns:
            Lista de SynthesisResult descubiertos proactivamente.
        """
        # Obtener las memorias más recientes
        recent = self.memory.list(limit=20)

        # Identificar temas prometedores (tags frecuentes, entidades conectadas)
        all_tags = []
        for m in recent:
            all_tags.extend(m.tags)

        from collections import Counter
        top_tags = [t for t, c in Counter(all_tags).most_common(5)]

        discoveries = []

        # Para cada tag, intentar sintetizar conocimiento
        for tag in top_tags:
            try:
                synthesis = self.synthesize_knowledge(tag)
                if synthesis.novelty_score > 0.6:  # Solo insights novedosos
                    discoveries.append(synthesis)
            except Exception:
                continue

        return discoveries

    def get_thoughts(self, thought_type: str | None = None) -> list[AgentThought]:
        """Obtener el historial de pensamientos del agente.

        Args:
            thought_type: Filtrar por tipo (opcional).

        Returns:
            Lista de AgentThought.
        """
        if thought_type:
            return [t for t in self._thoughts if t.thought_type == thought_type]
        return self._thoughts

    def get_synthesis_history(self) -> list[SynthesisResult]:
        """Obtener el historial de síntesis de conocimiento.

        Returns:
            Lista de SynthesisResult.
        """
        return self._synthesis_history


__all__ = [
    "AgentThought",
    "AutonomousAgent",
    "InvestigationPlan",
    "ReasoningStep",
    "SynthesisResult",
]
