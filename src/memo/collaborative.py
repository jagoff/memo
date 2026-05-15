"""Memoria Social Colaborativa con Grafo de Conocimiento Compartido.

Una red neural de memorias compartidas entre múltiples usuarios que aprende
de las conexiones de todos.

## Gamechanger

- El conocimiento crece colectivamente, no solo individualmente
- Conexiones que otros usuarios descubren te benefician a ti
- Grafo de conocimiento compartido que evoluciona
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class SharedConnection:
    """Una conexión descubierta por otro usuario."""
    connection_id: str
    from_user: str
    entity_a: str
    entity_b: str
    relationship: str
    confidence: float
    discovered_at: str
    votes: int


@dataclass
class CollectiveInsight:
    """Un insight generado colectivamente."""
    insight_id: str
    content: str
    contributors: list[str]
    upvotes: int
    downvotes: int
    created_at: str


@dataclass
class UserProfile:
    """Perfil de usuario en la red colaborativa."""
    user_id: str
    username: str
    reputation: int
    contributions_count: int
    joined_at: str


class CollaborativeGraph:
    """Grafo de conocimiento compartido entre usuarios.

    Args:
        state_dir: Directorio para almacenar el grafo.
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.graph_file = state_dir / "collaborative_graph.json"
        self._connections: dict[str, SharedConnection] = {}
        self._insights: dict[str, CollectiveInsight] = {}
        self._users: dict[str, UserProfile] = {}
        self._load()

    def _load(self) -> None:
        """Carga el grafo desde disco."""
        if self.graph_file.is_file():
            try:
                data = json.loads(self.graph_file.read_text(encoding="utf-8"))

                # Load connections
                for cid, cdata in data.get("connections", {}).items():
                    self._connections[cid] = SharedConnection(**cdata)

                # Load insights
                for iid, idata in data.get("insights", {}).items():
                    self._insights[iid] = CollectiveInsight(**idata)

                # Load users
                for uid, udata in data.get("users", {}).items():
                    self._users[uid] = UserProfile(**udata)
            except Exception:
                self._connections = {}
                self._insights = {}
                self._users = {}

    def _save(self) -> None:
        """Guarda el grafo a disco."""
        try:
            data = {
                "connections": {cid: c.__dict__ for cid, c in self._connections.items()},
                "insights": {iid: i.__dict__ for iid, i in self._insights.items()},
                "users": {uid: u.__dict__ for uid, u in self._users.items()},
            }
            self.graph_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def add_connection(
        self,
        from_user: str,
        entity_a: str,
        entity_b: str,
        relationship: str,
        confidence: float = 0.7,
    ) -> SharedConnection:
        """Agrega una conexión descubierta por un usuario.

        Args:
            from_user: Usuario que descubrió la conexión.
            entity_a: Primera entidad.
            entity_b: Segunda entidad.
            relationship: Tipo de relación.
            confidence: Confianza en la conexión.

        Returns:
            SharedConnection agregado.
        """
        import uuid

        cid = str(uuid.uuid4())
        conn = SharedConnection(
            connection_id=cid,
            from_user=from_user,
            entity_a=entity_a,
            entity_b=entity_b,
            relationship=relationship,
            confidence=confidence,
            discovered_at=datetime.now(UTC).isoformat(),
            votes=0,
        )
        self._connections[cid] = conn
        self._save()

        # Update user reputation
        if from_user not in self._users:
            self._users[from_user] = UserProfile(
                user_id=from_user,
                username=from_user,
                reputation=10,
                contributions_count=1,
                joined_at=datetime.now(UTC).isoformat(),
            )
        else:
            self._users[from_user].contributions_count += 1
            self._users[from_user].reputation += 5

        self._save()
        return conn

    def vote_connection(self, connection_id: str, upvote: bool = True) -> bool:
        """Vota por una conexión (upvote/downvote).

        Args:
            connection_id: ID de la conexión.
            upvote: True para upvote, False para downvote.

        Returns:
            True si exitoso.
        """
        if connection_id not in self._connections:
            return False

        conn = self._connections[connection_id]
        if upvote:
            conn.votes += 1
            # Reward the discoverer
            if conn.from_user in self._users:
                self._users[conn.from_user].reputation += 2
        else:
            conn.votes -= 1

        self._save()
        return True

    def get_connections_for_entity(self, entity: str) -> list[SharedConnection]:
        """Obtiene todas las conexiones para una entidad.

        Args:
            entity: Nombre de la entidad.

        Returns:
            Lista de SharedConnection.
        """
        return [
            c for c in self._connections.values()
            if c.entity_a == entity or c.entity_b == entity
        ]

    def add_insight(
        self,
        user_id: str,
        content: str,
    ) -> CollectiveInsight:
        """Agrega un insight colectivo.

        Args:
            user_id: ID del usuario.
            content: Contenido del insight.

        Returns:
            CollectiveInsight agregado.
        """
        import uuid

        iid = str(uuid.uuid4())
        insight = CollectiveInsight(
            insight_id=iid,
            content=content,
            contributors=[user_id],
            upvotes=0,
            downvotes=0,
            created_at=datetime.now(UTC).isoformat(),
        )
        self._insights[iid] = insight
        self._save()
        return insight

    def vote_insight(self, insight_id: str, upvote: bool = True) -> bool:
        """Vota por un insight.

        Args:
            insight_id: ID del insight.
            upvote: True para upvote, False para downvote.

        Returns:
            True si exitoso.
        """
        if insight_id not in self._insights:
            return False

        insight = self._insights[insight_id]
        if upvote:
            insight.upvotes += 1
        else:
            insight.downvotes += 1

        self._save()
        return True

    def get_top_insights(self, limit: int = 10) -> list[CollectiveInsight]:
        """Obtiene los insights más votados.

        Args:
            limit: Máximo de resultados.

        Returns:
            Lista de CollectiveInsight ordenados por upvotes.
        """
        sorted_insights = sorted(
            self._insights.values(),
            key=lambda i: i.upvotes - i.downvotes,
            reverse=True,
        )
        return sorted_insights[:limit]

    def get_user_profile(self, user_id: str) -> UserProfile | None:
        """Obtiene el perfil de un usuario.

        Args:
            user_id: ID del usuario.

        Returns:
            UserProfile o None.
        """
        return self._users.get(user_id)


class CollaborativeFilter:
    """Recomendaciones basadas en patrones colectivos.

    Args:
        graph: CollaborativeGraph.
    """

    def __init__(self, graph: CollaborativeGraph) -> None:
        self.graph = graph

    def recommend_connections(
        self,
        entity: str,
        limit: int = 10,
    ) -> list[SharedConnection]:
        """Recomienda conexiones basadas en patrones colectivos.

        Args:
            entity: Entidad de interés.
            limit: Máximo de resultados.

        Returns:
            Lista de SharedConnection recomendados.
        """
        # Obtener conexiones existentes para la entidad
        existing = self.graph.get_connections_for_entity(entity)

        # Buscar conexiones similares de otros usuarios
        recommendations = []
        for conn in self.graph._connections.values():
            if conn.entity_a == entity or conn.entity_b == entity:
                continue

            # Si la conexión comparte una entidad con las existentes, recomendarla
            for existing_conn in existing:
                if (conn.entity_a == existing_conn.entity_b or
                    conn.entity_b == existing_conn.entity_a):
                    recommendations.append(conn)

        # Ordenar por votos y confianza
        recommendations.sort(
            key=lambda c: (c.votes + c.confidence * 10),
            reverse=True,
        )

        return recommendations[:limit]

    def recommend_insights(
        self,
        user_interests: list[str],
        limit: int = 10,
    ) -> list[CollectiveInsight]:
        """Recomienda insights basados en intereses del usuario.

        Args:
            user_interests: Lista de intereses del usuario.
            limit: Máximo de resultados.

        Returns:
            Lista de CollectiveInsight recomendados.
        """
        # Buscar insights que mencionen los intereses
        recommendations = []
        for insight in self.graph._insights.values():
            for interest in user_interests:
                if interest.lower() in insight.content.lower():
                    recommendations.append(insight)
                    break

        # Ordenar por upvotes
        recommendations.sort(key=lambda i: i.upvotes - i.downvotes, reverse=True)

        return recommendations[:limit]


class CollaborativeManager:
    """Gestiona funcionalidad colaborativa.

    Args:
        graph: CollaborativeGraph.
        filter: CollaborativeFilter.
    """

    def __init__(
        self,
        graph: CollaborativeGraph,
        filter: CollaborativeFilter,
    ) -> None:
        self.graph = graph
        self.filter = filter

    def share_connection(
        self,
        user_id: str,
        entity_a: str,
        entity_b: str,
        relationship: str,
        confidence: float = 0.7,
    ) -> SharedConnection:
        """Comparte una conexión descubierta con la comunidad.

        Args:
            user_id: ID del usuario.
            entity_a: Primera entidad.
            entity_b: Segunda entidad.
            relationship: Tipo de relación.
            confidence: Confianza.

        Returns:
            SharedConnection compartido.
        """
        return self.graph.add_connection(
            from_user=user_id,
            entity_a=entity_a,
            entity_b=entity_b,
            relationship=relationship,
            confidence=confidence,
        )

    def get_shared_connections(self, entity: str) -> list[SharedConnection]:
        """Obtiene conexiones compartidas para una entidad.

        Args:
            entity: Entidad de interés.

        Returns:
            Lista de SharedConnection.
        """
        return self.graph.get_connections_for_entity(entity)

    def get_recommended_connections(self, entity: str, limit: int = 10) -> list[SharedConnection]:
        """Obtiene conexiones recomendadas.

        Args:
            entity: Entidad de interés.
            limit: Máximo de resultados.

        Returns:
            Lista de SharedConnection recomendados.
        """
        return self.filter.recommend_connections(entity, limit)

    def share_insight(self, user_id: str, content: str) -> CollectiveInsight:
        """Comparte un insight con la comunidad.

        Args:
            user_id: ID del usuario.
            content: Contenido del insight.

        Returns:
            CollectiveInsight compartido.
        """
        return self.graph.add_insight(user_id, content)

    def get_top_insights(self, limit: int = 10) -> list[CollectiveInsight]:
        """Obtiene los insights más votados.

        Args:
            limit: Máximo de resultados.

        Returns:
            Lista de CollectiveInsight.
        """
        return self.graph.get_top_insights(limit)


__all__ = [
    "CollaborativeFilter",
    "CollaborativeGraph",
    "CollaborativeManager",
    "CollectiveInsight",
    "SharedConnection",
    "UserProfile",
]

