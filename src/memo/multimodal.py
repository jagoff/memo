"""Memoria Multi-Modal con Embeddings Universales.

Captura información que no es texto (diagramas, capturas, audio) y permite
búsqueda semántica cruzada entre modalidades.

## Gamechanger

- Unifica texto, imágenes, audio y video en un solo espacio semántico
- Busca en un modalidad, encuentra en otras (ej: busca "gráfico de arquitectura" → encuentra imágenes)
- Embeddings universales que entienden contenido visual/auditivo
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class Modality(Enum):
    """Tipos de modalidad de contenido."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MultiModalContent:
    """Contenido multi-modal con embedding."""
    id: str
    memoria_id: str | None  # Si está asociado a una memoria
    modality: str
    content: bytes  # Datos binarios (imagen, audio, video)
    embedding: list[float]  # Embedding universal
    metadata: dict[str, Any]
    created_at: str


@dataclass
class CrossModalResult:
    """Resultado de búsqueda cross-modal."""
    content_id: str
    modality: str
    similarity: float
    metadata: dict[str, Any]


class MultiModalStore:
    """Almacena y busca contenido multi-modal.

    Args:
        state_dir: Directorio para almacenar embeddings.
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.content_file = state_dir / "multimodal_content.json"
        self.embeddings_file = state_dir / "multimodal_embeddings.json"
        self._contents: dict[str, MultiModalContent] = {}
        self._load()

    def _load(self) -> None:
        """Carga contenido desde disco."""
        if self.content_file.is_file():
            try:
                data = json.loads(self.content_file.read_text(encoding="utf-8"))
                for cid, cdata in data.items():
                    # content está en base64
                    content_bytes = base64.b64decode(cdata["content"])
                    self._contents[cid] = MultiModalContent(
                        id=cid,
                        memoria_id=cdata["memoria_id"],
                        modality=cdata["modality"],
                        content=content_bytes,
                        embedding=cdata["embedding"],
                        metadata=cdata["metadata"],
                        created_at=cdata["created_at"],
                    )
            except Exception:
                self._contents = {}

    def _save(self) -> None:
        """Guarda contenido a disco."""
        try:
            data = {}
            for cid, content in self._contents.items():
                data[cid] = {
                    "id": content.id,
                    "memoria_id": content.memoria_id,
                    "modality": content.modality,
                    "content": base64.b64encode(content.content).decode("utf-8"),
                    "embedding": content.embedding,
                    "metadata": content.metadata,
                    "created_at": content.created_at,
                }
            self.content_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def add_content(
        self,
        content: bytes,
        modality: str,
        embedding: list[float],
        memoria_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MultiModalContent:
        """Agrega contenido multi-modal.

        Args:
            content: Datos binarios del contenido.
            modality: Tipo de modalidad.
            embedding: Embedding universal.
            memoria_id: ID de memoria asociada (opcional).
            metadata: Metadatos adicionales.

        Returns:
            MultiModalContent agregado.
        """
        import uuid

        cid = str(uuid.uuid4())
        mmc = MultiModalContent(
            id=cid,
            memoria_id=memoria_id,
            modality=modality,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            created_at=datetime.now(UTC).isoformat(),
        )
        self._contents[cid] = mmc
        self._save()
        return mmc

    def get_content(self, content_id: str) -> MultiModalContent | None:
        """Obtiene contenido por ID.

        Args:
            content_id: ID del contenido.

        Returns:
            MultiModalContent o None.
        """
        return self._contents.get(content_id)

    def search_by_embedding(
        self,
        query_embedding: list[float],
        modality_filter: str | None = None,
        limit: int = 10,
    ) -> list[CrossModalResult]:
        """Busca por similitud de embedding.

        Args:
            query_embedding: Embedding de query.
            modality_filter: Filtrar por modalidad (opcional).
            limit: Máximo de resultados.

        Returns:
            Lista de CrossModalResult.
        """
        results = []

        for cid, content in self._contents.items():
            if modality_filter and content.modality != modality_filter:
                continue

            # Calcular similitud coseno
            sim = self._cosine_similarity(query_embedding, content.embedding)

            results.append(
                CrossModalResult(
                    content_id=cid,
                    modality=content.modality,
                    similarity=sim,
                    metadata=content.metadata,
                )
            )

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:limit]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calcula similitud coseno entre dos embeddings."""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot / (mag_a * mag_b)

    def list_by_memoria(self, memoria_id: str) -> list[MultiModalContent]:
        """Lista todo el contenido asociado a una memoria.

        Args:
            memoria_id: ID de la memoria.

        Returns:
            Lista de MultiModalContent.
        """
        return [c for c in self._contents.values() if c.memoria_id == memoria_id]

    def delete_content(self, content_id: str) -> bool:
        """Elimina contenido.

        Args:
            content_id: ID del contenido.

        Returns:
            True si eliminado.
        """
        if content_id in self._contents:
            del self._contents[content_id]
            self._save()
            return True
        return False


class UniversalEmbedder:
    """Genera embeddings universales multi-modales.

    Para una implementación real, esto usaría CLIP (OpenAI) o
    modelos similares que generan embeddings compartidos entre
    texto, imágenes, audio y video.

    Args:
        model_name: Nombre del modelo a usar.
    """

    def __init__(self, model_name: str = "clip-base") -> None:
        self.model_name = model_name
        # En una implementación real, cargaríamos el modelo aquí

    def embed_text(self, text: str) -> list[float]:
        """Genera embedding para texto.

        Args:
            text: Texto a embeddear.

        Returns:
            Embedding vector.
        """
        # Placeholder: en implementación real usaríamos CLIP o similar
        # Por ahora, generamos un embedding determinista basado en hash
        import hashlib

        hash_val = hashlib.md5(text.encode()).hexdigest()
        # Convertir a vector de floats
        vector = [float(int(h, 16) % 256) / 256.0 for h in hash_val[:512]]
        # Normalizar
        mag = sum(x * x for x in vector) ** 0.5
        if mag > 0:
            vector = [x / mag for x in vector]
        return vector

    def embed_image(self, image_bytes: bytes) -> list[float]:
        """Genera embedding para imagen.

        Args:
            image_bytes: Bytes de la imagen.

        Returns:
            Embedding vector.
        """
        # Placeholder: en implementación real usaríamos CLIP vision encoder
        # Por ahora, generamos embedding basado en hash del contenido
        import hashlib

        hash_val = hashlib.md5(image_bytes).hexdigest()
        vector = [float(int(h, 16) % 256) / 256.0 for h in hash_val[:512]]
        mag = sum(x * x for x in vector) ** 0.5
        if mag > 0:
            vector = [x / mag for x in vector]
        return vector

    def embed_audio(self, audio_bytes: bytes) -> list[float]:
        """Genera embedding para audio.

        Args:
            audio_bytes: Bytes del audio.

        Returns:
            Embedding vector.
        """
        # Placeholder: en implementación real usaríamos modelo de audio
        import hashlib

        hash_val = hashlib.md5(audio_bytes).hexdigest()
        vector = [float(int(h, 16) % 256) / 256.0 for h in hash_val[:512]]
        mag = sum(x * x for x in vector) ** 0.5
        if mag > 0:
            vector = [x / mag for x in vector]
        return vector


class CrossModalSearch:
    """Búsqueda cross-modal entre diferentes modalidades.

    Args:
        store: MultiModalStore.
        embedder: UniversalEmbedder.
    """

    def __init__(self, store: MultiModalStore, embedder: UniversalEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def search_text_find_images(
        self,
        query: str,
        limit: int = 10,
    ) -> list[CrossModalResult]:
        """Busca con texto, encuentra imágenes.

        Args:
            query: Query de texto.
            limit: Máximo de resultados.

        Returns:
            Lista de CrossModalResult de imágenes.
        """
        query_embedding = self.embedder.embed_text(query)
        return self.store.search_by_embedding(
            query_embedding,
            modality_filter=Modality.IMAGE.value,
            limit=limit,
        )

    def search_text_find_audio(
        self,
        query: str,
        limit: int = 10,
    ) -> list[CrossModalResult]:
        """Busca con texto, encuentra audio.

        Args:
            query: Query de texto.
            limit: Máximo de resultados.

        Returns:
            Lista de CrossModalResult de audio.
        """
        query_embedding = self.embedder.embed_text(query)
        return self.store.search_by_embedding(
            query_embedding,
            modality_filter=Modality.AUDIO.value,
            limit=limit,
        )

    def search_image_find_text(
        self,
        image_bytes: bytes,
        limit: int = 10,
    ) -> list[CrossModalResult]:
        """Busca con imagen, encuentra texto.

        Args:
            image_bytes: Bytes de la imagen query.
            limit: Máximo de resultados.

        Returns:
            Lista de CrossModalResult de texto.
        """
        query_embedding = self.embedder.embed_image(image_bytes)
        return self.store.search_by_embedding(
            query_embedding,
            modality_filter=Modality.TEXT.value,
            limit=limit,
        )

    def search_all_modalities(
        self,
        query: str,
        limit: int = 10,
    ) -> dict[str, list[CrossModalResult]]:
        """Busca en todas las modalidades.

        Args:
            query: Query de texto.
            limit: Máximo de resultados por modalidad.

        Returns:
            Dict con resultados por modalidad.
        """
        query_embedding = self.embedder.embed_text(query)

        results = {}
        for modality in [Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO]:
            mod_results = self.store.search_by_embedding(
                query_embedding,
                modality_filter=modality.value,
                limit=limit,
            )
            if mod_results:
                results[modality.value] = mod_results

        return results


class MultiModalManager:
    """Gestiona funcionalidad multi-modal.

    Args:
        store: MultiModalStore.
        embedder: UniversalEmbedder.
        search: CrossModalSearch.
    """

    def __init__(
        self,
        store: MultiModalStore,
        embedder: UniversalEmbedder,
        search: CrossModalSearch,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.search = search

    def add_image(
        self,
        image_path: Path,
        memoria_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MultiModalContent:
        """Agrega imagen al corpus multi-modal.

        Args:
            image_path: Path a la imagen.
            memoria_id: ID de memoria asociada.
            metadata: Metadatos adicionales.

        Returns:
            MultiModalContent agregado.
        """
        image_bytes = image_path.read_bytes()
        embedding = self.embedder.embed_image(image_bytes)

        return self.store.add_content(
            content=image_bytes,
            modality=Modality.IMAGE.value,
            embedding=embedding,
            memoria_id=memoria_id,
            metadata=metadata or {"filename": image_path.name},
        )

    def add_audio(
        self,
        audio_path: Path,
        memoria_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MultiModalContent:
        """Agrega audio al corpus multi-modal.

        Args:
            audio_path: Path al audio.
            memoria_id: ID de memoria asociada.
            metadata: Metadatos adicionales.

        Returns:
            MultiModalContent agregado.
        """
        audio_bytes = audio_path.read_bytes()
        embedding = self.embedder.embed_audio(audio_bytes)

        return self.store.add_content(
            content=audio_bytes,
            modality=Modality.AUDIO.value,
            embedding=embedding,
            memoria_id=memoria_id,
            metadata=metadata or {"filename": audio_path.name},
        )


__all__ = [
    "MultiModalStore",
    "UniversalEmbedder",
    "CrossModalSearch",
    "MultiModalManager",
    "MultiModalContent",
    "CrossModalResult",
    "Modality",
]

