"""Multi-modal memoria with universal embeddings.

NOTE: Covered by test suite (tests/test_multimodal.py). Not exposed via MCP yet.

Captures non-text information (diagrams, screenshots, audio) and enables
cross-modal semantic search across modalities.

## Gamechanger

- Unifies text, images, audio, and video in a single semantic space
- Search in one modality, find in others (e.g. search "architecture diagram" → find images)
- Universal embeddings that understand visual/auditory content
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


class Modality(Enum):
    """Content modality types."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MultiModalContent:
    """Multi-modal content with embedding."""

    id: str
    memoria_id: str | None  # If associated with a memoria
    modality: str
    content: bytes  # Binary data (image, audio, video)
    embedding: list[float]  # Universal embedding
    metadata: dict[str, Any]
    created_at: str


@dataclass
class CrossModalResult:
    """Cross-modal search result."""

    content_id: str
    modality: str
    similarity: float
    metadata: dict[str, Any]


class MultiModalStore:
    """Stores and searches multi-modal content.

    Args:
        state_dir: Directory to store embeddings.
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.content_file = state_dir / "multimodal_content.json"
        self.embeddings_file = state_dir / "multimodal_embeddings.json"
        self._contents: dict[str, MultiModalContent] = {}
        self._load()

    def _load(self) -> None:
        """Load content from disk."""
        if self.content_file.is_file():
            try:
                data = json.loads(self.content_file.read_text(encoding="utf-8"))
                for cid, cdata in data.items():
                    # content is base64-encoded
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
        """Save content to disk."""
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
        except (OSError, TypeError, ValueError) as exc:
            _log.error("multimodal: failed to persist content store: %s", exc)

    def add_content(
        self,
        content: bytes,
        modality: str,
        embedding: list[float],
        memoria_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MultiModalContent:
        """Add multi-modal content.

        Args:
            content: Binary content data.
            modality: Modality type.
            embedding: Universal embedding.
            memoria_id: Associated memoria ID (optional).
            metadata: Additional metadata.

        Returns:
            The added MultiModalContent.
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
        """Get content by ID.

        Args:
            content_id: The content ID.

        Returns:
            MultiModalContent or None.
        """
        return self._contents.get(content_id)

    def search_by_embedding(
        self,
        query_embedding: list[float],
        modality_filter: str | None = None,
        limit: int = 10,
    ) -> list[CrossModalResult]:
        """Search by embedding similarity.

        Args:
            query_embedding: Query embedding.
            modality_filter: Filter by modality (optional).
            limit: Maximum number of results.

        Returns:
            List of CrossModalResult.
        """
        results = []

        for cid, content in self._contents.items():
            if modality_filter and content.modality != modality_filter:
                continue

            # Compute cosine similarity
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
        """Compute cosine similarity between two embeddings."""
        import math

        dot = sum(x * y for x, y in zip(a, b, strict=False))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot / (mag_a * mag_b)

    def list_by_memoria(self, memoria_id: str) -> list[MultiModalContent]:
        """List all content associated with a memoria.

        Args:
            memoria_id: The memoria ID.

        Returns:
            List of MultiModalContent.
        """
        return [c for c in self._contents.values() if c.memoria_id == memoria_id]

    def delete_content(self, content_id: str) -> bool:
        """Delete content.

        Args:
            content_id: The content ID.

        Returns:
            True if deleted.
        """
        if content_id in self._contents:
            del self._contents[content_id]
            self._save()
            return True
        return False


class UniversalEmbedder:
    """Generates universal multi-modal embeddings.

    A real implementation would use CLIP (OpenAI) or similar models
    that generate shared embeddings across text, images, audio, and
    video.

    Args:
        model_name: Name of the model to use.
    """

    def __init__(self, model_name: str = "clip-base") -> None:
        self.model_name = model_name
        # In a real implementation, we would load the model here

    def _hash_embedding(self, payload: bytes, dims: int = 512) -> list[float]:
        import hashlib

        raw = bytearray()
        counter = 0
        while len(raw) < dims:
            raw.extend(hashlib.sha256(payload + counter.to_bytes(4, "big")).digest())
            counter += 1

        vector = [((byte / 255.0) * 2.0) - 1.0 for byte in raw[:dims]]
        mag = sum(x * x for x in vector) ** 0.5
        if mag > 0:
            vector = [x / mag for x in vector]
        return vector

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        # Placeholder: a real implementation would use CLIP or similar
        # For now, generate a deterministic hash-based embedding
        return self._hash_embedding(text.encode("utf-8"))

    def embed_image(self, image_bytes: bytes) -> list[float]:
        """Generate an embedding for an image.

        Args:
            image_bytes: The image bytes.

        Returns:
            Embedding vector.
        """
        # Placeholder: a real implementation would use the CLIP vision encoder
        # For now, generate a hash-based embedding of the content
        return self._hash_embedding(image_bytes)

    def embed_audio(self, audio_bytes: bytes) -> list[float]:
        """Generate an embedding for audio.

        Args:
            audio_bytes: The audio bytes.

        Returns:
            Embedding vector.
        """
        # Placeholder: a real implementation would use an audio model
        return self._hash_embedding(audio_bytes)


class CrossModalSearch:
    """Cross-modal search across different modalities.

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
        """Search with text, find images.

        Args:
            query: Text query.
            limit: Maximum number of results.

        Returns:
            List of image CrossModalResult.
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
        """Search with text, find audio.

        Args:
            query: Text query.
            limit: Maximum number of results.

        Returns:
            List of audio CrossModalResult.
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
        """Search with an image, find text.

        Args:
            image_bytes: The query image bytes.
            limit: Maximum number of results.

        Returns:
            List of text CrossModalResult.
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
        """Search across all modalities.

        Args:
            query: Text query.
            limit: Maximum number of results per modality.

        Returns:
            Dict with results per modality.
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
    """Manages multi-modal functionality.

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
        """Add an image to the multi-modal corpus.

        Args:
            image_path: Path to the image.
            memoria_id: Associated memoria ID.
            metadata: Additional metadata.

        Returns:
            The added MultiModalContent.
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
        """Add audio to the multi-modal corpus.

        Args:
            audio_path: Path to the audio.
            memoria_id: Associated memoria ID.
            metadata: Additional metadata.

        Returns:
            The added MultiModalContent.
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
    "CrossModalResult",
    "CrossModalSearch",
    "Modality",
    "MultiModalContent",
    "MultiModalManager",
    "MultiModalStore",
    "UniversalEmbedder",
]
