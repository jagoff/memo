"""Tests para multimodal module (EXPERIMENTAL)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from memo.multimodal import (
    CrossModalResult,
    Modality,
    MultiModalContent,
    MultiModalStore,
)


@pytest.fixture
def temp_state_dir() -> Path:
    """Temp directory para MultiModalStore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestMultiModalStore:
    """Tests para MultiModalStore."""

    def test_init_creates_store(self, temp_state_dir: Path) -> None:
        """MultiModalStore.__init__ crea instancia."""
        store = MultiModalStore(temp_state_dir)
        assert store.state_dir == temp_state_dir
        assert store.content_file == temp_state_dir / "multimodal_content.json"
        assert store.embeddings_file == temp_state_dir / "multimodal_embeddings.json"

    def test_load_empty_store(self, temp_state_dir: Path) -> None:
        """_load() maneja directorio vacío."""
        store = MultiModalStore(temp_state_dir)
        assert store._contents == {}

    def test_load_existing_content(self, temp_state_dir: Path) -> None:
        """_load() carga contenido existente."""
        import base64

        # Pre-crear archivo de contenido
        content_data = {
            "test-id": {
                "memoria_id": "mem-123",
                "modality": "text",
                "content": base64.b64encode(b"test content").decode("utf-8"),
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {"source": "test"},
                "created_at": "2026-05-27T00:00:00Z",
            }
        }
        content_file = temp_state_dir / "multimodal_content.json"
        content_file.write_text(json.dumps(content_data), encoding="utf-8")

        store = MultiModalStore(temp_state_dir)
        assert "test-id" in store._contents
        assert store._contents["test-id"].modality == "text"

    def test_add_content(self, temp_state_dir: Path) -> None:
        """add_content() persiste contenido."""
        store = MultiModalStore(temp_state_dir)
        content = MultiModalContent(
            id="test-1",
            memoria_id="mem-1",
            modality="text",
            content=b"test data",
            embedding=[0.1, 0.2],
            metadata={"key": "value"},
            created_at="2026-05-27T00:00:00Z",
        )
        store.add_content(content)
        assert "test-1" in store._contents

    def test_search_empty_store(self, temp_state_dir: Path) -> None:
        """search() retorna [] en store vacío."""
        store = MultiModalStore(temp_state_dir)
        results = store.search([0.1, 0.2], limit=5)
        assert results == []

    def test_search_returns_cross_modal_results(self, temp_state_dir: Path) -> None:
        """search() retorna CrossModalResult objects."""
        store = MultiModalStore(temp_state_dir)
        content = MultiModalContent(
            id="test-1",
            memoria_id="mem-1",
            modality="text",
            content=b"test",
            embedding=[0.1, 0.2],
            metadata={},
            created_at="2026-05-27T00:00:00Z",
        )
        store.add_content(content)

        results = store.search([0.1, 0.2], limit=5)
        assert len(results) > 0
        assert isinstance(results[0], CrossModalResult)

    def test_get_by_id(self, temp_state_dir: Path) -> None:
        """get_by_id() retorna contenido."""
        store = MultiModalStore(temp_state_dir)
        content = MultiModalContent(
            id="test-1",
            memoria_id="mem-1",
            modality="text",
            content=b"test",
            embedding=[0.1, 0.2],
            metadata={},
            created_at="2026-05-27T00:00:00Z",
        )
        store.add_content(content)

        retrieved = store.get_by_id("test-1")
        assert retrieved is not None
        assert retrieved.id == "test-1"

    def test_get_by_id_missing(self, temp_state_dir: Path) -> None:
        """get_by_id() retorna None si no existe."""
        store = MultiModalStore(temp_state_dir)
        assert store.get_by_id("missing") is None

    def test_delete_content(self, temp_state_dir: Path) -> None:
        """delete_content() remueve contenido."""
        store = MultiModalStore(temp_state_dir)
        content = MultiModalContent(
            id="test-1",
            memoria_id="mem-1",
            modality="text",
            content=b"test",
            embedding=[0.1, 0.2],
            metadata={},
            created_at="2026-05-27T00:00:00Z",
        )
        store.add_content(content)
        assert "test-1" in store._contents

        store.delete_content("test-1")
        assert "test-1" not in store._contents

    def test_modality_enum(self) -> None:
        """Modality enum tiene valores correctos."""
        assert Modality.TEXT.value == "text"
        assert Modality.IMAGE.value == "image"
        assert Modality.AUDIO.value == "audio"
        assert Modality.VIDEO.value == "video"
