"""Tests para multimodal module (EXPERIMENTAL).

Coverage smoke for `MultiModalStore` against the actual API shape — these
tests existed in a stub form that called methods that don't exist on the
store; rewriting against the real signatures locks down the experimental
contract so future renames break loudly.
"""

from __future__ import annotations

import base64
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
        store = MultiModalStore(temp_state_dir)
        assert store.state_dir == temp_state_dir
        assert store.content_file == temp_state_dir / "multimodal_content.json"
        assert store.embeddings_file == temp_state_dir / "multimodal_embeddings.json"

    def test_load_empty_store(self, temp_state_dir: Path) -> None:
        store = MultiModalStore(temp_state_dir)
        assert store._contents == {}

    def test_load_existing_content(self, temp_state_dir: Path) -> None:
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
        store = MultiModalStore(temp_state_dir)
        mmc = store.add_content(
            content=b"test data",
            modality="text",
            embedding=[0.1, 0.2],
            memoria_id="mem-1",
            metadata={"key": "value"},
        )
        assert isinstance(mmc, MultiModalContent)
        assert mmc.id in store._contents
        assert mmc.modality == "text"
        assert mmc.memoria_id == "mem-1"

    def test_search_empty_store(self, temp_state_dir: Path) -> None:
        store = MultiModalStore(temp_state_dir)
        results = store.search_by_embedding([0.1, 0.2], limit=5)
        assert results == []

    def test_search_returns_cross_modal_results(self, temp_state_dir: Path) -> None:
        store = MultiModalStore(temp_state_dir)
        store.add_content(
            content=b"test",
            modality="text",
            embedding=[0.1, 0.2],
            memoria_id="mem-1",
        )

        results = store.search_by_embedding([0.1, 0.2], limit=5)
        assert len(results) > 0
        assert isinstance(results[0], CrossModalResult)

    def test_get_content(self, temp_state_dir: Path) -> None:
        store = MultiModalStore(temp_state_dir)
        mmc = store.add_content(
            content=b"test",
            modality="text",
            embedding=[0.1, 0.2],
            memoria_id="mem-1",
        )

        retrieved = store.get_content(mmc.id)
        assert retrieved is not None
        assert retrieved.id == mmc.id

    def test_get_content_missing(self, temp_state_dir: Path) -> None:
        store = MultiModalStore(temp_state_dir)
        assert store.get_content("missing") is None

    def test_delete_content(self, temp_state_dir: Path) -> None:
        store = MultiModalStore(temp_state_dir)
        mmc = store.add_content(
            content=b"test",
            modality="text",
            embedding=[0.1, 0.2],
            memoria_id="mem-1",
        )
        assert mmc.id in store._contents

        assert store.delete_content(mmc.id) is True
        assert mmc.id not in store._contents
        # Idempotent: second delete returns False.
        assert store.delete_content(mmc.id) is False

    def test_modality_enum(self) -> None:
        assert Modality.TEXT.value == "text"
        assert Modality.IMAGE.value == "image"
        assert Modality.AUDIO.value == "audio"
        assert Modality.VIDEO.value == "video"
