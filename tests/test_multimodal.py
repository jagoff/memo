"""Tests for multi-modal module."""

import pytest

from memo.multimodal import (
    CrossModalResult,
    CrossModalSearch,
    Modality,
    MultiModalContent,
    MultiModalManager,
    MultiModalStore,
    UniversalEmbedder,
)


@pytest.fixture
def multimodal_store(tmp_cfg):
    """Fixture providing MultiModalStore instance."""
    return MultiModalStore(tmp_cfg.state_dir)


@pytest.fixture
def universal_embedder():
    """Fixture providing UniversalEmbedder instance."""
    return UniversalEmbedder()


@pytest.fixture
def cross_modal_search(multimodal_store, universal_embedder):
    """Fixture providing CrossModalSearch instance."""
    return CrossModalSearch(multimodal_store, universal_embedder)


@pytest.fixture
def multimodal_manager(multimodal_store, universal_embedder, cross_modal_search):
    """Fixture providing MultiModalManager instance."""
    return MultiModalManager(multimodal_store, universal_embedder, cross_modal_search)


def test_multimodal_store_init(multimodal_store):
    """Test MultiModalStore initialization."""
    assert multimodal_store.state_dir.is_dir()


def test_multimodal_store_add_content(multimodal_store):
    """Test adding content."""
    content = multimodal_store.add_content(
        content=b"test data",
        modality=Modality.IMAGE.value,
        embedding=[0.1, 0.2, 0.3],
    )

    assert content.id
    assert content.modality == Modality.IMAGE.value


def test_multimodal_store_get_content(multimodal_store):
    """Test getting content."""
    content = multimodal_store.add_content(
        content=b"test data",
        modality=Modality.IMAGE.value,
        embedding=[0.1, 0.2, 0.3],
    )

    retrieved = multimodal_store.get_content(content.id)

    assert retrieved is not None
    assert retrieved.id == content.id


def test_multimodal_store_search_by_embedding(multimodal_store):
    """Test searching by embedding."""
    multimodal_store.add_content(
        content=b"test1",
        modality=Modality.IMAGE.value,
        embedding=[1.0, 0.0, 0.0],
    )
    multimodal_store.add_content(
        content=b"test2",
        modality=Modality.IMAGE.value,
        embedding=[0.0, 1.0, 0.0],
    )

    results = multimodal_store.search_by_embedding(
        query_embedding=[1.0, 0.0, 0.0],
        limit=10,
    )

    assert len(results) >= 1
    assert results[0].similarity > 0.5


def test_multimodal_store_persistence(tmp_cfg):
    """Test content persistence across instances."""
    state_dir = tmp_cfg.state_dir

    # Create first instance and add content
    store1 = MultiModalStore(state_dir)
    content1 = store1.add_content(
        content=b"test",
        modality=Modality.IMAGE.value,
        embedding=[0.1, 0.2, 0.3],
    )

    # Create second instance and verify persistence
    store2 = MultiModalStore(state_dir)
    retrieved = store2.get_content(content1.id)

    assert retrieved is not None
    assert retrieved.id == content1.id


def test_universal_embedder_init(universal_embedder):
    """Test UniversalEmbedder initialization."""
    assert universal_embedder.model_name == "clip-base"


def test_universal_embedder_embed_text(universal_embedder):
    """Test embedding text."""
    embedding = universal_embedder.embed_text("test text")

    assert len(embedding) == 512
    assert all(isinstance(x, float) for x in embedding)


def test_universal_embedder_embed_image(universal_embedder):
    """Test embedding image."""
    embedding = universal_embedder.embed_image(b"fake image data")

    assert len(embedding) == 512
    assert all(isinstance(x, float) for x in embedding)


def test_universal_embedder_embed_audio(universal_embedder):
    """Test embedding audio."""
    embedding = universal_embedder.embed_audio(b"fake audio data")

    assert len(embedding) == 512
    assert all(isinstance(x, float) for x in embedding)


def test_cross_modal_search_init(cross_modal_search):
    """Test CrossModalSearch initialization."""
    assert cross_modal_search.store is not None
    assert cross_modal_search.embedder is not None


def test_cross_modal_search_search_text_find_images(multimodal_store, cross_modal_search):
    """Test searching text to find images."""
    # Add test content
    multimodal_store.add_content(
        content=b"image data",
        modality=Modality.IMAGE.value,
        embedding=cross_modal_search.embedder.embed_text("architecture"),
    )

    results = cross_modal_search.search_text_find_images("architecture")

    assert isinstance(results, list)


def test_cross_modal_search_search_text_find_audio(multimodal_store, cross_modal_search):
    """Test searching text to find audio."""
    multimodal_store.add_content(
        content=b"audio data",
        modality=Modality.AUDIO.value,
        embedding=cross_modal_search.embedder.embed_text("meeting"),
    )

    results = cross_modal_search.search_text_find_audio("meeting")

    assert isinstance(results, list)


def test_cross_modal_search_search_all_modalities(multimodal_store, cross_modal_search):
    """Test searching all modalities."""
    multimodal_store.add_content(
        content=b"text data",
        modality=Modality.TEXT.value,
        embedding=cross_modal_search.embedder.embed_text("test"),
    )

    results = cross_modal_search.search_all_modalities("test")

    assert isinstance(results, dict)


def test_multimodal_manager_init(multimodal_manager):
    """Test MultiModalManager initialization."""
    assert multimodal_manager.store is not None
    assert multimodal_manager.embedder is not None
    assert multimodal_manager.search is not None


def test_multimodal_manager_add_image(tmp_path, multimodal_manager):
    """Test adding image."""
    # Create test image file
    test_image = tmp_path / "test.png"
    test_image.write_bytes(b"fake png data")

    content = multimodal_manager.add_image(test_image)

    assert content.id
    assert content.modality == Modality.IMAGE.value


def test_multimodal_manager_add_audio(tmp_path, multimodal_manager):
    """Test adding audio."""
    test_audio = tmp_path / "test.mp3"
    test_audio.write_bytes(b"fake mp3 data")

    content = multimodal_manager.add_audio(test_audio)

    assert content.id
    assert content.modality == Modality.AUDIO.value


def test_modality_enum():
    """Test Modality enum values."""
    assert Modality.TEXT.value == "text"
    assert Modality.IMAGE.value == "image"
    assert Modality.AUDIO.value == "audio"
    assert Modality.VIDEO.value == "video"


def test_multimodal_content_dataclass():
    """Test MultiModalContent dataclass structure."""
    content = MultiModalContent(
        id="test-id",
        memoria_id="mem-123",
        modality="image",
        content=b"data",
        embedding=[0.1, 0.2],
        metadata={"filename": "test.png"},
        created_at="2026-01-01T00:00:00Z",
    )
    assert content.id == "test-id"
    assert content.modality == "image"


def test_cross_modal_result_dataclass():
    """Test CrossModalResult dataclass structure."""
    result = CrossModalResult(
        content_id="test-id",
        modality="image",
        similarity=0.95,
        metadata={"filename": "test.png"},
    )
    assert result.similarity == 0.95
    assert result.modality == "image"
