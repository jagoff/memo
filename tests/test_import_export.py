"""Tests for import/export module."""

import json

import pytest

from memo.import_export import (
    Exporter,
    ExportResult,
    Importer,
    ImportExportManager,
    ImportResult,
)


@pytest.fixture
def importer(mock_memory):
    """Fixture providing Importer instance."""
    return Importer(mock_memory)


@pytest.fixture
def exporter(mock_memory):
    """Fixture providing Exporter instance."""
    return Exporter(mock_memory)


@pytest.fixture
def import_export_manager(mock_memory):
    """Fixture providing ImportExportManager instance."""
    return ImportExportManager(mock_memory)


def test_importer_init(importer):
    """Test Importer initialization."""
    assert importer.memory is not None


def test_importer_import_json(tmp_path, importer, mock_memory):
    """Test importing from JSON."""
    # Create test JSON file
    data = [
        {
            "title": "Test 1",
            "content": "Content 1",
            "tags": ["test"],
            "type": "note",
        },
        {
            "title": "Test 2",
            "content": "Content 2",
            "tags": ["test"],
            "type": "decision",
        },
    ]
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps(data), encoding="utf-8")

    result = importer.import_json(json_file)

    assert result.imported_count == 2
    assert result.skipped_count == 0


def test_importer_import_csv(tmp_path, importer):
    """Test importing from CSV."""
    # Create test CSV file
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "title,content,tags,type\nTest 1,Content 1,test,note\nTest 2,Content 2,test,decision\n",
        encoding="utf-8",
    )

    result = importer.import_csv(csv_file)

    assert result.imported_count == 2
    assert result.skipped_count == 0


def test_exporter_init(exporter):
    """Test Exporter initialization."""
    assert exporter.memory is not None


def test_exporter_export_json(tmp_path, exporter, mock_memory):
    """Test exporting to JSON."""
    # Create test memorias
    mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    output_path = tmp_path / "export.json"
    result = exporter.export_json(output_path)

    assert result.exported_count >= 1
    assert output_path.is_file()

    import json

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(data) >= 1


def test_exporter_export_csv(tmp_path, exporter, mock_memory):
    """Test exporting to CSV."""
    mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    output_path = tmp_path / "export.csv"
    result = exporter.export_csv(output_path)

    assert result.exported_count >= 1
    assert output_path.is_file()

    content = output_path.read_text(encoding="utf-8")
    assert "id,title,body" in content


def test_exporter_export_markdown_bundle(tmp_path, exporter, mock_memory):
    """Test exporting to Markdown bundle."""
    mock_memory.save(
        content="Test content",
        title="Test",
        tags=["test"],
    )

    output_path = tmp_path / "export.zip"
    result = exporter.export_markdown_bundle(output_path)

    assert result.exported_count >= 1
    assert output_path.is_file()

    import zipfile

    with zipfile.ZipFile(output_path, "r") as zf:
        assert len(zf.namelist()) >= 1
        assert any(f.endswith(".md") for f in zf.namelist())


def test_import_export_manager_init(import_export_manager):
    """Test ImportExportManager initialization."""
    assert import_export_manager.memory is not None
    assert import_export_manager.importer is not None
    assert import_export_manager.exporter is not None


def test_import_export_manager_import_from(tmp_path, import_export_manager):
    """Test import_from with format."""
    data = [{"title": "Test", "content": "Content", "tags": [], "type": "note"}]
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps(data), encoding="utf-8")

    result = import_export_manager.import_from(json_file, "json")

    assert result.imported_count == 1


def test_import_export_manager_export_to(tmp_path, import_export_manager):
    """Test export_to with format."""
    output_path = tmp_path / "export.json"
    result = import_export_manager.export_to(output_path, "json")

    assert result.exported_count >= 0
    assert output_path.is_file()


def test_import_result_dataclass():
    """Test ImportResult dataclass structure."""
    result = ImportResult(
        imported_count=10,
        skipped_count=2,
        errors=["error1", "error2"],
    )
    assert result.imported_count == 10
    assert result.skipped_count == 2
    assert len(result.errors) == 2


def test_export_result_dataclass():
    """Test ExportResult dataclass structure."""
    result = ExportResult(
        exported_count=100,
        output_path="/path/to/export.json",
        format="json",
    )
    assert result.exported_count == 100
    assert result.format == "json"
