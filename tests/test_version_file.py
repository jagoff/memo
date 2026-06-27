import pytest
from memo.runtime.version_file import read_version_file, write_version_file
from pathlib import Path
import tempfile
import json

def test_read_version_file_missing():
    """Returns None if file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = read_version_file(Path(tmpdir))
        assert result is None

def test_read_version_file_exists():
    """Returns version dict if file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        version_file = Path(tmpdir) / "memo-version.json"
        version_file.write_text(json.dumps({"version": "v1.0.0", "updated_at": "2026-01-01T00:00:00Z"}))

        result = read_version_file(Path(tmpdir))
        assert result == {"version": "v1.0.0", "updated_at": "2026-01-01T00:00:00Z"}

def test_write_version_file():
    """Writes version file with current version."""
    import importlib.metadata
    with tempfile.TemporaryDirectory() as tmpdir:
        current = importlib.metadata.version("mlx-memo")
        write_version_file(Path(tmpdir), current)

        content = json.loads((Path(tmpdir) / "memo-version.json").read_text())
        assert content["version"] == current
        assert "updated_at" in content