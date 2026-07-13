from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_glama_dockerfile_builds_and_installs_checkout_wheel() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.13-slim@sha256:" in dockerfile
    assert "AS builder" in dockerfile
    assert "python -m build --wheel" in dockerfile
    assert "/dist/*.whl" in dockerfile
    assert 'pip install "mlx-memo[cpu]"' not in dockerfile
    assert "ARG EXPECTED_VERSION" in dockerfile
    assert "assert installed == expected" in dockerfile
    assert "HF_MODEL_REVISION=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3" in dockerfile
    assert "revision=os.environ[" in dockerfile
    assert "MEMO_NONINTERACTIVE=1" in dockerfile
    assert "MEMO_MCP_PROFILE=agent" in dockerfile
    assert "MEMO_EMBEDDER_BACKEND=st" in dockerfile
    assert 'CMD ["memo-mcp"]' in dockerfile


def test_glama_sidecar_matches_primary_dockerfile() -> None:
    primary = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    glama = (REPO_ROOT / "Dockerfile.glama").read_text(encoding="utf-8")

    assert primary == glama
