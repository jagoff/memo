from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_glama_dockerfile_uses_packaged_stdio_server() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert 'pip install "mlx-memo[cpu]"' in dockerfile
    assert "MEMO_NONINTERACTIVE=1" in dockerfile
    assert "MEMO_MCP_PROFILE=agent" in dockerfile
    assert "MEMO_EMBEDDER_BACKEND=st" in dockerfile
    assert 'CMD ["memo-mcp"]' in dockerfile


def test_glama_sidecar_matches_primary_dockerfile() -> None:
    primary = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    glama = (REPO_ROOT / "Dockerfile.glama").read_text(encoding="utf-8")

    assert primary == glama
