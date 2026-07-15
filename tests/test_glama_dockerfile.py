import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_glama_dockerfile_builds_and_installs_checkout_wheel() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    stages = re.findall(
        r"^FROM (python:(3\.\d+)-slim@sha256:[0-9a-f]{64}) AS (builder|runtime)$",
        dockerfile,
        re.MULTILINE,
    )
    assert len(stages) == 2
    assert stages[0] == (*stages[1][:2], "builder")
    assert stages[1][2] == "runtime"

    supported_python = {
        classifier.rsplit(" :: ", maxsplit=1)[-1]
        for classifier in project["classifiers"]
        if classifier.startswith("Programming Language :: Python :: 3.")
    }
    assert stages[0][1] in supported_python
    assert "python -m build --wheel" in dockerfile
    assert "--outdir /dist" in dockerfile
    assert "COPY --from=builder /dist/ /tmp/dist/" in dockerfile
    assert "wheel=$(find /tmp/dist -name '*.whl' -print -quit)" in dockerfile
    assert 'pip install "${wheel}[cpu]"' in dockerfile
    assert "/tmp/memo.whl" not in dockerfile
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
