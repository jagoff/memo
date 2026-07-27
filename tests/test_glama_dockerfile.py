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
    assert "uv export --frozen --extra cpu --no-dev --no-emit-project" in dockerfile
    assert "uv build --wheel --out-dir /dist" in dockerfile
    assert "COPY --from=builder /dist/ /tmp/dist/" in dockerfile
    assert "wheel=$(find /tmp/dist -name '*.whl' -print -quit)" in dockerfile
    assert "torch==2.13.0" in dockerfile
    assert "pip install --require-hashes -r /tmp/dist/runtime-requirements.txt" in dockerfile
    assert 'pip install --no-deps "$wheel"' in dockerfile
    assert "/tmp/memo.whl" not in dockerfile
    assert 'pip install "mlx-memo[cpu]"' not in dockerfile
    assert "ARG EXPECTED_VERSION" in dockerfile
    assert "assert installed == expected" in dockerfile
    assert "MEMO_ST_EMBEDDER_REVISION=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3" in dockerfile
    assert "revision=os.environ['MEMO_ST_EMBEDDER_REVISION']" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "MEMO_NONINTERACTIVE=1" in dockerfile
    assert "MEMO_MCP_PROFILE=agent" in dockerfile
    assert "MEMO_EMBEDDER_BACKEND=st" in dockerfile
    assert "HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3" in dockerfile
    assert 'CMD ["memo", "config", "validate"]' in dockerfile
    assert 'CMD ["memo-mcp"]' in dockerfile
    assert "COPY . ." not in dockerfile
    for required in (
        "pyproject.toml uv.lock README.md LICENSE server.json",
        "src ./src",
        ".agents ./.agents",
        ".claude-plugin ./.claude-plugin",
        "commands ./commands",
        "hooks ./hooks",
        "plugins ./plugins",
        "skills ./skills",
        "statusline ./statusline",
    ):
        assert f"COPY {required}" in dockerfile


def test_dockerignore_is_an_explicit_build_context_allowlist() -> None:
    ignored = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "**" in ignored
    assert "!pyproject.toml" in ignored
    assert "!uv.lock" in ignored
    assert "!src/**" in ignored
    assert not any(line.startswith("!.env") for line in ignored)
    assert not any("memories" in line or "memorias" in line for line in ignored)
    last_include = max(index for index, line in enumerate(ignored) if line.startswith("!"))
    for forbidden in (
        "**/.DS_Store",
        "**/.env",
        "**/.env.*",
        "**/.envrc",
        "**/__pycache__/",
        "**/*.db",
        "**/*.py[cdo]",
        "**/*.sqlite*",
        "**/config.local.*",
    ):
        assert ignored.index(forbidden) > last_include


def test_glama_sidecar_matches_primary_dockerfile() -> None:
    primary = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    glama = (REPO_ROOT / "Dockerfile.glama").read_text(encoding="utf-8")

    assert primary == glama
