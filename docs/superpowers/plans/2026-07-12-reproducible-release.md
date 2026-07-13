# Reproducible Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MCPB, Docker, dependency resolution, and publishing inputs deterministic and source-derived.

**Architecture:** A focused `memo.release_mcpb` module owns deterministic archive creation while `memo.cli_release` validates and exposes it. Release containers install a wheel built from the checkout, and workflows pin every external executable by immutable identity.

**Tech Stack:** Python 3.13, `zipfile`, Click, Docker, uv, GitHub Actions, pytest.

## Global Constraints

- Python remains `>=3.13`; Linux and Apple Silicon paths remain supported.
- The tracked MCPB is rebuilt from `packaging/mcpb` and must be byte-reproducible.
- Docker installs the checkout wheel, never an unqualified PyPI package.
- Every production change starts with a focused failing test.

---

### Task 1: Deterministic MCPB builder

**Files:**
- Create: `src/memo/release_mcpb.py`
- Modify: `src/memo/cli_release.py`
- Test: `tests/test_cli_release.py`

**Interfaces:**
- Produces: `build_mcpb(repo: Path, output: Path | None = None) -> Path`
- Produces: `MCPB_MEMBERS: tuple[str, ...]`

- [ ] **Step 1: Add failing reproducibility and archive-drift tests**

```python
def test_build_mcpb_is_reproducible(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    first = build_mcpb(repo).read_bytes()
    second = build_mcpb(repo).read_bytes()
    assert hashlib.sha256(first).digest() == hashlib.sha256(second).digest()

def test_release_check_rejects_archived_member_drift(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "1.2.3")
    build_mcpb(repo)
    (repo / "packaging/mcpb/server/main.py").write_text("raise SystemExit(9)\n")
    assert any("server/main.py" in issue for issue in release_check_report(repo).issues)
```

- [ ] **Step 2: Run the focused tests and observe missing builder failures**

Run: `uv run --no-sync pytest tests/test_cli_release.py -k 'mcpb and (reproducible or archived)' -v`

- [ ] **Step 3: Implement fixed-member deterministic ZIP creation and archive parity checking**

```python
MCPB_MEMBERS = ("icon.png", "manifest.json", "server/main.py")
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

def build_mcpb(repo: Path, output: Path | None = None) -> Path:
    source = repo / "packaging" / "mcpb"
    destination = output or repo / "packaging" / "memo.mcpb"
    with zipfile.ZipFile(destination, "w") as archive:
        for member in MCPB_MEMBERS:
            info = zipfile.ZipInfo(member, _ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (source / member).read_bytes())
    return destination
```

- [ ] **Step 4: Add `memo release mcpb`, run all release tests, and rebuild the tracked artifact**

Run: `uv run --no-sync pytest tests/test_cli_release.py -v`

- [ ] **Step 5: Commit the builder, tests, and rebuilt artifact**

```bash
git add src/memo/release_mcpb.py src/memo/cli_release.py tests/test_cli_release.py packaging/memo.mcpb
git commit -m "feat: make MCPB releases reproducible"
```

### Task 2: Source-built Docker image

**Files:**
- Modify: `Dockerfile`
- Modify: `Dockerfile.glama`
- Modify: `.github/workflows/docker-publish.yml`
- Test: `tests/test_glama_dockerfile.py`

**Interfaces:**
- Consumes: release tag as Docker `EXPECTED_VERSION` build argument.
- Produces: runtime image whose installed `memo.__version__` equals that argument.

- [ ] **Step 1: Replace the PyPI assertion with failing source-wheel, digest, revision, and version assertions**

```python
assert "python -m build --wheel" in dockerfile
assert "dist/*.whl" in dockerfile
assert "ARG EXPECTED_VERSION" in dockerfile
assert "revision=\"$HF_MODEL_REVISION\"" in dockerfile
assert "FROM python:3.13-slim@sha256:" in dockerfile
assert 'pip install "mlx-memo[cpu]"' not in dockerfile
```

- [ ] **Step 2: Run `uv run --no-sync pytest tests/test_glama_dockerfile.py -v` and observe failure**

- [ ] **Step 3: Implement matching multi-stage Dockerfiles and pass the tag version from the workflow**

```dockerfile
ARG PYTHON_IMAGE=python:3.13-slim@sha256:eb43ff125d8d58d7449dcba7d336c23bcac412f526d861db493b9994d8010280
FROM ${PYTHON_IMAGE} AS builder
WORKDIR /src
COPY . .
RUN python -m pip install build && python -m build --wheel
```

- [ ] **Step 4: Run Dockerfile tests and, when Docker is available, build with the checkout version**

Run: `docker build --build-arg EXPECTED_VERSION=$(uv run --no-sync python -c 'import memo; print(memo.__version__)') .`

- [ ] **Step 5: Commit Docker changes**

```bash
git add Dockerfile Dockerfile.glama .github/workflows/docker-publish.yml tests/test_glama_dockerfile.py
git commit -m "build: install checkout wheel in container"
```

### Task 3: Pinned publishing inputs and frozen dependencies

**Files:**
- Create: `uv.lock`
- Create: `.github/mcp-publisher.sha256`
- Modify: `.github/workflows/*.yml`
- Test: `tests/test_release_workflows.py`

**Interfaces:**
- Produces: immutable action references, `mcp-publisher` v1.7.9 with Linux
  amd64 SHA-256 `ab128162b0616090b47cf245afe0a23f3ef08936fdce19074f5ba0a4469281ac`,
  and CI resolution from `uv.lock`.

- [ ] **Step 1: Add workflow contract tests for SHA-pinned actions, checksum verification, frozen uv, and propagation failure**

```python
assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", ref) for ref in action_refs)
assert "sha256sum --check" in publish
assert "exit 1" in propagation_block
assert "uv sync --frozen" in test_workflow
```

- [ ] **Step 2: Run `uv run --no-sync pytest tests/test_release_workflows.py -v` and observe failures**

- [ ] **Step 3: Resolve immutable upstream identities, write the checksum file, generate `uv.lock`, and update workflows**

Run: `uv lock`

The Docker model revision is
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`; the publisher URL is
`https://github.com/modelcontextprotocol/registry/releases/download/v1.7.9/mcp-publisher_linux_amd64.tar.gz`.

- [ ] **Step 4: Run workflow contract tests and frozen installation smoke**

Run: `uv sync --frozen --extra dev`

- [ ] **Step 5: Commit publishing and lock inputs**

```bash
git add uv.lock .github/mcp-publisher.sha256 .github/workflows tests/test_release_workflows.py
git commit -m "build: pin release supply chain"
```
