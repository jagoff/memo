# memo Full QA Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox - [ ] syntax for tracking.

**Goal:** Prove the current memo worktree works end to end across every discoverable CLI/MCP surface, native MLX, Linux CPU Docker, runtime integrations, packaging, and recovery paths; fix every confirmed defect with an atomic regression-tested commit.

**Architecture:** Eleven cumulative QA gates run against isolated state and locally built artifacts. Existing tests provide component evidence, dynamic surface discovery proves wiring completeness, and real subprocess/protocol journeys prove user behavior. Each confirmed defect interrupts the current gate for a red-green repair commit, after which the affected gate restarts from a fresh sandbox.

**Tech Stack:** Python 3.13 and 3.14, uv 0.11.21, Click, FastMCP 3.4.x, pytest, Ruff, mypy, sqlite/sqlite-vec, MLX/MLX-LM on Apple Silicon, sentence-transformers on Linux CPU, Docker 29.x, Homebrew 6.x, bash, JSON, Markdown.

## Global Constraints

- Authority is the current local worktree. Do not pull, merge, rebase, or cherry-pick origin/master.
- Preserve all pre-existing modified and untracked files. Stage explicit campaign-owned paths only.
- Do not use destructive Git commands, broad staging, or commit-a.
- Never touch the real vault, default state directory, real MCP client configuration, or production daemon labels.
- Every CLI/CliRunner execution sets MEMO_NONINTERACTIVE, MEMO_DATA_DIR, and MEMO_STATE_DIR to isolated values.
- Real MLX tests keep deferred mlx/mlx_lm imports and use Sequence[str] for MLXEmbedder.embed.
- Markdown remains source of truth. Rebuild through memo reindex --rebuild, never by deleting memvec.db.
- CI parity order is Ruff, mypy, pytest.
- Retrieval/ranking/ingest changes require the recall regression evaluation with k=5 and force enabled.
- Hook, daemon, runtime, installer, and migration changes require their focused contract suites.
- A confirmed defect receives a failing regression test before the production fix.
- Each coherent defect is committed atomically after focused and domain verification.
- Raw logs, caches, temporary databases, model downloads, Docker layers, and coverage artifacts stay outside Git.
- The only durable campaign documents are this plan, the approved design, regression tests/fixes, and the final report.
- The campaign is not complete while any manifest row is unexplained or any confirmed S0-S3 defect remains open.

## File Structure

Durable files:

- Existing: docs/superpowers/specs/2026-07-22-memo-full-qa-design.md — approved contract.
- Create: docs/superpowers/reports/2026-07-22-memo-full-qa.md — final evidence and findings.
- Create only for confirmed defects: a focused regression test in the existing test module that owns the failing public contract; the failure ledger records that exact path before editing.
- Modify only for confirmed defects: the smallest owning source, packaging, workflow, Docker, or documentation file.

Disposable evidence tree:

    /tmp/memo-full-qa-20260722/
      baseline/
      inventory/
      logs/gate-02/ through logs/gate-11/
      native/py313/
      native/py314/
      native/journey/
      docker/
      findings/
      final/

All commands run from /Users/fer/repos/memo. QA_ROOT means
/tmp/memo-full-qa-20260722.

---

### Task 1: Freeze baseline and protect the shared worktree

**Files:**
- Read: AGENTS.md
- Read: docs/superpowers/specs/2026-07-22-memo-full-qa-design.md
- Create outside Git: /tmp/memo-full-qa-20260722/baseline/*

**Interfaces:**
- Produces: immutable baseline evidence consumed by every later task.

- [ ] **Step 1: Create evidence directories**

Run:

    QA_ROOT=/tmp/memo-full-qa-20260722
    mkdir -p "$QA_ROOT"/baseline "$QA_ROOT"/inventory "$QA_ROOT"/logs "$QA_ROOT"/native/py313 "$QA_ROOT"/native/py314 "$QA_ROOT"/docker "$QA_ROOT"/findings "$QA_ROOT"/final

Expected: directories exist and no repository file changes.

- [ ] **Step 2: Record Git chain of custody**

Run:

    git status --short --branch > "$QA_ROOT/baseline/git-status.txt"
    git diff --binary > "$QA_ROOT/baseline/git-diff.patch"
    git diff --cached --binary > "$QA_ROOT/baseline/git-diff-cached.patch"
    git log --date=iso-strict --pretty=fuller -30 > "$QA_ROOT/baseline/git-log.txt"
    git rev-parse HEAD > "$QA_ROOT/baseline/head.txt"
    git rev-list --left-right --count origin/master...HEAD > "$QA_ROOT/baseline/divergence.txt"

Expected: HEAD is 5348f0cd or a later campaign commit; the snapshot includes
all pre-existing .superpowers, privacy, telemetry, and .wrangler changes.

- [ ] **Step 3: Record environment and versions without secrets**

Run:

    { uname -a; sw_vers; sysctl -n hw.memsize; df -h .; uv --version; python3 --version; docker version; brew --version; git --version; sqlite3 --version; command -v memo; command -v memo-mcp; } > "$QA_ROOT/baseline/environment.txt" 2>&1
    env | cut -d= -f1 | grep -E '^(MEMO|SYNAPSE|HF|XDG)_' | sort > "$QA_ROOT/baseline/inherited-env-names.txt" || true

Expected: no environment values or secrets, only variable names.

- [ ] **Step 4: Record release metadata and checksums**

Run:

    uv run --no-sync python -c "import json,tomllib,pathlib; p=tomllib.loads(pathlib.Path('pyproject.toml').read_text()); print(json.dumps({'version':p['project']['version'],'python':p['project']['requires-python'],'scripts':p['project']['scripts']},indent=2))" > "$QA_ROOT/baseline/versions.json"
    shasum -a 256 pyproject.toml uv.lock server.json .claude-plugin/plugin.json CHANGELOG.md Dockerfile Dockerfile.glama > "$QA_ROOT/baseline/checksums.txt"

Expected: version and both console entry points are present.

- [ ] **Step 5: Prove no mutation**

Run:

    cmp "$QA_ROOT/baseline/git-diff.patch" <(git diff --binary)
    cmp "$QA_ROOT/baseline/git-diff-cached.patch" <(git diff --cached --binary)

Expected: both exit 0.

---

### Task 2: Inventory every CLI, MCP, config, and packaging surface

**Files:**
- Verify: tests/test_cli_mcp_surface_smoke.py
- Verify: src/memo/cli.py
- Verify: src/memo/server.py
- Verify: src/memo/experimental_index.md
- Create outside Git: /tmp/memo-full-qa-20260722/inventory/*

**Interfaces:**
- Produces: live manifests used by the CLI and MCP user journeys.

- [ ] **Step 1: Collect recursive CLI routes**

Run the Click tree walker that emits GROUP or LEAF plus the full command path
from memo.cli.cli into "$QA_ROOT/inventory/cli.txt".

Expected initial count: 320 total routes.

- [ ] **Step 2: Run every CLI help path**

Run:

    mkdir -p "$QA_ROOT/logs/gate-02"
    uv run --no-sync pytest tests/test_cli_mcp_surface_smoke.py::test_cli_command_help_smoke -q --disable-warnings --maxfail=1 | tee "$QA_ROOT/logs/gate-02/cli-help.txt"

Expected: one pass per route, no model download or default-vault access.

- [ ] **Step 3: Verify MCP profiles and decorated-tool wiring**

Run:

    uv run --no-sync pytest tests/test_cli_mcp_surface_smoke.py::test_mcp_profile_tool_counts tests/test_cli_mcp_surface_smoke.py::test_mcp_full_profile_registers_every_decorated_server_tool -q | tee "$QA_ROOT/logs/gate-02/mcp-wiring.txt"

Expected initial counts: agent=14, core=34, full=137; every decorated tool is
registered in full.

- [ ] **Step 4: Export MCP resources and exact schemas**

Build an isolated Config with embedder_dims=4, stub embedder output
[1.0, 0.0, 0.0, 0.0], list tools/resources/templates/prompts, close Memory in
finally, and write sorted JSON to the inventory directory.

Expected: 137 tools, memo://recent, memo://memory/{id}, zero prompts.

- [ ] **Step 5: Verify wiring, config, and boundary contracts**

Run:

    uv run --no-sync pytest tests/test_no_unwired_loops.py tests/test_verification_wiring.py tests/test_interject_wire.py tests/test_cli_consult_attribution.py tests/test_surface_profiles.py tests/test_server_resources.py tests/test_config_catalog.py tests/test_flags.py tests/test_contract_stub_compatibility.py -q | tee "$QA_ROOT/logs/gate-02/contracts.txt"

Expected: all pass.

- [ ] **Step 6: Verify documentation and experimental boundary**

Run:

    uv run --no-sync pytest tests/test_offline_docs.py tests/test_config_md.py tests/test_code_traceability.py -q | tee "$QA_ROOT/logs/gate-02/docs.txt"
    rg -n '^## ' src/memo/experimental_index.md > "$QA_ROOT/inventory/experimental-modules.txt"
    rg -n '^from memo import server_|register\(' src/memo/server.py src/memo/server_*.py > "$QA_ROOT/inventory/server-registers.txt"

Expected: tests pass and exposed experimental modules are classified.

---

### Task 3: Run structural, security, packaging, and release gates

**Files:**
- Verify: pyproject.toml, uv.lock, Dockerfile, Dockerfile.glama, server.json
- Verify: packaging/* and .github/workflows/*

**Interfaces:**
- Produces: clean artifact baseline.

- [ ] **Step 1: Verify frozen Python 3.13 environment**

Run:

    uv sync --frozen --extra dev --extra http --python 3.13

Expected: exit 0 without changing uv.lock.

- [ ] **Step 2: Run quality order**

Run:

    mkdir -p "$QA_ROOT/logs/gate-03"
    uv run --no-sync ruff format --check .
    uv run --no-sync ruff check src/ tests/ scripts/
    uv run --no-sync mypy src/memo
    uv run --no-sync python scripts/quality_gate.py

Expected: all exit 0.

- [ ] **Step 3: Run release and supply-chain contracts**

Run:

    uv run --no-sync pytest tests/test_supply_chain.py tests/test_release_workflows.py tests/test_release_mcpb_node.py tests/test_cli_release.py tests/test_model_pins.py tests/test_runtime_prewarm_pins.py tests/test_install_seed.py tests/test_runtime_isolation.py tests/test_runtime_update.py tests/test_autoupdate.py -q | tee "$QA_ROOT/logs/gate-03/release.txt"

Expected: all pass.

- [ ] **Step 4: Build wheel and sdist**

Run:

    rm -rf "$QA_ROOT/artifacts"
    mkdir -p "$QA_ROOT/artifacts"
    uv build --wheel --sdist --out-dir "$QA_ROOT/artifacts"

Expected: one version-3.11.0 wheel and one sdist unless an explicit campaign
fix synchronously changes every required version file.

- [ ] **Step 5: Inspect artifact contents**

Run:

    uv run --no-sync python -m zipfile -l "$QA_ROOT"/artifacts/*.whl > "$QA_ROOT/logs/gate-03/wheel-contents.txt"
    tar -tzf "$QA_ROOT"/artifacts/*.tar.gz > "$QA_ROOT/logs/gate-03/sdist-contents.txt"
    ! rg -i '(^|/)(\.env|memvec\.db|\.wrangler|__pycache__|\.DS_Store|config\.local)' "$QA_ROOT/logs/gate-03/"*-contents.txt
    rg 'memo/agent_assets/.*/(hooks.json|plugin.json|memo.md)|memo/agent_assets/server.json' "$QA_ROOT/logs/gate-03/wheel-contents.txt"

Expected: forbidden state absent and required agent assets present.

- [ ] **Step 6: Install artifact under Python 3.13 and 3.14**

For each Python, create a QA_ROOT venv, install the wheel, change directory to
/tmp, import memo and memo.server, and run memo --version.

Expected: imports resolve from isolated site-packages, not the checkout.

---

### Task 4: Run the complete automated verification matrix

**Files:**
- Verify: tests/*
- Verify: eval/regression_labels.json

**Interfaces:**
- Produces: green component baseline before user journeys.

- [ ] **Step 1: Run resource hygiene serially**

Run:

    mkdir -p "$QA_ROOT/logs/gate-04"
    uv run --no-sync python -m pytest -m "resource_hygiene" -n 0 --timeout=120 --resource-hygiene | tee "$QA_ROOT/logs/gate-04/resource-hygiene.txt"

Expected: no unclosed database, stream, socket, or file warnings.

- [ ] **Step 2: Run non-slow with coverage**

Run:

    uv run --no-sync python -m pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing --cov-report=xml:"$QA_ROOT/logs/gate-04/coverage.xml" | tee "$QA_ROOT/logs/gate-04/non-slow.txt"

Expected: all pass and fail_under is met.

- [ ] **Step 3: Run int8 shipped-default lane**

Run:

    MEMO_VEC_QUANTIZE=int8 uv run --no-sync python -m pytest -m "not slow and not float32_precision" -n auto --timeout=120 | tee "$QA_ROOT/logs/gate-04/int8.txt"

Expected: all pass.

- [ ] **Step 4: Run slow and real MLX**

Run:

    uv run --no-sync python -m pytest -m "slow" --timeout=300 -v | tee "$QA_ROOT/logs/gate-04/slow.txt"
    uv run --no-sync python -m pytest tests/test_smoke_mlx.py -m requires_mlx --timeout=300 -v | tee "$QA_ROOT/logs/gate-04/mlx.txt"

Expected: every locally applicable test passes.

- [ ] **Step 5: Run deterministic stability checks**

Run:

    uv sync --frozen --extra dev --extra test-stability --python 3.13
    uv run --no-sync pytest -m "not slow" -n 0 --randomly-seed=20260722 --timeout=120 | tee "$QA_ROOT/logs/gate-04/random.txt"
    uv run --no-sync pytest -m "concurrency or resource_hygiene" -n 0 --count=10 --repeat-scope=session --timeout=120 | tee "$QA_ROOT/logs/gate-04/repeated.txt"

Expected: all pass without retry masking.

- [ ] **Step 6: Run retrieval, grounding, and graph evaluations**

Run:

    uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force | tee "$QA_ROOT/logs/gate-04/eval-recall.txt"
    uv run --no-sync pytest tests/test_eval_recall.py tests/test_eval_grounding.py tests/test_grounding.py tests/test_search_graph_signal.py tests/test_graph_signal.py -q | tee "$QA_ROOT/logs/gate-04/eval-contracts.txt"

Expected: no quality/noise regression.

- [ ] **Step 7: Repeat non-slow under Python 3.14**

Run:

    uv sync --frozen --extra dev --extra http --python 3.14
    uv run --no-sync python -m pytest -m "not slow" -n auto --timeout=120 | tee "$QA_ROOT/logs/gate-04/python314.txt"

Expected: all selected tests pass.

---

### Task 5: Prove every post-audit change domain

**Files:**
- Verify: src/memo/temporal.py
- Verify: src/memo/memory/search_ops.py
- Verify: src/memo/store/schema.py and src/memo/store/queries.py
- Verify: src/memo/presence.py, src/memo/capture_core.py, src/memo/recall_server.py

**Interfaces:**
- Produces: focused evidence for all work added after the 2026-07-21 audit.

- [ ] **Step 1: Run valid-time and bitemporal contracts**

Run:

    mkdir -p "$QA_ROOT/logs/gate-05"
    uv run --no-sync pytest tests/test_temporal.py tests/test_validity_filter.py tests/test_cli_as_of.py tests/test_server_asof.py tests/test_briefing_temporal_facts.py tests/test_save_normalize_dates.py tests/test_cli_invalidate.py tests/test_supersede_stamp.py tests/test_superseded_pairs.py tests/test_memory_reindex.py tests/test_cli_migrate_vault.py -q | tee "$QA_ROOT/logs/gate-05/temporal.txt"

Expected: offsets, bare-date end-of-day, inverted interval clamping,
invalidation undo, reindex, and migration idempotency all pass.

- [ ] **Step 2: Run graph-first retrieval contracts**

Run:

    uv run --no-sync pytest tests/test_search_graph_signal.py tests/test_graph_signal.py tests/test_graph_projection.py tests/test_graph_canonical.py tests/test_graph_entity_merge.py tests/test_graph_semantic_relations.py tests/test_graph_proximity.py tests/test_server_graph_tool.py tests/test_cli_repo.py tests/test_repo_index.py tests/test_codegraph_loader.py -q | tee "$QA_ROOT/logs/gate-05/graph.txt"

Expected: all pass.

- [ ] **Step 3: Run capture, presence, proactive, and daemon contracts**

Run:

    uv run --no-sync pytest tests/test_capture_core.py tests/test_capture_grounding.py tests/test_capture_incremental.py tests/test_presence.py tests/test_proactive_surfaces.py tests/test_proactive_integration.py tests/test_recall_daemon_health.py tests/test_recall_daemon_restart.py tests/test_daemon_startup_flock.py tests/test_recall_server.py tests/test_dream_progress_tty.py tests/test_dream_validity_extract.py -q | tee "$QA_ROOT/logs/gate-05/runtime.txt"

Expected: all pass.

- [ ] **Step 4: Run security and telemetry contracts**

Run:

    uv run --no-sync pytest tests/test_reference_noise_gate.py tests/test_html_security.py tests/test_http_auth.py tests/test_guard_telemetry.py tests/test_supply_chain.py tests/test_redact.py -q | tee "$QA_ROOT/logs/gate-05/security.txt"

Expected: all pass and tests emit no real telemetry.

---

### Task 6: Run native CLI journeys as a real user

**Files:**
- Exercise: wheel-installed memo under QA_ROOT/native/py313
- Create outside Git: /tmp/memo-full-qa-20260722/native/journey/*

**Interfaces:**
- Produces: cross-command state and persistence evidence.

- [ ] **Step 1: Create isolated native environment**

Run:

    NATIVE="$QA_ROOT/native/journey"
    rm -rf "$NATIVE"
    mkdir -p "$NATIVE"/home "$NATIVE"/data "$NATIVE"/state "$NATIVE"/config "$NATIVE"/vault "$NATIVE"/repo
    MEMO_BIN="$QA_ROOT/native/py313/venv/bin/memo"
    export HOME="$NATIVE/home"
    export MEMO_DATA_DIR="$NATIVE/data"
    export MEMO_STATE_DIR="$NATIVE/state"
    export MEMO_CONFIG_DIR="$NATIVE/config"
    export MEMO_VAULT_PATH="$NATIVE/vault"
    export MEMO_NONINTERACTIVE=1
    export MEMO_EMBEDDER_VIA_DAEMON=0
    export MEMO_UPDATE_CHECK_ENABLED=0
    export MEMO_AUTO_UPDATE=0

Expected: every path is below QA_ROOT.

- [ ] **Step 2: Validate first-run, config, and doctor**

Run:

    "$MEMO_BIN" config validate
    "$MEMO_BIN" config show --effective
    "$MEMO_BIN" doctor
    "$MEMO_BIN" stats

Expected: no prompt blocks; effective paths are isolated; doctor recognizes the
native backend; empty stats are coherent.

- [ ] **Step 3: Exercise CRUD, retrieval, history, and valid time**

Save a dated PostgreSQL decision with JSON output and capture its ID. Run get,
BM25 search, hybrid search without reranking, edit, rename, record-history,
as-of list/search, invalidate, undo, and get again.

Expected: the ID remains stable, history records mutations, invalidation removes
current visibility, undo restores it, and every JSON response parses.

- [ ] **Step 4: Exercise graph, entities, links, and advanced retrieval**

Save graph evidence linking PostgreSQL and the recall daemon. Run
extract-entities, graph rebuild/stats/neighbors/trace, entities, related, and
search with explain.

Expected: the graph has nodes and edges, trace contains evidence IDs, and search
explanation contains graph reasoning when enabled.

- [ ] **Step 5: Exercise backup, export/import, reindex, and Markdown truth**

Run backup create/list, export JSON, export Markdown bundle, reindex --rebuild,
doctor --gc, and list. Hand-edit a copied sandbox Markdown record before rebuild.

Expected: artifacts exist, hand-edited Markdown wins, records survive rebuild,
and doctor has no unexplained orphan.

- [ ] **Step 6: Exercise invalid and boundary inputs**

Save Unicode text containing Córdoba, emoji, decomposed accents, Arabic, and
Markdown. Try a missing ID, empty search, invalid type/date/limit, and malformed
stdin.

Expected: valid Unicode round-trips; invalid inputs return nonzero actionable
errors without traceback or partial state.

- [ ] **Step 7: Delete and restore**

Delete the seeded ID, prove get fails, restore it, and prove get plus search
succeed.

Expected: Markdown and derived indexes agree after both transitions.

---

### Task 7: Run all MCP profiles over real stdio and cross-check CLI

**Files:**
- Exercise: isolated memo-mcp binary
- Verify: tests/test_server*.py

**Interfaces:**
- Consumes: FastMCP Client and StdioTransport(command, args, env, cwd).
- Produces: real handshake, schema, tool, resource, and cross-interface evidence.

- [ ] **Step 1: Handshake and enumerate profiles**

For agent, core, and full, construct StdioTransport with the isolated binary and
environment, enter FastMCP Client, then call list_tools, list_resources,
list_resource_templates, and list_prompts.

Expected initial counts: 14, 34, 137; one resource, one template, zero prompts;
serverInfo reports memo and the isolated artifact version.

- [ ] **Step 2: Execute core MCP lifecycle**

Call memo_version, memo_save, memo_get, memo_search, memo_search_trace,
memo_update, memo_rename, memo_history, memo_record_diff, memo_forget,
memo_unforget, and memo_delete in a single stateful narrative.

Expected: structured results preserve the record ID and state transitions.

- [ ] **Step 3: Cross-check MCP and CLI**

Save through MCP and read/search through the isolated CLI. Update through CLI
and read through MCP.

Expected: both directions observe identical state.

- [ ] **Step 4: Exercise advanced MCP domains**

Call successful scenarios for valid-time/as-of, fact edges, graph/trace, backup,
passport export/import, repo index/search, idle capture, notification pop, and
unified briefing.

Expected: structured success or a documented EXPECTED-UNAVAILABLE only for an
explicit missing model/dependency; errors never leave partial mutations.

- [ ] **Step 5: Read all resources/templates**

Read memo://recent, save a record, then read memo://memory/{id}.

Expected: recent and exact record content match persisted state.

- [ ] **Step 6: Run all MCP domain tests**

Run:

    mkdir -p "$QA_ROOT/logs/gate-07"
    uv run --no-sync pytest tests/test_server*.py -q | tee "$QA_ROOT/logs/gate-07/server-tests.txt"

Expected: all pass. These tests provide behavior evidence for full-profile tools
outside the cross-interface narrative.

---

### Task 8: Exercise HTTP, REST, hooks, daemons, and TUI

**Files:**
- Verify: src/memo/server.py, src/memo/server_http.py
- Verify: hooks/hooks.json and statusline/memo-statusline.sh

**Interfaces:**
- Produces: network and operational runtime evidence without production changes.

- [ ] **Step 1: Launch authenticated MCP HTTP on loopback**

Select a free port, generate a 40-character token, start isolated memo-mcp with
HTTP transport and host 127.0.0.1, and poll until ready.

Expected: process remains alive and binds only loopback.

- [ ] **Step 2: Test valid and rejected clients**

Use StreamableHttpTransport with bearer auth to list tools and call memo_version
and memo_stats. Repeat with no token and a wrong token.

Expected: valid calls pass; invalid clients receive 401; logs contain no token.

- [ ] **Step 3: Test unsafe bind refusal**

Attempt 0.0.0.0 without MEMO_MCP_ALLOW_NON_LOOPBACK, then explicit allow with
auth disabled.

Expected: both unsafe combinations fail closed with actionable errors.

- [ ] **Step 4: Run HTTP and REST suites**

Run:

    mkdir -p "$QA_ROOT/logs/gate-08"
    uv run --no-sync pytest tests/test_http_auth.py tests/test_server_http.py tests/test_cli_http.py -q | tee "$QA_ROOT/logs/gate-08/http.txt"

Expected: all pass.

- [ ] **Step 5: Run hook and daemon lifecycle suites**

Run:

    uv run --no-sync pytest tests/test_hook_contract.py tests/test_recall_hooks.py tests/test_recall_server.py tests/test_recall_shutdown.py tests/test_recall_daemon_health.py tests/test_recall_daemon_restart.py tests/test_daemon_startup_flock.py tests/test_embedder_client.py tests/test_maint_daemon.py tests/test_runtime_daemon_install.py tests/test_install_watcher.py -q | tee "$QA_ROOT/logs/gate-08/daemons-hooks.txt"

Expected: all pass within their timeout budgets.

- [ ] **Step 6: Run live sandbox recall-daemon lifecycle**

Start, status, recall, restart, status, stop, and final status using the native
sandbox.

Expected: warm recall uses daemon, restart replaces it, stop removes sandbox
socket/PID, and final status is stopped.

- [ ] **Step 7: Drive interactive surfaces**

Run:

    uv run --no-sync pytest tests/test_tui_package.py tests/test_tui_recall_quality.py tests/test_config_tui_app.py tests/test_config_tui_controls.py tests/test_dashboard.py tests/test_dashboard_build.py -q | tee "$QA_ROOT/logs/gate-08/tui.txt"

Expected: all snapshots and PTY interactions pass.

---

### Task 9: Validate installers, client wiring, updates, and Homebrew

**Files:**
- Verify: install.sh
- Verify: src/memo/cli_install_mcp.py and src/memo/runtime/mcp_config.py
- Verify: public Homebrew formula

**Interfaces:**
- Produces: isolated installation and fake-client wiring evidence.

- [ ] **Step 1: Run installer into a fake home**

Use a fresh HOME plus MEMO_INSTALL_SPEC=/Users/fer/repos/memo,
MEMO_INSTALL_DOWNLOAD_MODELS=no, and MEMO_INSTALL_SKIP_AGENT_CONFIG=1.

Expected: installed memo reports the worktree version; real runtime unchanged.

- [ ] **Step 2: Run installer and MCP-config contracts**

Run:

    mkdir -p "$QA_ROOT/logs/gate-09"
    uv run --no-sync pytest tests/test_cli_install_mcp.py tests/test_runtime_mcp_config.py tests/test_runtime_daemon_install.py tests/test_runtime_isolation.py tests/test_cli_first_run.py tests/test_cli_init.py tests/test_cli_onboard.py tests/test_setup_config_io.py tests/test_setup_vaults.py -q | tee "$QA_ROOT/logs/gate-09/installers.txt"

Expected: all pass.

- [ ] **Step 3: Test every supported fake client**

For every path in KNOWN_MCP_CONFIGS, run installation/generation twice in its
own fake HOME. Parse resulting JSON, YAML, or TOML and preserve an unrelated
sentinel key.

Expected: idempotent valid config points to isolated memo-mcp.

- [ ] **Step 4: Validate Homebrew**

Discover the tap formula, run brew style, brew audit --strict --online, a clean
install/build, brew test, and doctor --strict-runtime with isolated data/state.
Uninstall the QA formula afterward.

Expected: all pass. Published version differences are channel evidence, not a
current-worktree failure.

- [ ] **Step 5: Verify update and migration contracts**

Run:

    uv run --no-sync pytest tests/test_runtime_update.py tests/test_autoupdate.py tests/test_cli_migrate_vault.py tests/test_store_migrators.py -q | tee "$QA_ROOT/logs/gate-09/update-migrate.txt"

Expected: all pass and real installed runtime is unchanged.

---

### Task 10: Build and dogfood Linux CPU Docker

**Files:**
- Verify: Dockerfile, Dockerfile.glama, docs/docker.md
- Create outside Git: /tmp/memo-full-qa-20260722/docker/*

**Interfaces:**
- Produces: memo-full-qa:20260722 and CPU user evidence.

- [ ] **Step 1: Build the local image**

Read the version from pyproject.toml and run Docker build with EXPECTED_VERSION,
tagging memo-full-qa:20260722.

Expected: build succeeds, pinned CPU model downloads, version assertion passes,
and the final image is non-root.

- [ ] **Step 2: Verify image, package, and offline model**

Inspect the image. Run id, memo --version, and import memo inside it. Run doctor
with --network none.

Expected: uid is nonzero, memo imports from site-packages, offline model works.

- [ ] **Step 3: Run persistent CLI journey**

Create volume memo-full-qa-data. In separate containers save a Docker QA
decision, search it, list it, and inspect graph stats.

Expected: later containers read state from the first.

- [ ] **Step 4: Verify graceful MLX-only degradation**

Run ask, synthesize, dream, and rerank inside the CPU image.

Expected: documented backend-unavailable responses, no traceback, misleading
success, or partial mutation.

- [ ] **Step 5: Run stdio and HTTP MCP**

Use docker run -i plus FastMCP Client for stdio. Start authenticated HTTP on an
ephemeral host port, call memo_version/search, reject missing auth, then stop.

Expected: both transports pass and share the named volume.

- [ ] **Step 6: Test permissions, limits, signals, and restart**

Run with read-only root plus writable /data and /tmp, CPU/memory limits, and send
SIGTERM during idle server operation.

Expected: supported writes work, unwritable paths fail clearly, shutdown is
graceful, and restart opens the volume cleanly.

- [ ] **Step 7: Validate Glama and docs contracts**

Build Dockerfile.glama with the same expected version and run:

    uv run --no-sync pytest tests/test_offline_docs.py tests/test_release_workflows.py -q

Expected: image and contracts pass.

---

### Task 11: Stress storage, migrations, recovery, security, and performance

**Files:**
- Verify: src/memo/store/*
- Verify: src/memo/runtime/migrate.py
- Verify: src/memo/recall_server.py

**Interfaces:**
- Produces: resilience and repeatable performance evidence.

- [ ] **Step 1: Run storage/state-machine/concurrency tests**

Run:

    mkdir -p "$QA_ROOT/logs/gate-11"
    uv run --no-sync pytest tests/test_vector_store_state_machine.py tests/test_vector_database_contracts.py tests/test_store.py tests/test_store_migrators.py tests/test_db_health.py tests/test_sqlite_cleanup.py tests/test_sqlite_resource_hygiene.py tests/test_sync_flows.py tests/test_daemon_startup_flock.py -q | tee "$QA_ROOT/logs/gate-11/storage.txt"

Expected: all pass.

- [ ] **Step 2: Run backup, restore, reindex, and history tests**

Run:

    uv run --no-sync pytest tests/test_cli_backup.py tests/test_import_export.py tests/test_history.py tests/test_history_store.py tests/test_memory_history.py tests/test_memory_reindex.py tests/test_cli_migrate_vault.py tests/test_supersede_stamp.py -q | tee "$QA_ROOT/logs/gate-11/recovery.txt"

Expected: all pass.

- [ ] **Step 3: Run malformed-input and security suites**

Run:

    uv run --no-sync pytest tests/test_errors.py tests/test_write_ops_failure.py tests/test_save_gate.py tests/test_html_security.py tests/test_reference_noise_gate.py tests/test_redact.py tests/test_http_auth.py tests/test_supply_chain.py -q | tee "$QA_ROOT/logs/gate-11/security.txt"

Expected: all pass.

- [ ] **Step 4: Measure contractual performance**

Run:

    uv run --no-sync pytest tests/test_recall_latency.py tests/test_perf.py tests/test_tier2_perf_fixes.py -v | tee "$QA_ROOT/logs/gate-11/performance.txt"

Expected: contractual budgets pass; non-contractual timings are recorded only.

- [ ] **Step 5: Repeat real use 100 times**

In a new native sandbox, run 100 save/search/get cycles, record elapsed time,
maximum RSS, open files, ResourceWarning output, restart, and retrieve all IDs.

Expected: no descriptor growth, warning, lock, or missing record.

---

### Task 12: Repair each confirmed defect atomically

**Files:**
- Test: the existing focused test module owning the failed contract.
- Modify: the smallest source/docs/packaging file found by the reduced reproduction.
- Create outside Git: one Markdown record below /tmp/memo-full-qa-20260722/findings/, named with the assigned finding ID before reproduction begins.

**Interfaces:**
- Consumes: one FAIL from Tasks 2-11.
- Produces: one atomic fix commit and a green restart of the gate.

- [ ] **Step 1: Reduce and classify**

Record ID, severity S0-S3, exact command/environment, expected/actual output,
logs, and minimal isolated fixture.

Expected: failure reproduces twice in fresh sandboxes.

- [ ] **Step 2: Write focused regression**

Use tmp_cfg or explicit Config. CliRunner supplies isolated env. MCP tests close
Memory in finally.

Expected: new test fails for the original reason.

- [ ] **Step 3: Verify red**

Run exact pytest node with -v.

Expected: deterministic FAIL, not setup contamination.

- [ ] **Step 4: Apply smallest systemic fix**

Use flags.py for behavior flags, config.py for storage/model settings,
MemoError subclasses for domain errors, and deferred optional imports.

Expected: no unrelated refactor.

- [ ] **Step 5: Verify focused and domain green**

Run node, file, originating task's domain suite, Ruff on touched Python, and
mypy when src/memo changed.

Expected: all pass.

- [ ] **Step 6: Repeat real-user reproduction**

Use a fresh sandbox and the original CLI/MCP/Docker/install invocation.

Expected: correct visible and persisted behavior.

- [ ] **Step 7: Commit explicit paths**

Run git diff --check on owned paths, git add only explicit owned paths, inspect
cached names, then commit with a concrete fix-scope and concise root-cause description.

Expected: only regression, fix, and directly required docs are committed.

- [ ] **Step 8: Restart affected gate**

Run the complete originating task.

Expected: gate green. Repeat Task 12 for every confirmed defect.

---

### Task 13: Final clean-room rerun and coverage proof

**Files:**
- Verify: all campaign commits
- Create outside Git: /tmp/memo-full-qa-20260722/final/*

**Interfaces:**
- Produces: final evidence for the report.

- [ ] **Step 1: Rebuild artifacts and Docker from final HEAD**

Repeat Task 3 artifact steps and Task 10 build steps with empty namespaces.

Expected: all build and isolated imports pass.

- [ ] **Step 2: Re-run structural and automated gates**

Repeat Tasks 3 and 4.

Expected: all pass.

- [ ] **Step 3: Re-run native CLI/MCP/runtime journeys**

Repeat Tasks 6-8 with new homes, data/state paths, sockets, and ports.

Expected: all pass without previous campaign state.

- [ ] **Step 4: Re-run install/distribution and Docker**

Repeat Tasks 9-10, including uninstall/cleanup.

Expected: all pass.

- [ ] **Step 5: Regenerate inventories**

Repeat Task 2 and compare exact names to initial inventory.

Expected: every addition/removal maps to a campaign commit and report finding;
all rows have discovery plus component/domain or real-journey behavior evidence.

- [ ] **Step 6: Prove worktree preservation**

Run:

    git status --short --branch > "$QA_ROOT/final/git-status.txt"
    git diff --binary > "$QA_ROOT/final/git-diff.patch"
    git diff --cached --binary > "$QA_ROOT/final/git-diff-cached.patch"
    cmp "$QA_ROOT/baseline/git-diff.patch" "$QA_ROOT/final/git-diff.patch"
    cmp "$QA_ROOT/baseline/git-diff-cached.patch" "$QA_ROOT/final/git-diff-cached.patch"

Expected: pre-existing patches are byte-identical; campaign work is committed.

---

### Task 14: Report, cleanup, durable memory, and completion

**Files:**
- Create: docs/superpowers/reports/2026-07-22-memo-full-qa.md

**Interfaces:**
- Consumes: every manifest, log, metric, finding, and fix commit.
- Produces: final auditable QA result.

- [ ] **Step 1: Write final report**

Include HEAD range, preserved baseline, environment matrix, exact surface counts,
gate commands/durations/results, user journeys, retrieval/performance metrics,
every finding and fix commit, justified unavailable rows, residual risk, and
cleanup proof.

Expected: no placeholders, unexplained skips, or unsupported claims.

- [ ] **Step 2: Clean campaign resources**

Stop sandbox daemons/servers. Remove QA Docker containers, volumes, networks,
and images. Uninstall QA Homebrew formula. Remove fake clients and sandboxes
after extracting evidence.

Expected: no campaign process, socket, port, service, container, volume, network,
or formula remains.

- [ ] **Step 3: Record cleanup proof**

Capture filtered ps, lsof, Docker, launchctl, and Homebrew output in
"$QA_ROOT/final/cleanup.txt".

Expected: empty resource lists.

- [ ] **Step 4: Commit final report only**

Run:

    git add -f docs/superpowers/reports/2026-07-22-memo-full-qa.md
    git diff --cached --check
    git diff --cached --name-status
    git commit -m "docs: record exhaustive memo full QA"

Expected: only report committed.

- [ ] **Step 5: Persist outcome**

Save final version, surface counts, gates, finding/fix commits, remaining risks,
and report path through memo_save. Run memo_idle_capture and
memo_pop_notification.

Expected: next session can recover complete outcome.

- [ ] **Step 6: Verify 100 percent completion**

Confirm every checkbox is checked, every gate green, every confirmed defect
fixed, final report committed, and cleanup proof empty.

Expected: 100 percent complete. Only then notify the user.
