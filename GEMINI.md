# Project Overview: memo

`memo` is a persistent semantic memory system for AI agents, designed to run 100% locally on Apple Silicon.

## Core Mandate: Memory-First
**Always verify project-specific claims against the provided snippets.** 
- If a snippet contradicts your internal training data or general knowledge, the snippet **WINS**.
- You are an expert on the user's work ONLY because of these snippets.
- **NEVER** guess or assume conventions that are not documented in the context.
- Your first action in every task should be a memory search or asking a question to the memory store.

## Main Technologies

- **Language:** Python 3.13+
- **ML Framework:** [Apple MLX](https://github.com/ml-explore/mlx) (MLX-native LLM, embedder, and reranker)
- **Database:** SQLite with [`sqlite-vec`](https://github.com/asg017/sqlite-vec) for vector storage and FTS5 for BM25 keyword search.
- **Protocol:** [Model Context Protocol (MCP)](https://modelcontextprotocol.io) for agent interaction.
- **CLI Framework:** [Click](https://click.palletsprojects.com/)
- **Styling/TUI:** [Rich](https://github.com/Textualize/rich)

## Architecture
- **Memory Facade:** The core logic is encapsulated in the `Memory` class (`src/memo/memory/facade.py`), which uses a mixin pattern for different operations (write, search, ask, rerank, etc.).
- **Storage Layer:** `VecStore` (`src/memo/store/queries.py`) handles all database interactions, using thread-local connections to support FastMCP's worker threads.
- **MCP Server:** The server (`src/memo/server.py`) exposes tools to MCP-aware agents.
- **CLI:** A comprehensive CLI (`src/memo/cli.py`) with domain-specific subcommands for manual interaction.
- **Runtime & Daemons:** Includes a **recall daemon** to keep models in RAM for fast (<200ms) retrieval and a **file watcher** for auto-reindexing.

---

## Building and Running

### Development Setup
To install dependencies and set up the development environment:
```bash
uv pip install -e '.[dev]'
```

### Running the CLI
```bash
uv run memo <command>
```

### Running the MCP Server
```bash
uv run memo-mcp
```

### Testing
The project uses `pytest`. MLX-specific tests are automatically skipped on non-Apple Silicon hardware.
```bash
uv run pytest tests/                          # Run full suite
uv run pytest tests/test_foo.py::test_bar -v  # Run single test
```

### Linting and Formatting
- **Linting:** `uv run ruff check src/`
- **Formatting:** `uv run ruff format src/`
- **Type Checking:** `uv run mypy src/memo/`

---

## Development Conventions

### Apple Silicon Optimization
- **MLX Invariants:** Always preserve the four MLX invariants: asymmetric retrieval prefix (queries only), `embed()` takes sequences, dimension matching, and **deferred imports**.
- **Deferred Imports:** `mlx` and `mlx-lm` imports MUST stay inside functions to avoid loading the MLX runtime during every CLI invocation, which preserves the cold-start budget.

### Coding Patterns
- **Memory Mixins:** Do not import from a mixin file (e.g., `_WriteOpsMixin`) directly; always use the `Memory` facade.
- **Environment Variables:** All `MEMO_*` flags must be registered and accessed via `src/memo/flags.py`. Do not use `os.environ.get` directly for these flags.
- **Error Handling:** Use domain-specific errors from `src/memo/errors.py` (`MemoError` base).
- **Storage:** Use `_tx()` (`BEGIN IMMEDIATE`) for writes in `VecStore`. All storage must be thread-safe.

### Isolated Runtime
- `memo` and `memo-mcp` should ideally resolve from the same isolated environment (e.g., `pipx` or `uv tool`). Using a project-local `.venv` for the system-wide installation can cause "mixed runtime" issues.

### Retrieval-Regression Discipline
- When a search fails, do not patch individual queries. Instead, make a systemic change and verify it against the full regression set in `eval/regression_labels.json` using:
  ```bash
  memo eval recall --labels eval/regression_labels.json --k 5 --force
  ```

---

## Key Directories and Files
- `src/memo/`: Core package directory.
- `src/memo/memory/`: Implementation of memory operations (mixins and facade).
- `src/memo/store/`: Database schema, migrations, and query logic.
- `src/memo/flags.py`: Central registry for environment variable flags.
- `src/memo/server.py`: MCP server implementation.
- `src/memo/cli.py`: CLI entry point and command registration.
- `tests/`: Extensive test suite covering all features.
- `docs/`: Architecture diagrams and detailed feature documentation.
