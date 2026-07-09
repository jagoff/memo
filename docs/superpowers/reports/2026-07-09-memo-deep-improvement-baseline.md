# memo Deep Improvement Baseline

Date: 2026-07-09
Source spec: `docs/superpowers/specs/2026-07-09-memo-deep-improvement-roadmap-design.md`

## Verification

- Full non-slow suite: 3,534 passed, 29 skipped, 6 warnings.
- Coverage: 73.35%.
- Current coverage floor: 68%.
- Known warning class: sqlite `ResourceWarning: unclosed database`.

## Operational Health

- `memo health`: 5,982 memories, 949 archived, 693.5MB corpus, no warnings.
- Isolated doctor: passed with `/Users/fer/.local/bin/memo doctor --strict-runtime`.
- Dev-mode doctor: `uv run --no-sync memo doctor --strict-runtime` reports project `.venv` mode by design.

## Memory Utility

- Consults sampled: 437.
- Recall-hook hit rate: 98.7%.
- Strong-hit rate: 97.4%.
- Grounded rate: 0.371.
- Referenced rate: 0.009.
- Known gaps: 1.

## Corpus Lint

- `legacy_extra`: 0.
- `few_tags`: 2,635.
- `body_skinny`: 242.
- `untitled`: 0.

## Latency

- Daemon p50: 655ms.
- Daemon p95: 6,803ms.
- Daemon p99: 8,152ms.
- Subprocess p50: 8,896ms.
- Subprocess p95: 10,668ms.
- Subprocess p99: 11,572ms.

## Code Surface

- `src/memo`: about 86k Python lines.
- Top-level CLI/server modules: 113.
- CLI modules: 74.
- Server modules: 39.

## Source-Debt Counts

- Broad `except Exception`: 516 sites across 142 source files.
- Silent `pass`: 79 sites across 55 source files.
- Raw `os.environ.get("MEMO_*")`: 17 sites across 9 source files.
- todo/fixme/hack markers: 7 sites across 4 source files.

## Low-Coverage Risk Areas

- `src/memo/semantic_relations.py`: 0%.
- `src/memo/runtime/daemon.py`: 20%.
- `src/memo/memory/secret_ops.py`: 23%.
- `src/memo/cli_contradict.py`: 24%.
- `src/memo/cli_transcripts.py`: 24%.
- `src/memo/synapse_backend.py`: 24%.
- `src/memo/server_session_patterns.py`: 33%.
- `src/memo/server_idle_capture.py`: 36%.
- `src/memo/llm.py`: 37%.
- `src/memo/runtime/update.py`: 44%.

## Eval Gate

- Latest pre-push gate: 238 searches.
- Precision gate: `prec@k 0.884 >= 0.877`.
- Noise gate: `noise@k 0.000 <= 0.000`.
