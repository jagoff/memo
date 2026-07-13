# Wave 1 Implementation Progress

**Plan:** docs/superpowers/plans/2026-07-07-memo-token-economy-wave1.md  
**Start:** 2026-07-07  
**Branch:** worktree-token-economy-spec

## Tasks

- [x] Task 1: L1 Crusher Infrastructure (Scoring + Cache Module) — COMPLETE
      Commits: f500407, 8b04193
      Files: src/memo/store/crush_cache.py, src/memo/flags_capture.py, tests/test_token_economy_wave1.py
      Status: CrushCache class, crush_marker, 3 flags, 11 tests, spec compliance ✅
- [ ] Task 2: L1 JSON Crushing on Ingest + Retrieval Command — IN PROGRESS
      Implementer: ac0e676ff5faa9c30 (started: 2026-07-07)
- [ ] Task 3: L4 Verbosity Steering on Recall Output
- [ ] Task 4: Integration Testing + Token Measurement Gate

---

# Memory Quality Loop Progress

**Plan:** docs/superpowers/plans/2026-07-08-memory-quality-loop.md
**Start:** 2026-07-08
**Branch:** master

## Tasks

- [x] Task 1: Quality Signals And Pure Reranker — COMPLETE
      Commits: b6e9603..2d30432
      Status: quality module + default-off flags, 6 focused tests, review clean
- [x] Task 2: Wire Quality Rerank Into Explicit Search — COMPLETE
      Commits: 2d30432..a954f9b
      Status: explicit CLI/MCP/ask opt-in, recall hot path untouched, review clean
- [x] Task 3: Context Pack Builder — COMPLETE
      Commits: a954f9b..ea2c19c
      Status: ask/ask_stream context packs, chat opt-out, sensitive filtering, review clean
- [x] Task 4: Explicit CLI And MCP Context-Pack Surface — COMPLETE
      Commits: ea2c19c..3cd0718
      Status: CLI/MCP gated read-only context-pack surfaces with consult logging, review clean
- [x] Task 5: Quality Compaction Preview — COMPLETE
      Commits: 3cd0718..5d71096
      Status: default-off read-only preview, strict scope/target validation, review clean
- [x] Task 6: Quality Compaction Apply And Undo Receipt — COMPLETE
      Commits: 5d71096..56619df
      Status: explicit apply with atomic receipts and undo rollback coverage, review clean
- [x] Task 7: Evaluation Metrics And Final Verification — COMPLETE
      Commits: 56619df..a642833
      Status: additive eval metrics + changelog, mypy/focused/full tests green; recall eval gate hung with no output and remains residual verification risk

---

# Memo Deep Improvement First Sprint Progress

**Plan:** docs/superpowers/plans/2026-07-09-memo-deep-improvement-first-sprint.md
**Start:** 2026-07-09
**Branch:** master

## Tasks

- [x] Task 1: SQLite Resource Hygiene Guard - COMPLETE
      Commits: 46e4c0b..4952796
      Status: focused sqlite ResourceWarning guards, reproduction passed before/after, review clean
- [x] Task 2: Source Audit Helper For Flags And Exceptions - COMPLETE
      Commits: 4952796..124449c
      Status: dev_audit helper + policy tests/doc, review clean after corrected brief
- [x] Task 3: Eval Recall Profile Source Of Truth - COMPLETE
      Commits: 124449c..a2ebfa3
      Status: named recall eval profiles + CLI --profile, local pre-push hook updated, review clean
- [x] Task 4: Baseline Audit Report - COMPLETE
      Commits: a2ebfa3..895aff2
      Status: baseline report committed, draft-marker check clean
- [x] Opportunistic Fix: Recall eval config help - COMPLETE
      Commits: 7690ffa
      Status: --config help now advertises A-J, focused help test and ruff passed
- [x] Opportunistic Fix: Memory sqlite lifecycle - COMPLETE
      Commits: 4b8180b
      Status: Memory finalizer closes sqlite handles and close() closes all closeable capabilities
- [x] Task 5: Sprint Verification - COMPLETE
      Status: CI-order verification passed: ruff clean, mypy clean, focused sprint tests 74 passed, targeted ResourceWarning-as-error checks passed, quick recall eval smoke passed, full non-slow suite 3542 passed / 29 skipped / 6 baseline sqlite ResourceWarnings with 73.43% coverage
      Notes: strict runtime doctor passed except for the expected local SDD working-tree warning before this bookkeeping commit; serial ResourceWarning attribution run was interrupted as too slow after 303 passing tests with no warning failure

---

# Fase A — Chronicle + memo onboard

**Plans:** docs/superpowers/plans/2026-07-13-chronicle-dream-pass.md (C1-C9) + 2026-07-13-memo-onboard.md (O1-O4)
**Start:** 2026-07-13
**Branch:** master (shared tree — explicit-path staging only)
**Base:** f6ba622

## Tasks

- [x] C1: Chronicle flags — COMPLETE (f6ba622..d8416c4, review clean; Minor: test podría chequear flag_bool accessor)
- [x] C2: Module base — COMPLETE (d8416c4..31adc08, review clean)
- [x] C3: Provenance filter — COMPLETE (31adc08..d93c88d, review clean; Minor: sin caso test bullet `*`, trailing-newline note)
- [x] C4: Day-facts collection — COMPLETE (d93c88d..86f8422, 1 fix round: frontmatter window + YAML title; re-review clean; Minor pendiente: cap 2000 chars head)
- [x] C5: LLM narrate + run_chronicle_pass — COMPLETE (86f8422..1cda6e3, review clean; Minors: noop init muerto, QUIET_DAY exact-match)
- [x] C6: Weekly rollup — COMPLETE (1cda6e3..0020be4, review clean; Minor: falta test mixed-week selectivity)
- [x] C7: Pipeline wiring + subcommand — COMPLETE (0020be4..709ec9e, review clean; brief tenía import faltante, fixer lo agregó disclosed)
- [x] C8: memo chronicle reader — COMPLETE (709ec9e..fbb1799, 1 fix round: isort I001; Minors: sin validación formato --date, mensaje sugiere flag aunque falte fecha puntual)
- [x] C9: Final gate — COMPLETE (0ccd43e mypy fix; full suite 4066 pass, 6 fails PRE-EXISTENTES no-Chronicle: config real leak en test_autoupdate/test_config/test_server_startup_policy; config validate ok). CHRONICLE PLAN DONE f6ba622..0ccd43e
- [x] O1: flag — COMPLETE (0ccd43e..3874c16, review clean)
- [x] O2: _recent_memories — COMPLETE (3874c16..b88d625, 1 fix round: frontmatter-first title, patrón C4; Minor cerrado: removeprefix)
- [x] O3: onboard command — COMPLETE (b88d625..90dd5f2, review clean; Minors: const 90 duplicada vs registry, parse JSON test frágil)
- [x] O4: Final gate — COMPLETE (sin commits; full suite 4075 pass, 7 fails pre-existentes conocidos, 0 de onboard). ONBOARD PLAN DONE 0ccd43e..90dd5f2

## Final
- [x] Whole-branch review (fable): 1 Critical (_global bucket excluido en scanners — mataba Day-0 de onboard y subcontaba chronicle) + 2 Important (dry-run mutaba settings.json; whitelist/regex citas + piso provenance) → fix 7d276aa (32/32, full suite 4083 pass/6 pre-existentes) → re-review: Ready to merge YES.
- FASE A DONE: f6ba622..7d276aa (16 commits). Push/tag/release: decisión de Fer, no implícito.
