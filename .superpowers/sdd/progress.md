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

---

# Fase B — HyPE (question-space index)

**Plan:** docs/superpowers/plans/2026-07-13-hype-question-index.md (H1-H8)
**Start:** 2026-07-13 · **Branch:** master (shared tree) · **Base:** db196ee (v3.4.0)

## Tasks

- [x] H1: flags — COMPLETE (f08b545..ac5f291, review clean)
- [x] H2: HypeStore — COMPLETE (ac5f291..feec06c, review clean; nota H5: knn k capea post-colapso)
- [x] H3: dream_hype — COMPLETE (feec06c..1b4f994, 1 fix round: per-item isolation embed/persist; review approved)
- [x] H4: wiring dream — COMPLETE (1b4f994..032739d, review clean, clone fiel del template chronicle)
- [x] H5: fold — COMPLETE (032739d..4e07691, review clean; punto único=search_ops vec branch verificado 3 paths; PRE-FLIP: type-filter en candidatos fold + Memory.close de _hype_store)
- [x] H6: eval config K — COMPLETE (4e07691..1520898, review clean; perfil hype=[A,K], expensive lockeado a J)
- [x] H7: memo hype status — COMPLETE (1520898..effab4a, review clean)
- [x] H8: COMPLETE — NO FLIP: curated A 0.812 vs K 0.829 / harvested A 0.133 vs K 0.087 (cobertura 10.6%, 474/4466, dilución parcial); gate PASS 0.835; MEMO_HYPE_ENABLED queda OFF; recomendación: encender MEMO_DREAM_HYPE_ENABLED nightly y re-medir a cobertura mayoritaria. HYPE PLAN DONE db196ee..effab4a

---

# Fase B2 — MCPB Node bootstrap

**Plan:** docs/superpowers/plans/2026-07-13-mcpb-node-bootstrap.md (N1-N6)
**Start:** 2026-07-13 · **Branch:** master · **Estado:** plan commiteado; SDD arranca cuando cierre HyPE (no implementers paralelos)

- [x] N1: bootstrap.js — COMPLETE (aae2ae4 en worktree-agent-a3c18c4c3a92408b1, review clean; cherry-pick pendiente)
- [x] N2: manifest node — COMPLETE (d1a3efc en worktree, review clean; __dirname verificado vs spec oficial)
- [x] N3: build_mcpb_node — COMPLETE (bb0dfeb en worktree, review clean; _build_zip DRY con byte-identity probada)
- [x] N4: pin-chain sync — COMPLETE (2ce25d3 en worktree, review clean)
- [x] N5: release bump — COMPLETE (d7d01af en worktree, review clean; follow-up a N6: release check no valida manifest node)
- [ ] N6: gate final + smoke manual Desktop (Fer)

## Fase B — Final
- [x] N6 + fix release-check: COMPLETE (871b897→cherry-picked 8b8fc05; bundle memo-node.mcpb determinístico, NO trackeado — .gitignore packaging/*.mcpb, decisión humana con add -f)
- [x] Review final whole-branch (fable, 17 commits db196ee..8b8fc05): 1 Important pre-push (isPinInstalled matcheaba `==` pero uv tool list imprime `v` → reinstall por arranque + rotura offline) → fix 576a892. PRE-FLIP confirmados sin empeorar: type-filter en candidatos fold + Memory.close _hype_store.
- ANÁLISIS ESTRUCTURAL (pre-flip, no re-litigar sin leerlo): el fold compara escalas incomparables — preguntas embebidas CON prefijo (query↔query) vs docs sin prefijo → question-scores sistemáticamente más altos; a cobertura parcial desplaza hits correctos no cubiertos (harvested −0.046 @ 10.6%), a cobertura total converge a ranking pregunta-only. Opciones pre-flip: (a) embeber preguntas RAW, (b) fold en rank-space (RRF). Re-medir tras elegir.
- FASE B DONE: db196ee..576a892 (18 commits). HyPE dark (MEMO_HYPE_ENABLED OFF, NO FLIP por gate b); MCPB Node listo para smoke manual de Fer.

---

# memo organic traction launch — July 2026

**Plan:** docs/superpowers/plans/2026-07-18-memo-organic-traction-launch.md
**Start:** 2026-07-18
**Branches:** launch/2026-07-memo in memo and memo-web
**Base:** memo 7ff4ae68; memo-web origin/main 18beb655

## Tasks

- [x] Task 1: Establish the launch control plane and account inventory — COMPLETE (7ff4ae68..4704bdce, review clean; live account access remains correctly `blocked_user` except GitHub; launch date 2026-07-28)
- [x] Task 2: Prove product and landing readiness — COMPLETE (4704bdce..5e181518, review approved; product gate remains `failed` only because no local Docker-compatible runtime is installed; five other rows ready)
- [x] Task 3: Capture current, citable claims — COMPLETE (5e181518..b4eb6861, 1 fix round: Claims gate remains `not_checked` until exact 2026-07-27 recapture; re-review clean)
- [x] Task 4: Produce and verify the visual evidence pack — COMPLETE (memo-web 18beb655..9f479f9; memo b4eb6861..0e979b37; user batch approval; review clean)
- [ ] Task 5: Write and approve channel-native launch copy — IN PROGRESS
- [ ] Task 6: Prepare platform drafts without publishing
- [ ] Task 7: Pass the final readiness gate and capture the baseline
- [ ] Task 8: Execute the coordinated launch day
- [ ] Task 9: Operate the first 24 hours and approved contingencies
- [ ] Task 10: Run the Spanish wave and day-three review
- [ ] Task 11: Close D5/D7 measurement and publish the postmortem

---

# memo Proactive Engine — v1

**Plan:** docs/superpowers/plans/2026-07-21-memo-proactive-engine.md
**Start:** 2026-07-21 · **Branch:** feat/proactive-engine · **Base:** 82f11db0

## Tasks
- [ ] Task 1: Flag registry
- [ ] Task 2: Nudge model
- [ ] Task 3: ProactiveStore
- [ ] Task 4: Arbiter
- [ ] Task 5: Reliability detector
- [ ] Task 6: Continuity detector
- [ ] Task 7: Surfaces
- [ ] Task 8: Engine + push_gate
- [ ] Task 9: memo digest CLI + feedback
- [ ] Task 10: Acted-detection
- [ ] Task 11: Dream integration
- [ ] Task 12: Briefing/statusline/hook wiring
- [ ] Task 13: dream_flags gate + empty-corpus contract
Task 1: complete (commit c6d5d216, review clean; dream_flags gate pulled forward → Task 13 reduces to empty-corpus test)
Task 2: complete (commit 27dbfa92, review clean — spec PASS, quality approved)
Task 3: complete (commit 6392e378, review clean — multiplier clamp + TTL/snooze/dismiss verified)
Task 4: complete (commit 7a3be4ce, review clean — score/route/urgent-gating verified)
NOTE for Task 11: build Memory.superseded_pairs() there — scan inactive/*.md for extra.superseded_by frontmatter (bounded, dream-only, not recall path). Task 5 detector is interface-driven and works once wired.
Task 5: complete (commit 32a39636, review clean — detector spec ✅; superseded_pairs() accessor deferred to Task 11)
Task 6: complete (commit b81c38d2, review clean — detector + real open_loops thin wrapper verified)
COLLISION FIX (commit fb20c033): absorbed legacy src/memo/proactive.py -> proactive/suggester.py + re-export in __init__ (module/package name clash shadowed ProactiveSuggester). OLD test_proactive.py 11 pass, NEW suites 12 pass. FLAG for user veto at final review (rename-vs-absorb).
Task 7: complete (commit 329333fe, review clean — pure surface renderers, spec ✅)
Task 8: complete (commit 3d6e1285, review clean — push_gate + compute_routed; 5 flag defaults verified exact)
Task 9: complete (commit 9161f980, review clean — digest+dismiss/snooze; NOTE: `ignored` outcome deferred to v1.1 — needs last-shown primitive)
NOTE for Task 12/final: reliability detector action="memo review <id>" but NO `memo review` command exists. Either create it, or repoint the action to an existing command (memo get / memo invalidate). Acted-detection helper (Task 10) matches whatever action string is set.
Task 10: complete (commit bbe9dd23, review clean — acted helper; wiring deferred, no memo review cmd exists)
Task 10.5 BRANCH-FIX (commit 0b497949): fixed 2 regressions from my branch — test_prompt_overrides path (proactive.py→proactive/suggester.py, from collision-fix) + README CLI inventory (added digest, 125→126, from Task 9). Both suites green.
Task 11: complete (commit 3328fdea, review clean — superseded_pairs scan inactive/*.md + dream pass flag-gated; failure→receipt[errors])
Task 12: complete (commit ff93bd4a, review clean — briefing+Stop-urgent wired flag-gated+guarded; statusline badge DEFERRED to v1.1 (bash statusline needs snapshot-file); action repointed memo review→memo get. Full suite 4881 pass 0 fail)
Task 13: complete (commits bd2fecbe empty-corpus test + 09929f59 quality baseline ratchet; all 6 CI gates PASS — pytest/ruff/format/mypy/quality_gate/config-validate). ALL 13 TASKS DONE.
FINAL REVIEW (opus, whole-branch 16 commits): READY TO MERGE default-OFF, 0 Critical, 4 Important pre-graduation + 5 Minor.
FINAL FIX (commit 37b7a30c): I1 acted wired into `memo get`; I2 urgent on raw score + 30d feedback window; I3 except Exception; I4 lazy suggester re-export; M1/M2/M3 done. Full suite 4890 pass 0 fail, all gates green. PROACTIVE ENGINE v1 DONE.

RELEASE v3.9.0 SHIPPED: PR#54 merged abcc724e + tag v3.9.0 pushed (dispara auto-update). Contiene #50 int8 + #52 int8-fix + #51 proactive v1 + #53 v2 detectors. Traps release: mcpb bundles no auto-rebuild (memo release mcpb), SECURITY.md 3.8.x→3.9.x. Follow-up: formula Homebrew URL post-PyPI. SESIÓN COMPLETA.

---

# Testing System Hardening

**Plan:** docs/superpowers/plans/2026-07-22-testing-system-hardening.md
**Start:** 2026-07-22 · **Branch:** master (shared checkout, user-approved explicit-path staging) · **Base:** b6919a5a

## Tasks
- [ ] Task 1: Testing dependencies, markers, and generated-artifact boundaries
- [ ] Task 2: Opt-in resource-hygiene pytest plugin
- [ ] Task 3: Clean current resource owners and broad-exception baseline
- [ ] Task 4: Stateful Hypothesis model for VecStore
- [ ] Task 5: Branch-aware aggregate and changed-lines coverage gates
- [ ] Task 6: Scheduled randomized and repeated stability workflow
- [ ] Task 7: Scoped scheduled mutation testing
- [ ] Task 8: End-to-end verification and ratchet documentation
Task 1: complete (commits b6919a5a..c2bd0ae2, review clean — dependency extras, strict markers, ignores, and universal lock verified)
Task 2: complete (commits 120e5958..4ecd695b, review clean after socket-matcher fix — opt-in leak detector and pytester isolation verified)
Task 3: complete (commits 738d1fbc..c89966cd, review approved — 85 serial resource tests green; Minor for final review: housekeeping fixture insert closes without explicit commit)
Task 4: complete (commits 66ab8af6..cf4f8354, review clean after three oracle fixes — 100×75 extended profile green)
Task 5: complete (commit d2951663, review approved — branch total 48,614/64,750=75.0795%, floor 74, PR diff gate 90%; Minor for final: workflow contract uses broad substrings)
Task 6: complete (commit 1b5fc898, review clean — 148 tests selected, bounded 60-test run green, scheduled/manual replayable workflow approved)
Task 7: complete (commits 6e10e76a..4d305c60, review clean after six gate fixes — 49 contracts + 126 baseline tests green; mutmut 3.6 schema/status propagation verified)
Task 7.5: complete (commits 4d305c60..2f9172c8, final review zero findings — scoring/replay/housekeeping baseline integrated; atomic vacuum race + malformed timestamps resolved)
