# Pre-written answers to predictable objections

**"Apple Silicon only? Hard pass."**
Fair. The CPU backend covers search/recall/save on Linux today (`pipx install "mlx-memo[cpu]"`, or Docker). What you lose is the reranker and the LLM verbs, because those lean on MLX. Porting the reranker to a CPU cross-encoder is the obvious next step — if there's demand I'll prioritize it.

**"8 GB of models before I know if I want it?"**
Agreed, that's too much friction. The Docker image gets you to `memo doctor` in seconds. A small-model quickstart profile is on my list — genuinely useful signal if that's the blocker for you.

**"How is this different from putting stuff in CLAUDE.md?"**
CLAUDE.md is loaded in full, every session — a fixed tax that grows forever. memo retrieves only the 1–3 relevant memories per prompt; the bundled Claude Code hook defaults to one memory on a ~160-token budget. Below ~30 facts, CLAUDE.md is honestly fine. Past that it eats your context window and quality drops.

**"Your comparison table is wrong about [project X]."**
Thanks — which cell? I verified against their docs in July 2026 and I'd rather have it accurate than flattering. Open an issue with a link and I'll change it today.
→ Answer FAST and without defensiveness. How you handle a correction is more visible than the correction itself.

**"`curl | bash` is a security anti-pattern."**
Reasonable. `uv tool install mlx-memo`, `pipx install mlx-memo` and `brew tap jagoff/memo && brew install mlx-memo` get you the same two binaries, and install.sh is short enough to read first.

**"Is this maintained or another abandoned weekend project?"**
45 GitHub releases, latest this month, 1,512 commits on `master`. Just say the numbers.

**Jokes about the username.**
`jagoff` is Chicago slang for "jackass" and someone will make the joke. Laugh first — one self-aware line buys goodwill. Never get defensive, never explain it earnestly.
