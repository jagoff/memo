# Show HN

## Title (max 80 chars)

**Recommended:** `Show HN: Memo – Local agent memory that costs fewer tokens, not more`

Alternatives:
- `Show HN: Memo – Local-first memory for AI agents, with time-travel`
- `Show HN: Memo – Agent memory that self-optimizes nightly, 100% on-device`

No emoji, no all-caps, no "blazing-fast", no exclamation marks.

## URL
https://github.com/jagoff/memo

## First comment — post within 60 seconds of submitting

I build with Claude Code every day and the thing that kept killing me wasn't capability — it was amnesia. Every morning I'd re-explain the same architecture decisions to a fresh context window. "We use Postgres, not Mongo." "Don't touch the auth module." Again and again.

I tried the existing memory layers and hit two walls. First, most of them route through a cloud API — my client work can't leave my machine. Second, and less obvious: they made things *worse* on tokens. A memory server's full surface exposes 159 MCP tools and costs ~18k schema tokens in every session, in every client, before you've asked anything.

So memo is built around the opposite constraint. The default MCP profile exposes 38 tools (~3.8k tokens), and the bundled Claude Code hook injects the relevant memory on a ~160-token budget instead of letting the model re-derive what it worked out last week. `memo roi` reads the actual grounding and re-ask ledgers and estimates accumulated savings with disclosed defaults: 350 tokens per grounded recall and 900 per avoided re-ask. I'd rather you inspect the formula and your own ledger than take a corpus-size extrapolation on faith.

Technically: memories are plain Markdown files (source of truth — SQLite is a derived index that rebuilds from them). Hybrid retrieval, vector + BM25 fused with RRF, then an optional cross-encoder rerank. Embedder, reranker, and LLM all run in-process via Apple MLX. Prompts and memories have no outbound path during normal use; the default update tag check can be disabled with `MEMO_AUTO_UPDATE=0`.

Two things I haven't seen elsewhere and would like feedback on:

- **Time-machine.** `memo as-of ask "..." --date 2026-02-01` reverse-replays the history log so you can query the corpus as it existed then. Turns out "what did we know when we made this call?" is a question I ask a lot.
- **Contradiction radar.** When you change your mind about something you saved months ago, memo flags the stale version instead of quietly serving both.

Honest limitations: the good path is macOS on Apple Silicon — MLX is load-bearing for the reranker and the LLM verbs. Linux gets a standalone CPU backend (search/recall/save) via `pipx install "mlx-memo[cpu]"` or Docker, but not the full feature set. And the first install pulls ~8 GB of models, which is a lot to ask before you know if you like it. I'm open to a small-model quickstart if that's the blocker.

MIT. Happy to answer anything, including where it's weak.

## Timing
Tuesday–Thursday, 9:00–10:30 AM ET (11:00–12:30 ART). Never Friday or weekends.
Block 4 uninterrupted hours after posting. Response speed in the first two hours is most of the outcome.
