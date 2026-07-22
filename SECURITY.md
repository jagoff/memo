# Security policy

## Reporting a vulnerability

If you find a security issue in memo, please **do not open a public issue**.
Submit a [private vulnerability report](https://github.com/jagoff/memo/security/advisories/new)
so the discussion, affected versions, and remediation stay confidential. If
GitHub reporting is unavailable, email <fernandoferrari@gmail.com> with:

- A description of the vulnerability
- Steps to reproduce (or a minimal proof-of-concept)
- The affected version (`memo --version`)
- The impact you observed

I aim to acknowledge reports within 7 days and ship a fix or mitigation
within 30 days for confirmed issues.

## Threat model

memo runs **entirely on your local machine**. On Apple Silicon, embeddings,
reranking, and chat happen in-process via MLX; on Linux/Intel macOS the CPU
backend supports save/search/recall without cloud calls. The data plane is plain
Markdown on disk plus local sqlite files. Concretely:

- **No cloud memory service.** The normal CLI/MCP path uses stdio, local files,
  and optional local Unix sockets for the recall/embedder daemons. `memo http-api`
  and HTTP MCP transport are opt-in localhost surfaces, not hosted services.
- **No telemetry.** memo emits no usage data.
- **Credentials are isolated and opt-in.** Secret storage is disabled unless
  `MEMO_SECRET_STORAGE_ENABLED=1`. When enabled, values are AES-256-GCM sealed
  under a random private master key and excluded from Markdown, indexing,
  recall, generated context, backup, and git sync. Explicit `secret get` and
  `secret export` commands disclose plaintext to the invoking user.
- **Network APIs authenticate by default.** Every `/api/*` route requires
  `Authorization: Bearer <token>`; non-loopback REST and MCP HTTP binds require
  an explicit acknowledgement and cannot disable authentication.

Reasonable threats memo aims to mitigate:

- A malicious `.md` file in the vault should not be able to execute code
  during indexing. (memo only parses YAML frontmatter and embeds plain
  text.)
- A malicious MCP client should not be able to escape the configured
  `MEMO_DATA_DIR` via path traversal.
- A malicious `memo://memory/{id}` resource URI should not return data
  outside the vault.

Out of scope:

- Whoever has write access to your `MEMO_DATA_DIR` can read and modify
  your memories. That is by design — the Markdown vault is the storage
  of record.
- Whoever can execute as your user can register an MCP client that talks
  to memo over stdio. memo trusts the caller.

## Supported versions

Only the latest minor release (currently the `3.12.x` line) receives security
fixes. Older releases will not be backported.
