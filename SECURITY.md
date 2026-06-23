# Security policy

## Reporting a vulnerability

If you find a security issue in memo, please **do not open a public issue**.
Instead, email <fernandoferrari@gmail.com> with:

- A description of the vulnerability
- Steps to reproduce (or a minimal proof-of-concept)
- The affected version (`memo --version`)
- The impact you observed

I aim to acknowledge reports within 7 days and ship a fix or mitigation
within 30 days for confirmed issues.

## Threat model

memo runs **entirely on your local Mac**. There are no network calls in
the hot path: embeddings, reranking, and chat all happen in-process via
MLX. The data plane is plain Markdown on disk plus three local sqlite
files. Concretely:

- **No remote endpoints.** memo opens no sockets and accepts no
  inbound connections.
- **No telemetry.** memo emits no usage data.
- **No credentials.** memo does not request, store, or transmit API keys.

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

Only the latest minor release (currently the `1.0.x` line) receives
security fixes. Older releases will not be backported.
