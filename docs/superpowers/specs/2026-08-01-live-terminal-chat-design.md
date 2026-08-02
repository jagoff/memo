# Live Terminal Chat Design

## Goal

Turn Memo's asynchronous handoff/attention coordination into an optional
same-machine live channel. A Memo MCP client or CLI user can address one
explicitly registered agent terminal, place a prompt in its input buffer, and
submit it as if Return had been pressed. The receiver can reply through the
same channel without polling a journal.

## Chosen approach

Memo will add a narrow local terminal bridge. This is preferred over reviving
the closed Memflow/Synapse mesh because it has a smaller trust boundary and
does not depend on the unsafe migration, remote identity, or distributed
delivery code from PR #157. It is preferred over polling handoffs because
polling cannot release a terminal that is already blocked on interactive
input.

Two alternatives were rejected:

- A background handoff poller remains asynchronous and cannot emulate Return.
- A network terminal-control service would enlarge the authentication and
  remote-code-execution boundary beyond the requested same-machine workflow.

## Trust and registration

The operating-system user is the security boundary. A registration contains a
random terminal id, canonical `/dev/...` TTY, PID, process start marker, agent
name, terminal application, project directory, and timestamps. The TTY must be
a character device owned by the current UID, and the PID must belong to that
UID and be attached to the same TTY. Registrations live in a mode-0600 SQLite
sidecar under `Config.state_dir`; they are operational state, not durable
memory.

The Memo agent shim registers itself before `exec`, preserving the PID across
the exec. Every list/send operation revalidates the PID start marker and TTY,
and stale registrations are removed. Delivery is refused unless the registered
agent owns the TTY foreground process group, preventing input from landing in
a child shell or unrelated command.

## Delivery

`TerminalBridge.send()` accepts an exact terminal id, a UTF-8 message no larger
than 16 KiB, an optional sender id, an idempotency key, and a submit flag. It
removes terminal control sequences and forbidden control bytes while
preserving printable Unicode, tabs, and newlines. Return is never accepted as
message content; submit appends exactly one carriage return.

The bridge first tries `TIOCSTI`, which writes into the selected terminal's
input queue without changing focus. On macOS denial it falls back to an exact
TTY match through Terminal.app, iTerm/iTerm2, or Ghostty AppleScript. Ghostty
submission is a separate Return event after the target terminal is focused,
matching the behavior required by raw-mode Codex TUIs.

Each attempt creates a receipt with `delivered`, `failed`, or `duplicate`
status. A repeated idempotency key returns the original receipt and never types
the message twice. `enter()` uses the same validation and receipt path but
delivers only a carriage return.

## User surfaces

The root CLI gains a `terminal` group:

- `memo terminal register --agent codex --tty /dev/ttys003 --pid 123`
- `memo terminal list --json`
- `memo terminal send --to <id> --message <text> [--submit/--no-submit]`
- `memo terminal enter --to <id>`
- `memo terminal history --json`

The always-on agent MCP profile gains `memo_terminal_list`,
`memo_terminal_send`, and `memo_terminal_enter`. A live message is wrapped with
the sender terminal id and a concise instruction explaining how to reply with
`memo_terminal_send`; the body remains clearly delimited as sender content.
Handoffs and attention remain available for durable/asynchronous coordination.

## Error behavior

Invalid or stale targets, wrong UID/TTY associations, non-foreground agents,
oversized messages, control-only messages, and unsupported terminal apps raise
Memo domain errors. CLI reports a nonzero error without a traceback. MCP tools
return structured status and safe error text; message bodies are never copied
into errors or logs.

## Verification

Tests cover registry validation and pruning, sanitization, idempotency,
foreground refusal, receipt persistence, CLI behavior, MCP schemas, shim
registration, and macOS fallback selection. A PTY end-to-end test proves text
and carriage-return delivery through the real input queue without touching a
user terminal. Final verification follows repository CI order and includes a
controlled live exchange with a separately registered Codex terminal.
