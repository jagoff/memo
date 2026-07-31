# Memo Secret Storage System — Design Spec

> Security amendment (2026-07-14): the implementation intentionally does not
> create the metadata markdown proposed below. Storage is explicit opt-in,
> ciphertext lives only in `secret_store`, and a random 256-bit local master key
> replaces hostname/device-derived key material. Legacy markdown is excluded
> from reindex, backup, recall context, and git sync, and is removed on delete.
> The historical sections remain as design context; this amendment and the
> current code are authoritative where they differ.

**Date:** 2026-07-07  
**Status:** Design (pending implementation)  
**Scope:** Local-first encrypted credential storage (passwords, API keys, SSH keys, DB credentials)

---

## Executive Summary

memo gains the ability to store and manage secrets (passwords, tokens, API keys, SSH keys, DB credentials) **locally and encrypted at rest**. Secrets are:
- Detected automatically via regex + LLM heuristics during capture
- Encrypted with AES-256-GCM (device-bound key derivation)
- Stored in markdown + indexed in SQLite
- Accessed via a warm daemon + MCP interface
- Never synced to remote; strictly local

This reverses the blanket "never store secrets" rule — memo can now be a **trusted local credential store** when secrets stay offline and encrypted.

---

## 1. Requirements & Constraints

### Functional Requirements

1. **Store all credential types:** API tokens, passwords, SSH keys, database credentials, certificates
2. **Automatic detection:** Regex patterns + LLM classification (high confidence)
3. **User confirmation:** Prompt before saving ("Found possible secret. Save encrypted? [Y/n]")
4. **Transparent decryption:** Applications (synapse, agents) retrieve secrets via daemon without user re-auth
5. **Offline-only:** No sync to remote; local encryption only
6. **CLI commands:** save, get, list, delete, export (as env vars)
7. **MCP interface:** synapse/agents query via `memo_get_secret`, `memo_list_secrets`
8. **Audit trail:** Access logged to grounding.log (who accessed what, when, how many times)

### Non-Functional Requirements

1. **Security:** AES-256-GCM at rest; device-bound key; socket-only access (not stdout)
2. **Performance:** Sub-millisecond daemon response (LRU cache, warm socket)
3. **Simplicity:** Device key auto-derived; no user passphrase prompt
4. **Durability:** Secrets survive daemon restart (encrypted on disk, re-loaded from sqlite)
5. **Isolation:** Secrets in separate table; not exposed in recall/briefing

### Constraints

- **No cross-machine sync:** Secrets are machine-local, not git-synced
- **No export to stdout:** `memo get-secret` uses daemon socket; never prints to terminal
- **No decryption in child processes:** Only daemon decrypts; prevent key leakage
- **Markdown is source of truth:** Encrypted blob stored in markdown + sqlite index

---

## 2. Architecture

### 2.1 High-Level Flow

```
┌─ Input (CLI or transcript) ─────────────────┐
│ "my API key is sk_live_12345..."            │
└────────────────────┬──────────────────────┘
                     │
         ┌─ Detect (regex + LLM) ─────┐
         │ - Fast: regex patterns       │
         │ - Confirm: LLM classification│
         └─────────────┬────────────────┘
                       │
         ┌─ User Confirms (interactive) ┐
         │ "Found secret. Save? [Y/n]"  │
         └─────────────┬────────────────┘
                       │
    ┌──── Encrypt (AES-256-GCM) ───────┐
    │ key = PBKDF2(hostname+device_id) │
    │ nonce = random 96 bits            │
    │ ciphertext = encrypt(value, key)  │
    └────────────────┬──────────────────┘
                     │
    ┌─ Write ────────┴──────────┐
    │ markdown: secrets/YYYY/MM/ │
    │ sqlite: secret_store table │
    └────────────────┬──────────┘
                     │
    ┌─ Daemon ───────┴──────────────────┐
    │ memo-secret-daemon (warm, optional)│
    │ Unix socket: ~/.memo/secret.sock   │
    │ LRU cache (max 100 decrypted keys) │
    │ TTL: 1h default                    │
    └────────────────┬──────────────────┘
                     │
    ┌─ Access ───────┴────────────────┐
    │ - CLI: memo get-secret --name X  │
    │ - MCP: memo_get_secret(name)     │
    │ - Synapse/agents via MCP         │
    └──────────────────────────────────┘
```

### 2.2 Data Model

#### New Tier in `src/memo/tiers.py`

```python
DURABLE_TYPES: frozenset[str] = frozenset(
    {
        "decision",
        "fact",
        "bug",
        "feedback",
        "preference",
        "note",
        "manual",
        "synthesis",
        "procedure",
        "failure_pattern",
        "secret",  # NEW
    }
)

SECRET_KINDS: frozenset[str] = frozenset(
    {
        "api_token",
        "password",
        "ssh_key",
        "db_credential",
        "certificate",
        "generic",
    }
)
```

#### Historical Markdown Format (not created by current releases)

```markdown
---
id: sec_abc123def456
type: secret
kind: api_token
name: OpenAI API Key
created: 2026-07-07T15:30:00Z
tags: [openai, production, gpt4]
detection_method: llm
confidence: 0.95
---

[ENCRYPTED: AES-256-GCM]
```

Current releases create no markdown record. Old `[ENCRYPTED: AES-256-GCM]`
markers contain metadata only and are treated as migration artifacts.

#### SQLite Schema (`src/memo/store/schema.py`)

```sql
CREATE TABLE secret_store (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN (
        'api_token', 'password', 'ssh_key', 'db_credential', 'certificate', 'generic'
    )),
    encrypted_blob BLOB NOT NULL,
    nonce BLOB NOT NULL,
    created_at TEXT NOT NULL,
    accessed_at TEXT,
    accessed_count INTEGER DEFAULT 0,
    detection_method TEXT CHECK (detection_method IN ('regex', 'llm', 'manual')),
    confidence REAL CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

CREATE INDEX idx_secret_name ON secret_store(name);
CREATE INDEX idx_secret_kind ON secret_store(kind);
```

Migration adds this table in `src/memo/store/migrations.py`.

### 2.3 Encryption & Key Derivation

#### Historical Key Derivation (read-only migration compatibility)

**File:** `src/memo/secret_store.py`

```python
def _load_or_create_machine_salt() -> str:
    """Persist a random salt per machine at ~/.memo/machine.salt."""
    salt_path = Path.home() / ".memo" / "machine.salt"
    if salt_path.exists():
        return salt_path.read_text(encoding="utf-8").strip()

    salt = secrets.token_hex(16)
    salt_path.parent.mkdir(parents=True, exist_ok=True)
    salt_path.write_text(salt, encoding="utf-8")
    salt_path.chmod(0o600)
    return salt


def derive_secret_key() -> bytes:
    """Device-bound key: hostname + device_id + machine salt."""
    from consciousness_contracts.uri import device_id as get_device_id
    import socket

    hostname = socket.gethostname()
    device_id = get_device_id()
    machine_salt = _load_or_create_machine_salt()

    material = f"{hostname}:{device_id}:{machine_salt}".encode("utf-8")

    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend

    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"memo_secret_v1",
        iterations=100000,
        backend=default_backend(),
    )
    return kdf.derive(material)
```

#### Encryption/Decryption (AES-256-GCM)

```python
def encrypt_secret(value: str) -> tuple[bytes, bytes]:
    """Encrypt a secret value. Returns: (ciphertext, nonce)."""
    import secrets
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = derive_secret_key()
    nonce = secrets.token_bytes(12)

    cipher = AESGCM(key)
    ciphertext = cipher.encrypt(nonce, value.encode("utf-8"), None)

    return ciphertext, nonce


def decrypt_secret(ciphertext: bytes, nonce: bytes) -> str:
    """Decrypt a secret value."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = derive_secret_key()
    cipher = AESGCM(key)

    plaintext = cipher.decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")
```

---

## 3. Detection (Regex + LLM)

### 3.1 Fast Heuristic (Regex)

```python
HEURISTIC_PATTERNS: dict[str, re.Pattern] = {
    "api_token": re.compile(
        r"(sk_[a-z0-9]{20,}|token[=:]\s*[a-z0-9]{32,}|api[_-]?key[=:]\s*\S+)", re.IGNORECASE
    ),
    "password": re.compile(r"(password[=:]\s*\S+|passwd\s*:\s*\S+|pwd[=:]\s*\S+)", re.IGNORECASE),
    "ssh_key": re.compile(r"(-----BEGIN [A-Z]+ PRIVATE KEY|-----BEGIN RSA PRIVATE KEY)"),
    "db_credential": re.compile(
        r"(postgres://|mysql://|mongodb://|user[=:]\s*\w+.*password[=:]\s*\S+)"
    ),
}


def detect_secrets_heuristic(content: str) -> list[tuple[str, float]]:
    """Fast regex-based detection. Returns [(kind, confidence), ...]. 0.7 = heuristic match."""
    found = []
    for kind, pattern in HEURISTIC_PATTERNS.items():
        if pattern.search(content):
            found.append((kind, 0.7))
    return found
```

### 3.2 LLM Confirmation (Optional, High Confidence)

```python
def detect_secrets_llm(
    content: str,
    heuristic_matches: list[tuple[str, float]],
    llm_instance,
) -> list[tuple[str, float]]:
    """Ask LLM to confirm/refine heuristic matches."""
    if not heuristic_matches or not flag_bool("MEMO_DETECT_SECRETS_LLM"):
        return heuristic_matches
    
    prompt = f"""
    Analyze the following text snippet for secrets (API keys, passwords, tokens, credentials).
    
    For each potential secret, respond with:
    - kind: api_token | password | ssh_key | db_credential | certificate | generic
    - confidence: 0.0-1.0 (1.0 = definitely a secret)
    
    Text:
    {content[:600]}
    
    Format your response as JSON:
    [
        {{"kind": "...", "confidence": 0.95}},
        ...
    ]
    """
    
    response = llm_instance.chat([{"role": "user", "content": prompt}])
    try:
        parsed = json.loads(response)
        return [(item["kind"], item["confidence"]) for item in parsed]
    except (json.JSONDecodeError, KeyError):
        return heuristic_matches
```

---

## 4. CLI Commands (`src/memo/cli_secret.py`)

```bash
# Save secret (explicit)
memo save --type secret --kind api_token --name openai "sk_live_..."

# Get secret (decrypts via daemon)
memo get-secret --name openai

# List secrets (names only)
memo list-secrets --kind api_token

# Delete secret
memo forget-secret openai

# Export all as env vars (interactive confirm)
memo secret-export --format env
```

---

## 5. MCP Interface (`src/memo/server_secret.py`)

```python
@server.tool()
def memo_get_secret(name: str) -> dict[str, str]:
    """Retrieve a secret by name (decrypted). Called by synapse/agents."""
    daemon = SecretDaemon()
    value = daemon.query_socket("get", name=name, client=server.client_name)
    return {"value": value, "kind": "secret"}


@server.tool()
def memo_list_secrets(kind: str | None = None) -> list[dict]:
    """List secrets (metadata only, no values)."""
    secrets = memory.list_secrets(kind=kind)
    return [{"name": s.name, "kind": s.kind, "accessed_count": s.accessed_count} for s in secrets]


@server.tool()
def memo_delete_secret(name: str) -> dict[str, bool]:
    """Delete a secret."""
    memory.forget_secret(name)
    return {"deleted": True}
```

---

## 6. Daemon (`src/memo/runtime/daemon.py` extension)

Warm socket daemon (like memo-recall-daemon):
- Listens on `~/.memo/secret.sock`
- Decrypts on-demand
- LRU cache (100 items, 1h TTL)
- Logs access to grounding.log
- Runs via launchd

---

## 7. Security

| Threat | Mitigation |
|--------|----------|
| Filesystem access | AES-256-GCM at rest; file perms 0600 |
| Process memory | Secrets only in daemon + MCP consumer |
| Key leakage | Device-bound; PBKDF2 100k iterations |
| Accidental logging | Never in stdout; audit via grounding.log |
| Cross-machine | Strictly local; device-specific key |

---

## 8. Files & Changes

| File | Change |
|------|--------|
| `src/memo/tiers.py` | Add `secret` to DURABLE_TYPES |
| `src/memo/secret_store.py` | **NEW:** encryption, key derivation, detection |
| `src/memo/store/schema.py` | Add `secret_store` table + migration |
| `src/memo/memory/secret_ops.py` | **NEW:** `_SecretOpsMixin` |
| `src/memo/memory/facade.py` | Inherit `_SecretOpsMixin` |
| `src/memo/cli_secret.py` | **NEW:** CLI commands |
| `src/memo/server_secret.py` | **NEW:** MCP tools |
| `src/memo/runtime/daemon.py` | Extend: SecretDaemon |
| `src/memo/flags_misc.py` | Add secret flags |
| `src/memo/capture_hooks.py` | Integrate secret detection |
| `tests/test_secret_*.py` | **NEW:** test files |

---

## 9. Flags

```python
flag_bool("MEMO_SECRET_STORAGE_ENABLED", default=False)
flag_bool("MEMO_CAPTURE_DETECT_SECRETS", default=False)
flag_bool("MEMO_DETECT_SECRETS_LLM", default=True)
flag_int("MEMO_SECRET_DAEMON_CACHE_MAX", default=100)
flag_int("MEMO_SECRET_DAEMON_CACHE_TTL_SECONDS", default=3600)
```

---

## 10. Success Criteria

1. ✅ Secrets encrypted at rest (AES-256-GCM)
2. ✅ Random local 256-bit master key with strict filesystem permissions
3. ✅ Auto-detection (regex + LLM)
4. ✅ User confirmation before save
5. ✅ Daemon warm socket
6. ✅ MCP tools for synapse/agents
7. ✅ Audit trail in grounding.log
8. ✅ CLI commands
9. ✅ No sync to remote
10. ✅ Not exposed in recall/briefing
11. ✅ 80%+ test coverage
