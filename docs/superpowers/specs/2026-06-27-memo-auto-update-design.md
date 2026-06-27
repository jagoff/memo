# Auto-update Cross-Machine Design

## Problem

`memo update` only runs locally — each Mac must execute it manually. When a Mac is left behind (days/weeks without running), there's no automatic catch-up mechanism.

## Solution

Use the existing sync infrastructure to propagate software updates across machines.

### How It Works

1. The sync repo contains a `memo-version.json` with the current version
2. On `memo sync pull`, the client compares remote vs local version
3. If remote version > local, automatically runs `memo update`
4. The version file is updated on every `memo update` and pushed with the next sync

## Metadata File

Location: `{sync_repo}/memo-version.json`

```json
{
  "version": "v1.2.3",
  "updated_at": "2026-06-27T10:00:00Z"
}
```

Updated on every software update and pushed to the sync repo.

## Flow

### On memo update (local)

```python
def self_update():
    # ... existing update logic ...
    # After successful update:
    _update_version_file(new_version)
```

### On memo sync pull (remote machine)

```python
def sync_pull():
    # ... existing pull logic ...
    # Check version after pull
    remote_version = _read_version_from_repo()
    local_version = importlib.metadata.version("mlx-memo")
    if _version_gt(remote_version, local_version):
        subprocess.run(["memo", "update"])
```

## SessionStart Hook Integration

The existing `SessionStart` hook already runs `memo sync auto`. Add version check after the sync:

```
memo sync auto
  └─检出 remote version
  └─ if remote > local:
       └─ memo update
```

## Configuration

| Flag | Default | Description |
|------|--------|-------------|
| `MEMO_AUTO_UPDATE` | `1` | Enable auto-update (0 to disable) |
| `MEMO_AUTO_UPDATE_CHECK` | `1` | Check version on sync pull (0 to disable) |

## Edge Cases

- **Editale install**: Skip auto-update (existing behavior in `update.py`)
- **Update fails**: Log error, continue — don't block sync
- **Network offline**: Skip, retry on next session
- **Sync not configured**: No auto-update (local tier)

## Components

| Component | File | Role |
|-----------|------|------|
| `runtime/version_file.py` | NEW | Read/write version metadata |
| `runtime/update.py` | MODIFY | Update version file after update |
| `cli_sync.py` | MODIFY | Check version on pull |
| `hooks/hooks.json` | MODIFY | Add update step after sync |

## Testing

- Unit tests for version comparison
- Integration test: two machines (one updated, one behind) → sync pulls → second machine updates