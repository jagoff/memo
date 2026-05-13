#!/usr/bin/env bash
# Register this plugin against a running Paperclip instance via its install API.
# Idempotent: re-runs are safe; the host returns the existing record.
set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${PAPERCLIP_HOST:-http://127.0.0.1:3100}"

if [[ ! -d "$PLUGIN_DIR/dist" ]]; then
  echo "[memo-plugin] dist/ not found — running pnpm build first" >&2
  (cd "$PLUGIN_DIR" && pnpm build)
fi

if ! curl -fsS -m 3 "$HOST/api/health" >/dev/null 2>&1; then
  echo "[memo-plugin] Paperclip server at $HOST is not reachable." >&2
  echo "  Start it first: cd ~/repositories/paperclip && pnpm dev" >&2
  exit 1
fi

echo "[memo-plugin] Installing $PLUGIN_DIR into Paperclip at $HOST"
curl -fsS -X POST "$HOST/api/plugins/install" \
  -H "Content-Type: application/json" \
  -d "$(printf '{"packageName":"%s","isLocalPath":true}' "$PLUGIN_DIR")" \
  | python3 -m json.tool
