#!/usr/bin/env bash
# Smoke-test a running compose stack (default: http://localhost:80).
set -euo pipefail
BASE="${1:-http://localhost:80}"
fail() { echo "SMOKE FAIL: $1"; exit 1; }

echo "1/3 SPA served at / ..."
code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/")
[ "$code" = "200" ] || fail "SPA returned $code (want 200)"

echo "2/3 API reachable via /api ..."
code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/api/jobs/nope")
[ "$code" = "404" ] || fail "API returned $code (want 404 for unknown job)"

echo "3/3 MCP handshake via /mcp ..."
# initialize is a POST; a 200/2xx or a JSON-RPC body means the endpoint is live.
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/mcp" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}')
case "$code" in
  2*|400) echo "   mcp endpoint responded ($code)";;
  *) fail "MCP returned $code";;
esac

echo "ALL SMOKE CHECKS PASSED"
