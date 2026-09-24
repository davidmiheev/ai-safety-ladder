#!/usr/bin/env bash
# SessionStart hook: on fresh cloud containers, install deps and the GPU MCP servers.
# Local machines are left alone — run `uv sync` / scripts/setup_gpu_mcp.sh yourself there.
set -euo pipefail
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0
cd "$CLAUDE_PROJECT_DIR"
uv sync --quiet >&2
bash scripts/setup_gpu_mcp.sh >&2
