#!/usr/bin/env bash
# Set up remote-GPU MCP servers (Google Colab + Kaggle) borrowed from ds-agent.
#
#   bash scripts/setup_gpu_mcp.sh
#
# What it does (idempotent):
#   1. clones github.com/davidmiheev/ds-agent into external/ds-agent (git-ignored)
#   2. builds the Colab MCP venv (google-colab-cli + mcp<2) via ds-agent's own setup.sh,
#      and a separate small venv for the Kaggle MCP proxy (which needs mcp>=2)
#   3. installs credentials if provided through the environment:
#        COLAB_TOKEN_JSON  contents of ~/.config/colab-cli/token.json (from auth_once.py)
#        KAGGLE_API_TOKEN  Kaggle API token (KGAT...), read by .mcp.json at launch
#
# .mcp.json at the repo root then exposes `colab` and `kaggle` servers to Claude Code.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DS="$ROOT/external/ds-agent"
DS_REF="${DS_AGENT_REF:-main}"

if [ ! -d "$DS/.git" ]; then
    echo "==> cloning ds-agent ($DS_REF) into $DS"
    git clone --depth 1 --branch "$DS_REF" https://github.com/davidmiheev/ds-agent "$DS"
else
    echo "==> ds-agent already present at $DS ($(git -C "$DS" rev-parse --short HEAD))"
fi

VENV_PY="$DS/src/colab_mcp/.venv/bin/python"
if [ ! -x "$VENV_PY" ] || ! "$VENV_PY" -c "import colab_cli, mcp" 2>/dev/null; then
    echo "==> building Colab MCP venv"
    # Keep the proxy: sandboxed/cloud hosts can only reach the internet through it.
    KEEP_PROXY=1 COLAB_CLI_SRC="$ROOT/external/google-colab-cli" bash "$DS/src/colab_mcp/setup.sh"
fi

KAGGLE_VENV="$ROOT/external/kaggle-mcp-venv"
if ! "$KAGGLE_VENV/bin/python" -c "import mcp.server.lowlevel" 2>/dev/null; then
    echo "==> building Kaggle MCP venv"
    uv venv --python 3.13 "$KAGGLE_VENV"
    uv pip install --python "$KAGGLE_VENV/bin/python" "mcp==2.1.1"
fi

TOKEN_PATH="$HOME/.config/colab-cli/token.json"
if [ -n "${COLAB_TOKEN_JSON:-}" ]; then
    mkdir -p "$(dirname "$TOKEN_PATH")"
    (umask 077 && printf '%s' "$COLAB_TOKEN_JSON" > "$TOKEN_PATH")
    echo "==> wrote Colab token from COLAB_TOKEN_JSON to $TOKEN_PATH"
fi

echo
echo "Status:"
if [ -s "$TOKEN_PATH" ]; then
    echo "  colab : token present ($TOKEN_PATH)"
else
    echo "  colab : NO token — run '$VENV_PY $DS/src/colab_mcp/auth_once.py' on any machine"
    echo "          and set COLAB_TOKEN_JSON to the token.json contents (or use the colab_auth tool)"
fi
if [ -n "${KAGGLE_API_TOKEN:-}" ]; then
    echo "  kaggle: KAGGLE_API_TOKEN set"
else
    echo "  kaggle: KAGGLE_API_TOKEN not set — create one at kaggle.com → Settings → API"
fi
echo
echo "Restart Claude Code (or run /mcp) to pick up the colab and kaggle servers."
