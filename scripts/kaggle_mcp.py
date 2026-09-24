"""Launch ds-agent's Kaggle MCP proxy with proxy/CA env vars passed through to npx.

ds-agent's proxy starts `npx mcp-remote` without an explicit env, so the mcp SDK hands
the child only its "safe" default environment. That drops HTTPS_PROXY and
NODE_EXTRA_CA_CERTS, and behind a TLS-intercepting proxy (e.g. Claude Code cloud
sessions) npm then fails with SELF_SIGNED_CERT_IN_CHAIN. Extending the default env
here fixes that without patching ds-agent.
"""

import os

import mcp.client.stdio as stdio

PASSTHROUGH = (
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy",
    "NODE_EXTRA_CA_CERTS", "NODE_USE_ENV_PROXY", "SSL_CERT_FILE", "npm_config_cafile",
)
_default_env = stdio.get_default_environment


def _env_with_proxy() -> dict[str, str]:
    env = _default_env()
    env.update({k: os.environ[k] for k in PASSTHROUGH if k in os.environ})
    return env


stdio.get_default_environment = _env_with_proxy

from ds_agent.kaggle_mcp import main  # noqa: E402

if __name__ == "__main__":
    main()
