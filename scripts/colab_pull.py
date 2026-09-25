"""Download files from a live Colab runtime (the ds-agent colab MCP has no download tool yet).

  external/ds-agent/src/colab_mcp/.venv/bin/python scripts/colab_pull.py <session> <dest_dir> <remote_path>...

Uses the session endpoint/token that google-colab-cli keeps in ~/.config/colab-cli/sessions.json
and the Jupyter Contents API, which is rooted at `/` on Colab (so /content/x -> "content/x").
The runtime proxy token expires after ~1h; run any colab MCP tool first to refresh it.
"""

import sys
from pathlib import Path

from colab_cli.contents import ContentsClient
from colab_cli.state import StateStore

if len(sys.argv) < 4:
    raise SystemExit(__doc__)
session, dest, *paths = sys.argv[1:]
state = StateStore().get(session)
if state is None:
    raise SystemExit(f"no colab session named {session!r} in ~/.config/colab-cli/sessions.json")
client = ContentsClient(state)
Path(dest).mkdir(parents=True, exist_ok=True)
for p in paths:
    out = Path(dest) / Path(p).name
    client.download(p.lstrip("/"), str(out))
    print(f"{p} -> {out} ({out.stat().st_size} bytes)")
