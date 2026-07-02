"""Bootstrap stub for the memo MCP bundle.

Claude Desktop launches the server via manifest.json -> mcp_config
(`uvx --from mlx-memo memo-mcp`), so this file is never executed in
normal operation. It exists because the MCPB spec expects a python
entry_point, and doubles as a helpful error if run directly.
"""

import shutil
import sys

if shutil.which("uv") is None and shutil.which("uvx") is None:
    sys.stderr.write(
        "memo needs the `uv` runtime (https://docs.astral.sh/uv/). "
        "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh\n"
    )
    sys.exit(1)

sys.stderr.write("Run `uvx --from mlx-memo memo-mcp` instead of invoking this stub.\n")
sys.exit(1)
