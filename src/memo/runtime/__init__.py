"""memo CLI runtime + install package.

Split out of the monolithic ``memo.cli_runtime`` module by concern:

- ``install`` — install/MCP-registration helpers and the setup commands.
- ``daemon`` — launchctl / recall-daemon lifecycle commands.
- ``report`` — presentation of the runtime install report.

``memo.cli_runtime`` remains as a thin re-export shim for backwards compat.
"""
