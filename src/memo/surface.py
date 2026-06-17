from __future__ import annotations

from memo.flags import flag_bool, flag_str

CORE_CLI_COMMANDS: frozenset[str] = frozenset(
    {
        "ask",
        "as-of",
        "briefing",
        "capture-stop",
        "config",
        "delete",
        "diff",
        "doctor",
        "get",
        "historia",
        "history",
        "init",
        "install-mcp",
        "install-slash",
        "install-watcher",
        "list",
        "mcp-command",
        "migrate",
        "migrate-vault",
        "prewarm",
        "provenance",
        "recall-daemon",
        "recall-hook",
        "reindex",
        "restore",
        "save",
        "search",
        "self-update",
        "stats",
        "uninstall-watcher",
        "update",
        "watch",
    },
)

_CORE_PROFILES = {"core", "slim"}


def _profile(name: str, *, default: str = "default") -> str:
    value = flag_str(name).strip().lower()
    return value or default


def cli_command_visible(command_name: str) -> bool:
    """Return whether a root CLI command is visible in the selected profile."""
    profile = _profile("MEMO_CLI_PROFILE")
    if profile in _CORE_PROFILES:
        return command_name in CORE_CLI_COMMANDS
    return True


def mcp_include_advanced_tools() -> bool:
    """Return whether MCP should expose advanced/experimental tool modules."""
    profile = _profile("MEMO_MCP_PROFILE")
    if profile in _CORE_PROFILES:
        return False
    return not flag_bool("MEMO_MCP_SLIM")
