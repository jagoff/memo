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
_FULL_PROFILES = {"default", "full"}

AGENT_MCP_TOOLS: frozenset[str] = frozenset(
    {
        "memory_ask",
        "memory_get",
        "memory_save",
        "memory_search",
        "memory_unified_briefing",
    }
)

CORE_MCP_TOOLS: frozenset[str] = frozenset(
    {
        "memory_ask",
        "memory_chat_ask",
        "memory_consolidate",
        "memory_delete",
        "memory_embed_batch",
        "memory_embed_query",
        "memory_forget",
        "memory_get",
        "memory_get_embedder_profile",
        "memory_history",
        "memory_lint",
        "memory_list",
        "memory_provenance",
        "memory_record_diff",
        "memory_reindex",
        "memory_rerank",
        "memory_save",
        "memory_search",
        "memory_search_trace",
        "memory_session_get",
        "memory_session_list",
        "memory_stats",
        "memory_unforget",
        "memory_unified_briefing",
        "memory_update",
    }
)


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
    return mcp_profile() in _FULL_PROFILES


def mcp_profile() -> str:
    """Resolve the MCP surface profile, defaulting agent clients to five tools."""
    profile = _profile("MEMO_MCP_PROFILE", default="agent")
    if flag_bool("MEMO_MCP_SLIM"):
        return "core"
    if profile in {"agent", *_CORE_PROFILES, *_FULL_PROFILES}:
        return profile
    return "agent"


def mcp_tools_to_remove() -> frozenset[str]:
    """Core tools removed after registration for the minimal agent surface."""
    if mcp_profile() == "agent":
        return CORE_MCP_TOOLS - AGENT_MCP_TOOLS
    return frozenset()
