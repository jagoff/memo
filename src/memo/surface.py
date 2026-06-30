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
        "record-history",
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
        "update",
        "stats",
        "uninstall-watcher",
        "edit",
        "watch",
    },
)

_CORE_PROFILES = {"core", "slim"}
_FULL_PROFILES = {"default", "full"}

AGENT_MCP_TOOLS: frozenset[str] = frozenset(
    {
        "memo_ask",
        "memo_get",
        "memo_graph",
        "memo_save",
        "memo_search",
        "memo_unified_briefing",
        # Session/notification plumbing registered by _srv_idle_capture outside
        # the advanced gate and never removed.
        "memo_idle_capture",
        "memo_pop_notification",
        "memo_start_session",
        "memo_save_text",
        "memo_version",
    }
)

CORE_MCP_TOOLS: frozenset[str] = frozenset(
    {
        "memo_ask",
        "memo_chat_ask",
        "memo_consolidate",
        "memo_delete",
        "memo_embed_batch",
        "memo_embed_query",
        "memo_forget",
        "memo_get",
        "memo_get_embedder_profile",
        "memo_history",
        "memo_lint",
        "memo_list",
        "memo_provenance",
        "memo_record_diff",
        "memo_reindex",
        "memo_rerank",
        "memo_save",
        "memo_search",
        "memo_search_trace",
        "memo_session_get",
        "memo_session_list",
        "memo_stats",
        "memo_unforget",
        "memo_unified_briefing",
        "memo_update",
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
    """Resolve the MCP surface profile, defaulting agent clients to nine tools."""
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


# Per-profile token-cost estimates for the `memo doctor` advisory. Reduced
# profiles (agent/core/slim) are cheap; only the full/default surface warns.
_PROFILE_TOKEN_COST: dict[str, tuple[str, str]] = {
    "agent": ("~10", "~1.2k"),
    "core": ("~30", "~2.8k"),
    "slim": ("~30", "~2.8k"),
}


def mcp_profile_token_cost(profile: str | None = None) -> tuple[str, str, bool]:
    """Return ``(tool_count_label, token_label, is_reduced)`` for ``profile``
    (or the active profile when ``None``). ``is_reduced`` is False only for the
    full/default surface — the costly one doctor warns about."""
    resolved = profile if profile is not None else mcp_profile()
    count, cost = _PROFILE_TOKEN_COST.get(resolved, ("~123", "~15k"))
    return count, cost, resolved in _PROFILE_TOKEN_COST
