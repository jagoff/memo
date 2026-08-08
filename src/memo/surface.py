from __future__ import annotations

from memo import flags

CORE_CLI_COMMANDS: frozenset[str] = frozenset(
    {
        "ask",
        "as-of",
        "briefing",
        "capture-stop",
        "config",
        "context",
        "context-pack",
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
        "rename",
        "restore",
        "save",
        "search",
        "update",
        "stats",
        "terminal",
        "uninstall-watcher",
        "edit",
        "watch",
    },
)

_CORE_PROFILES = {"core", "slim"}
_FULL_PROFILES = {"default", "full"}

_OPERATIONAL_MCP_TOOLS: frozenset[str] = frozenset(
    {
        "memo_attention_ack",
        "memo_attention_add",
        "memo_conflict_open",
        "memo_conflict_resolve",
        "memo_evidence_pack",
        "memo_federation_preview",
        "memo_focus_clear",
        "memo_focus_set",
        "memo_handoff_consume",
        "memo_handoff_create",
        "memo_journal_verify",
        "memo_operational_state",
        "memo_outcome_record",
        "memo_procedure_candidates",
        "memo_procedure_promote",
        "memo_signal_list",
        "memo_signal_remember",
    }
)

AGENT_MCP_TOOLS: frozenset[str] = (
    frozenset(
        {
            "memo_ask",
            "memo_context",
            "memo_delete",
            "memo_get",
            "memo_graph",
            "memo_history",
            "memo_invalidate",
            "memo_mark_reviewed",
            "memo_review_due",
            "memo_supersede",
            # Context-economy primitive registered always-on and never removed —
            # present on every surface profile (incl. the minimal agent one).
            "memo_offload",
            "memo_rename",
            "memo_save",
            "memo_search",
            "memo_unified_briefing",
            "memo_update",
            # Session/notification plumbing registered by _srv_idle_capture outside
            # the advanced gate and never removed.
            "memo_idle_capture",
            "memo_pop_notification",
            "memo_profile",
            "memo_start_session",
            "memo_terminal_list",
            "memo_save_text",
            "memo_version",
            "memo_write_queue_status",
        }
    )
    | _OPERATIONAL_MCP_TOOLS
)

CORE_MCP_TOOLS: frozenset[str] = (
    frozenset(
        {
            "memo_ask",
            "memo_chat_ask",
            "memo_consolidate",
            "memo_context",
            "memo_delete",
            "memo_embed_batch",
            "memo_embed_query",
            "memo_forget",
            "memo_get",
            "memo_get_embedder_profile",
            "memo_history",
            "memo_invalidate",
            "memo_lint",
            "memo_mark_reviewed",
            "memo_list",
            "memo_provenance",
            "memo_profile",
            "memo_record_diff",
            "memo_reindex",
            "memo_rename",
            "memo_rerank",
            "memo_save",
            "memo_search",
            "memo_search_trace",
            "memo_session_get",
            "memo_session_list",
            "memo_stats",
            "memo_terminal_list",
            "memo_review_due",
            "memo_supersede",
            "memo_unforget",
            "memo_unified_briefing",
            "memo_update",
            "memo_write_queue_status",
        }
    )
    | _OPERATIONAL_MCP_TOOLS
)


def _profile(name: str, *, default: str = "default", strict: bool = False) -> str:
    # Resolve through the module on every call. Importing a function alias here
    # can permanently capture a temporary monkeypatch when this module is first
    # imported, poisoning all later CLI/MCP profile checks in that process.
    value = (flags.flag_str(name, strict=strict) or default).strip().lower()
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
    """Resolve the MCP surface profile, defaulting agent clients to 41 tools."""
    profile = _profile("MEMO_MCP_PROFILE", default="agent", strict=True)
    if flags.flag_bool("MEMO_MCP_SLIM"):
        return "core"
    return profile


def mcp_tools_to_remove() -> frozenset[str]:
    """Core tools removed after registration for the minimal agent surface."""
    if mcp_profile() == "agent":
        return CORE_MCP_TOOLS - AGENT_MCP_TOOLS
    return frozenset()


# Per-profile token-cost estimates for the `memo doctor` advisory. Reduced
# profiles (agent/core/slim) are cheap; only the full/default surface warns.
_PROFILE_TOKEN_COST: dict[str, tuple[str, str]] = {
    "agent": ("41", "~9.4k"),
    "core": ("58", "~12.9k"),
    "slim": ("58", "~12.9k"),
}


def mcp_profile_token_cost(profile: str | None = None) -> tuple[str, str, bool]:
    """Return ``(tool_count_label, token_label, is_reduced)`` for ``profile``
    (or the active profile when ``None``). ``is_reduced`` is False only for the
    full/default surface — the costly one doctor warns about."""
    resolved = profile if profile is not None else mcp_profile()
    count, cost = _PROFILE_TOKEN_COST.get(resolved, ("164", "~30.4k"))
    return count, cost, resolved in _PROFILE_TOKEN_COST
