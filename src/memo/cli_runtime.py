"""Runtime + install plumbing for the memo CLI (re-export shim).

The implementation was split into the ``memo.runtime`` package by concern
(install / daemon / report). This module remains as a thin re-export shim so
existing imports — entry points, ``memo.cli``, ``cli_diag`` — keep resolving
``from memo.cli_runtime import ...`` unchanged.
"""

from __future__ import annotations

# Module-level names re-exported from the original cli_runtime surface so any
# external `from memo.cli_runtime import <name>` keeps working.
import shutil  # noqa: F401
from collections.abc import Sequence  # noqa: F401
from importlib.resources import files as package_files  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any  # noqa: F401

from rich.panel import Panel  # noqa: F401

from memo.cli_common import console  # noqa: F401
from memo.config import Config  # noqa: F401
from memo.runtime.daemon import (  # noqa: F401
    install_watcher,
    prewarm,
    sleep_cycle,
    uninstall_watcher_cmd,
    watch,
)
from memo.runtime.install import (  # noqa: F401
    _MCP_ENV_FORWARD_KEYS,
    _MISSING_MCP_OK_ERRORS,
    _WRAPPER_SNIPPET_ZSH,
    _agent_asset_root,
    _codex_home,
    _codex_read_app_server_response,
    _codex_send_app_server_request,
    _copy_slash_skill,
    _devin_desktop_mcp_config_path,
    _env_flags,
    _env_root_for_bin,
    _format_command,
    _install_codex_plugin,
    _install_devin_desktop_mcp,
    _install_mode,
    _mcp_add_command,
    _mcp_server_env,
    _mcp_server_json,
    _path_is_relative_to,
    _resolve_command,
    _resolved_memo_mcp,
    _run_agent_command,
    _runtime_install_report,
    _safe_resolve,
    init_cmd,
    install_shell_wrapper,
    install_slash,
    mcp_command,
    migrate_vault,
    self_update,
)
from memo.runtime.report import _print_runtime_install_report  # noqa: F401
from memo.setup import run_picker, write_config_file  # noqa: F401
