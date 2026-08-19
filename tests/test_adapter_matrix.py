"""The adapter matrix must fail on drift, not just pass on a clean tree.

Each test mutates exactly one surface of a synthetic repo and asserts the
matching check flips to fail — a gate that only ever passes proves nothing.

Version parity is deliberately NOT tested here: `memo release check` owns it
(`cli_release._VERSION_TARGETS` + `_check_changelog` + `_check_mcpb_*` +
`_check_formula`), and a second, weaker opinion that can disagree with the real
gate is worse than no opinion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memo.adapter_matrix import adapter_issues, check_hook_commands, run


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A minimal repo where every adapter check passes."""
    _write_json(
        tmp_path / ".claude-plugin" / "marketplace.json",
        {"name": "memo", "plugins": [{"name": "memo", "source": "./"}]},
    )
    _write_json(tmp_path / ".claude-plugin" / "plugin.json", {"name": "memo"})
    _write_json(
        tmp_path / "plugins" / "memo" / ".codex-plugin" / "plugin.json",
        {"name": "memo", "mcpServers": "./.mcp.json"},
    )
    _write_json(
        tmp_path / "plugins" / "memo" / ".mcp.json",
        {"mcpServers": {"memo": {"command": "memo-mcp", "env": {"MEMO_NONINTERACTIVE": "1"}}}},
    )
    _write_json(
        tmp_path / ".mcp.json",
        {
            "mcpServers": {
                "memo": {
                    "command": "memo-mcp",
                    "env": {
                        "MEMO_EMBEDDER_MODEL": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
                        "MEMO_EMBEDDER_DIMS": "2560",
                    },
                }
            }
        },
    )
    _write_json(
        tmp_path / "hooks" / "hooks.json",
        {
            "hooks": {
                "SessionStart": [
                    {
                        "hooks": [
                            {"type": "command", "command": "MEMO_NONINTERACTIVE=1 memo prewarm"}
                        ]
                    }
                ],
                "Stop": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "MEMO_NONINTERACTIVE=1 memo sync once --quiet",
                            }
                        ]
                    }
                ],
            }
        },
    )
    return tmp_path


def _status(root: Path, check_id: str) -> str:
    for check in run(root):
        if check.check_id == check_id:
            return "skipped" if check.skipped else ("pass" if check.ok else "fail")
    raise AssertionError(f"no check named {check_id}")


def test_clean_repo_passes_every_check(repo: Path) -> None:
    assert [c.check_id for c in run(repo) if not c.ok] == []
    assert adapter_issues(repo) == []


# --- hooks ↔ CLI --------------------------------------------------------------


def test_hook_firing_an_unregistered_subcommand_fails(repo: Path) -> None:
    _write_json(
        repo / "hooks" / "hooks.json",
        {
            "hooks": {
                "Stop": [
                    {
                        "hooks": [
                            {"type": "command", "command": "MEMO_NONINTERACTIVE=1 memo sync onse"}
                        ]
                    }
                ]
            }
        },
    )

    assert _status(repo, "hook-commands-resolve") == "fail"
    assert any("onse" in issue for issue in adapter_issues(repo))


def test_hook_firing_an_unregistered_top_level_command_fails(repo: Path) -> None:
    _write_json(
        repo / "hooks" / "hooks.json",
        {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "memo prewarrm"}]}]}},
    )

    assert _status(repo, "hook-commands-resolve") == "fail"


def test_hook_env_prefix_and_flags_do_not_confuse_resolution(repo: Path) -> None:
    # The real graph prefixes env assignments (including a $(tty ...) subshell
    # with spaces and quotes) and appends flags; neither is a subcommand.
    _write_json(
        repo / "hooks" / "hooks.json",
        {
            "hooks": {
                "UserPromptSubmit": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    "MEMO_NONINTERACTIVE=1 "
                                    "MEMO_AGENT_TTY=${MEMO_AGENT_TTY:-$(tty 2>/dev/null | "
                                    "grep -E '^/dev/' | head -1)} "
                                    "memo session idle-maintenance --mode capture"
                                ),
                            }
                        ]
                    }
                ]
            }
        },
    )

    assert _status(repo, "hook-commands-resolve") == "pass"


def test_a_missing_hook_graph_is_not_drift(repo: Path) -> None:
    """No hook graph means no commands to resolve. Treating absence as drift
    made every partial tree (the release tests' synthetic repos, a sparse
    checkout) fail the gate for something it does not measure."""
    (repo / "hooks" / "hooks.json").unlink()

    assert _status(repo, "hook-commands-resolve") == "pass"
    assert adapter_issues(repo) == []


# --- embedder dims (MLX invariant 3) ------------------------------------------


def test_embedder_dims_not_matching_model_size_fails(repo: Path) -> None:
    # 4B is 2560-dim; 1024 belongs to the 0.6B model. This exact mismatch
    # corrupts the vec0 table.
    _write_json(
        repo / ".mcp.json",
        {
            "mcpServers": {
                "memo": {
                    "env": {
                        "MEMO_EMBEDDER_MODEL": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
                        "MEMO_EMBEDDER_DIMS": "1024",
                    }
                }
            }
        },
    )

    assert _status(repo, "embedder-dims-parity") == "fail"


def test_pinned_model_without_dims_fails(repo: Path) -> None:
    _write_json(
        repo / ".mcp.json",
        {"mcpServers": {"memo": {"env": {"MEMO_EMBEDDER_MODEL": "Qwen3-Embedding-8B"}}}},
    )

    assert _status(repo, "embedder-dims-parity") == "fail"


def test_dims_pinned_without_a_model_fails(repo: Path) -> None:
    _write_json(
        repo / ".mcp.json", {"mcpServers": {"memo": {"env": {"MEMO_EMBEDDER_DIMS": "2560"}}}}
    )

    assert _status(repo, "embedder-dims-parity") == "fail"


def test_config_pinning_neither_model_nor_dims_passes(repo: Path) -> None:
    # The shipped plugin config deliberately pins no model: the installed index
    # is self-describing and adopts its own profile. That must stay legal.
    _write_json(repo / ".mcp.json", {"mcpServers": {"memo": {"env": {"MEMO_NONINTERACTIVE": "1"}}}})

    assert _status(repo, "embedder-dims-parity") == "pass"


def test_a_repo_with_no_mcp_config_is_not_drift(repo: Path) -> None:
    """`skipped` is reserved for a surface that exists but could not be
    verified, because adapter_issues reports it. An absent surface is nothing
    to check — conflating the two made every partial tree fail the gate."""
    (repo / ".mcp.json").unlink()
    (repo / "plugins" / "memo" / ".mcp.json").unlink()

    assert _status(repo, "embedder-dims-parity") == "pass"
    assert [i for i in adapter_issues(repo) if "embedder-dims" in i] == []


def test_every_known_model_size_maps_to_its_own_dims(repo: Path) -> None:
    for model, dims in (
        ("Qwen3-Embedding-0.6B", 1024),
        ("Qwen3-Embedding-4B", 2560),
        ("Qwen3-Embedding-8B", 4096),
    ):
        _write_json(
            repo / ".mcp.json",
            {
                "mcpServers": {
                    "memo": {"env": {"MEMO_EMBEDDER_MODEL": model, "MEMO_EMBEDDER_DIMS": str(dims)}}
                }
            },
        )
        assert _status(repo, "embedder-dims-parity") == "pass", model


# --- referenced paths ---------------------------------------------------------


def test_codex_manifest_pointing_at_a_missing_mcp_config_fails(repo: Path) -> None:
    (repo / "plugins" / "memo" / ".mcp.json").unlink()

    assert _status(repo, "referenced-paths-exist") == "fail"


def test_marketplace_source_without_a_plugin_manifest_fails(repo: Path) -> None:
    _write_json(
        repo / ".claude-plugin" / "marketplace.json",
        {"plugins": [{"name": "memo", "source": "./does-not-exist"}]},
    )

    assert _status(repo, "referenced-paths-exist") == "fail"


# --- integration with the real gate -------------------------------------------


def test_release_check_surfaces_adapter_drift(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`memo release check` must actually run these checks — a module nobody
    calls is the failure mode this whole change exists to avoid."""
    from memo import cli_release

    _write_json(
        repo / "hooks" / "hooks.json",
        {
            "hooks": {
                "Stop": [{"hooks": [{"type": "command", "command": "memo definitely-not-a-cmd"}]}]
            }
        },
    )
    # Short-circuit the version half; this test is about the adapter half.
    monkeypatch.setattr(cli_release, "_read_current_version", lambda _repo: "1.2.3")
    monkeypatch.setattr(cli_release, "_check_json_version_targets", lambda *a, **k: None)
    monkeypatch.setattr(cli_release, "_check_install_pins", lambda *a, **k: None)
    monkeypatch.setattr(cli_release, "_check_additional_server_packages", lambda *a, **k: None)
    monkeypatch.setattr(cli_release, "_check_mcpb_manifest", lambda *a, **k: None)
    monkeypatch.setattr(cli_release, "_check_mcpb_node_manifest", lambda *a, **k: None)
    monkeypatch.setattr(cli_release, "_check_mcpb_archive", lambda *a, **k: None)
    monkeypatch.setattr(cli_release, "_check_changelog", lambda *a, **k: None)
    monkeypatch.setattr(cli_release, "_check_formula", lambda *a, **k: None)

    report = cli_release.release_check_report(repo)

    assert any("definitely-not-a-cmd" in issue for issue in report.issues), report.issues


def test_real_repo_has_no_adapter_drift() -> None:
    """The checked-in tree must be clean, so a red run means a real regression."""
    root = Path(__file__).resolve().parent.parent

    assert adapter_issues(root) == []


# --- error / skip paths -------------------------------------------------------


def test_unreadable_json_becomes_an_issue_not_a_silent_pass(repo: Path) -> None:
    """A gate that cannot parse its input must say so. Returning [] here would
    report "no drift" for a repo it never actually checked."""
    (repo / ".mcp.json").write_text("{ this is not json", encoding="utf-8")

    issues = adapter_issues(repo)

    assert issues and any("could not run" in i for i in issues)


def test_a_check_that_could_not_run_is_reported_as_an_issue(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`skipped` exists for a surface that IS there but could not be verified —
    e.g. the CLI is not importable. That must surface, never pass silently."""
    from memo import adapter_matrix as am

    def _unimportable(root: Path):
        check = am.Check(check_id="hook-commands-resolve", surface="s", description="d")
        check.skipped = True
        check.findings.append("memo package not importable, CLI not verified: boom")
        return check

    monkeypatch.setattr(am, "CHECKS", (_unimportable,))

    issues = adapter_issues(repo)

    assert issues == [
        "hook-commands-resolve did not run: memo package not importable, CLI not verified: boom"
    ]


def test_hook_commands_check_covers_the_nightly_script(tmp_path: Path) -> None:
    """The nightly LaunchAgent script is the OTHER surface that fires `memo`
    subcommands, and it broke exactly this way: `ops gc-emitted-ledgers` shipped
    in the template before the binary registered it, so the pass logged
    `Error: No such command` into a file nobody reads for four nights.
    """
    (tmp_path / "launchd").mkdir()
    (tmp_path / "launchd" / "memo-nightly.sh").write_text(
        '#!/bin/sh\n"__MEMO_BIN__" ops gc-nonexistent --json\n', encoding="utf-8"
    )

    check = check_hook_commands(tmp_path)

    assert check.findings, "a nightly script calling an unknown subcommand must fail the gate"
    assert "memo-nightly.sh" in check.findings[0]
    assert "gc-nonexistent" in check.findings[0]


def test_hook_commands_check_passes_on_the_shipped_nightly_script() -> None:
    """…and the shipped script must actually resolve."""
    check = check_hook_commands(Path(__file__).resolve().parents[1])

    assert not check.findings, check.findings
    assert not check.skipped
