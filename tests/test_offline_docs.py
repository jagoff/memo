from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_offline_default_and_network_opt_ins_are_documented() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    privacy = (ROOT / "docs" / "privacy.md").read_text(encoding="utf-8")
    combined = f"{readme}\n{privacy}"

    assert "offline by default" in combined.lower()
    for flag in (
        "MEMO_UPDATE_CHECK_ENABLED",
        "MEMO_AUTO_UPDATE",
        "MEMO_STATUSLINE_SELFHEAL",
        "MEMO_HOOK_SELFHEAL",
    ):
        assert flag in privacy
    for operation in ("memo update", "memo sync", "model downloads", "benchmark downloads"):
        assert operation in privacy
