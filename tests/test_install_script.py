from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _installer_env(tmp_path: Path, fake_bin: Path, log: Path) -> dict[str, str]:
    return {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "MEMO_INSTALL_SKIP_PLATFORM_CHECK": "1",
        "MEMO_INSTALL_DOWNLOAD_MODELS": "no",
        "MEMO_INSTALL_SKIP_AGENT_CONFIG": "1",
        "MEMO_INSTALL_TEST_LOG": str(log),
        "NO_COLOR": "1",
        "TERM": "dumb",
    }


def test_failed_uv_install_preserves_existing_tool_and_uses_release_pin(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "commands.log"
    _executable(
        fake_bin / "uv",
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$MEMO_INSTALL_TEST_LOG"\nexit 23\n',
    )
    sentinel = tmp_path / "home/.local/share/uv/tools/mlx-memo/existing.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("keep", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(ROOT / "install.sh")],
        env=_installer_env(tmp_path, fake_bin, log),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert "tool install mlx-memo==3.8.1 --force" in log.read_text(encoding="utf-8")
    assert "uninstall" not in log.read_text(encoding="utf-8")


def test_failed_pipx_install_preserves_existing_tool(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "commands.log"
    _executable(fake_bin / "python3.13", "#!/bin/sh\nexit 0\n")
    _executable(
        fake_bin / "pipx",
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$MEMO_INSTALL_TEST_LOG"\n'
        'if [ "$1" = install ]; then exit 23; fi\nexit 0\n',
    )
    sentinel = tmp_path / "home/.local/pipx/venvs/mlx-memo/existing.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("keep", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(ROOT / "install.sh")],
        env=_installer_env(tmp_path, fake_bin, log),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "keep"
    commands = log.read_text(encoding="utf-8")
    assert "install mlx-memo==3.8.1 --force" in commands
    assert "uninstall" not in commands


def test_ubuntu_installer_requests_managed_python_313_from_uv(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "commands.log"
    _executable(
        fake_bin / "uv",
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$MEMO_INSTALL_TEST_LOG"\n',
    )
    env = {**os.environ, "PATH": f"{fake_bin}:/usr/bin:/bin", "MEMO_INSTALL_TEST_LOG": str(log)}

    result = subprocess.run(
        ["bash", str(ROOT / "scripts/install-ubuntu.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8") == (
        "tool install --python 3.13 --find-links "
        "https://download.pytorch.org/whl/cpu/torch/ mlx-memo\n"
    )


def test_ubuntu_installer_rejects_pipx_without_python_313(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "commands.log"
    _executable(fake_bin / "python3", "#!/bin/sh\nexit 1\n")
    _executable(
        fake_bin / "pipx",
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$MEMO_INSTALL_TEST_LOG"\n',
    )
    env = {**os.environ, "PATH": f"{fake_bin}:/usr/bin:/bin", "MEMO_INSTALL_TEST_LOG": str(log)}

    result = subprocess.run(
        ["bash", str(ROOT / "scripts/install-ubuntu.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Python >= 3.13" in result.stderr
    assert not log.exists()


def test_ubuntu_pipx_fallback_uses_official_cpu_wheels(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "commands.log"
    _executable(fake_bin / "python3", "#!/bin/sh\nexit 0\n")
    _executable(
        fake_bin / "pipx",
        '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$MEMO_INSTALL_TEST_LOG"\n',
    )
    env = {**os.environ, "PATH": f"{fake_bin}:/usr/bin:/bin", "MEMO_INSTALL_TEST_LOG": str(log)}

    result = subprocess.run(
        ["bash", str(ROOT / "scripts/install-ubuntu.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8") == (
        "install mlx-memo --pip-args=--find-links https://download.pytorch.org/whl/cpu/torch/\n"
    )
