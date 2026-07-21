# Homebrew formula for mlx-memo.
#
# Lives in this source tree as a reference copy. To activate it for
# end users, mirror this file to a tap repo named `homebrew-memo`
# under your GitHub user (see `docs/homebrew/README.md`).
# This file tracks the exact source distribution URL published by PyPI.
# On each release, copy the URL and sha256 from the PyPI JSON API before
# mirroring the formula to the public tap.
#
# Users then install with:
#
#     brew tap jagoff/memo
#     brew install mlx-memo
#
# Apple Silicon only — MLX is the load-bearing dep and refuses to
# build on Intel / Linux. The formula enforces this with
# `depends_on arch: :arm64` so brew refuses early.

class MlxMemo < Formula
  desc "Local MCP memory for AI agents — MLX-native, sqlite-vec, markdown vault"
  homepage "https://github.com/jagoff/memo"
  url "https://files.pythonhosted.org/packages/e1/d1/162faaba41ec4de5cc2a439c2592a7db5fae5e1962f358047f1ed7852861/mlx_memo-3.8.2.tar.gz"
  sha256 "58e4dbfad30b6daf902edc20bbc4c290cd93c70124e8969c65b46eef6067ca0e"
  license "MIT"

  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.13"

  # We let pip resolve the dep tree at install time rather than
  # vendoring every `resource` block (mlx, mlx-lm, fastmcp, sqlite-vec,
  # frontmatter, watchdog, etc. — ~30 deps). This is fine for a
  # personal tap; homebrew-core would require explicit resources.
  def install
    venv = virtualenv_create(libexec, "python3.13")
    venv.pip_install_and_link buildpath
  end

  test do
    assert_match "memo, version", shell_output("#{bin}/memo --version")
    assert_match "Usage:", shell_output("#{bin}/memo --help")
    # MCP server entry point must be importable too.
    system bin/"memo-mcp", "--help"
  end
end
