# Homebrew formula for mlx-memo.
#
# Lives in this source tree as a reference copy. To activate it for
# end users, mirror this file to a tap repo named `homebrew-memo`
# under your GitHub user (see `docs/homebrew/README.md`).
# This file tracks the latest published PyPI sdist, not necessarily the
# in-flight source-tree version; update only after the sdist exists and
# the sha256 has been calculated.
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
  url "https://files.pythonhosted.org/packages/source/m/mlx-memo/mlx_memo-2.9.7.tar.gz"
  sha256 "4e90be465e26d7ed253b20c0a80d0e701aec5bde6d06caa321cb1512038aeee7"
  license "MIT"

  depends_on :macos
  depends_on arch: :arm64
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
