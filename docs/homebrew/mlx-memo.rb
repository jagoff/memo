# Homebrew formula for mlx-memo.
#
# Lives in this source tree as a reference copy. To activate it for
# end users, mirror this file to a tap repo named `homebrew-memo`
# under your GitHub user (see `docs/homebrew/README.md`).
# This file tracks the latest released git tag via GitHub's auto-generated
# source tarball (no PyPI publish required). On each release, bump the tag
# in the url and recompute the sha256:
#   curl -sL https://github.com/jagoff/memo/archive/refs/tags/vX.Y.Z.tar.gz | shasum -a 256
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
  url "https://github.com/jagoff/memo/archive/refs/tags/v3.1.0.tar.gz"
  sha256 "977e4dfa9691bb421ff9305b20ad48f49193c1a857fa42ade6541e2ad47ca0fc"
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
