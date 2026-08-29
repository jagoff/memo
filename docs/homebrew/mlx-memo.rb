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
  include Language::Python::Virtualenv

  desc "Local MCP memory for AI agents — MLX-native, sqlite-vec, markdown vault"
  homepage "https://github.com/jagoff/memo"
  url "https://files.pythonhosted.org/packages/fc/33/24398b2b7b1a39301a55d50d9ead3403caf6642fff84c5f4ee75cda7b4ba/mlx_memo-4.14.6.tar.gz"
  sha256 "3eb941cd9495903aaf6f2ad0711fd4985557d475c26d11e891835199b0c4631e"
  license "MIT"

  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.13"

  preserve_rpath

  # We let pip resolve the dep tree at install time rather than
  # vendoring every `resource` block (mlx, mlx-lm, fastmcp, sqlite-vec,
  # frontmatter, watchdog, etc. — ~30 deps). This is fine for a
  # personal tap; homebrew-core would require explicit resources.
  def install
    virtualenv_create(libexec, "python3.13", system_site_packages: false)
    system "python3.13", "-m", "pip", "--python=#{libexec}/bin/python",
           "install", "--no-compile", buildpath
    bin.install_symlink libexec/"bin/memo"
    bin.install_symlink libexec/"bin/memo-mcp"
  end

  test do
    assert_match "memo, version", shell_output("#{bin}/memo --version")
    assert_match "Usage:", shell_output("#{bin}/memo --help")
    # Import the MCP entry point without starting its blocking stdio loop.
    system libexec/"bin/python", "-c", "import memo.server"
  end
end
