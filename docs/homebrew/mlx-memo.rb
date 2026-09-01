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
  url "https://files.pythonhosted.org/packages/c2/1d/8b7d56a4e8c9e5fc230fe27a2988fa4807ad14563e63ea95a4f3c7692b6b/mlx_memo-4.16.0.tar.gz"
  sha256 "76391655f2260dc51fca3c373209140fdc0eb51348b6121f18806faebf31a307"
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
