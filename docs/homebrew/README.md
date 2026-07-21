# Homebrew tap for `mlx-memo`

This directory carries the reference Homebrew formula. The public tap lives at
[`jagoff/homebrew-memo`](https://github.com/jagoff/homebrew-memo); Homebrew
requires tap repos to start with the `homebrew-` prefix.

## One-time setup

1. **Create the tap repo on GitHub if it does not exist.** Empty, public,
   named `homebrew-memo`. Visit <https://github.com/new> and set owner
   `jagoff`, name `homebrew-memo`, and description
   `Homebrew tap for mlx-memo and friends`.

2. **Clone + populate.**

   ```bash
   gh repo clone jagoff/homebrew-memo
   cd homebrew-memo
   mkdir -p Formula
   cp /path/to/memo/docs/homebrew/mlx-memo.rb Formula/mlx-memo.rb
   cat > README.md <<'EOF'
   # homebrew-memo

   Homebrew tap for [`mlx-memo`](https://github.com/jagoff/memo) — local
   MCP memory for AI agents, MLX-native, Apple Silicon.

   ## Install

   ```bash
   brew tap jagoff/memo
   brew install mlx-memo
   memo prewarm --download-all
   memo install-slash --client claude-code --client codex --client devin-desktop
   ```

   Apple Silicon (M1/M2/M3/M4) only. The formula refuses to install on
   Intel Macs because MLX doesn't build there.

   ## Upgrading

   ```bash
   brew update && brew upgrade mlx-memo
   ```

   ## Source

   - Main repo: <https://github.com/jagoff/memo>
   - PyPI: <https://pypi.org/project/mlx-memo/>
   - License: MIT
   EOF
   git add Formula/ README.md
   git commit -m "Initial tap: mlx-memo <current-version>"
   git push -u origin master
   ```

3. **Test the install end-to-end.**

   ```bash
   brew tap jagoff/memo
   brew install mlx-memo
   memo --version    # should print the formula version
   memo doctor --strict-runtime
   memo mcp-command --client devin-desktop
   brew uninstall mlx-memo
   brew untap jagoff/memo
   ```

## Updating the formula on each release

After every `pyproject.toml` version bump and PyPI publish:

```bash
# In the memo repo
NEW_VERSION=<next-version>
URL="https://github.com/jagoff/memo/archive/refs/tags/v${NEW_VERSION}.tar.gz"
SHA=$(curl -sL "$URL" | shasum -a 256 | awk '{print $1}')

# Update the formula in this repo (docs/homebrew/mlx-memo.rb):
#   url "..."  ← new URL
#   sha256 "..."  ← new sha
# Then mirror to the homebrew-memo tap:
cd /path/to/homebrew-memo
cp /path/to/memo/docs/homebrew/mlx-memo.rb Formula/mlx-memo.rb
git commit -am "mlx-memo $NEW_VERSION"
git push
```

Do not bump this formula to a version until the matching GitHub tag exists and
its source tarball `sha256` has been calculated. Between a source-tree version
bump and a published tag, `memo release check` reports formula drift as a
warning; use `memo release check --strict-docs` after tagging.

## Why a personal tap, not homebrew-core?

`homebrew-core` requires every Python dependency to be vendored as an
explicit `resource` block — Mlx-memo pulls ~30 transitive deps via
mlx-lm, fastmcp, sqlite-vec, watchdog, frontmatter, etc. Maintaining
those blocks by hand is grinding work. For a single-author, single-
platform project a personal tap is the right scope; if we ever want
into `homebrew-core`, generate the resource list with
[`homebrew-pypi-poet`](https://pypi.org/project/homebrew-pypi-poet/).
