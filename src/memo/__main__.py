"""`python -m memo` entrypoint.

Exists so a checkout can be run without installing it: the `--against` eval
comparison invokes a second git worktree's code via
``PYTHONPATH=<worktree>/src python -m memo``. Going through the installed
`memo` console script would run the globally installed uv tool instead, and the
comparison would silently evaluate the same code twice.
"""

from memo.cli import main

if __name__ == "__main__":
    main()
