# Task 6 TOCTOU final-pass correction report

## Status

The late-subtree mutation window is closed. The retirement audit remains
read-only and retirement cleanup remains refusal-only.

## Finding and correction

The previous traversal validated a child directory immediately before closing
its descriptor. A sibling scanned afterward could mutate that already-returned
child without changing the child's identity in its parent, allowing the audit
to miss the new entry.

`_safe_files` now retains the descriptor and initial snapshot for every
directory visited during the complete traversal. Only after every sibling and
descendant has been scanned does it perform a final validation pass over every
retained frame:

- re-list the directory and require exact membership equality;
- re-stat every initial entry through the retained descriptor;
- require unchanged identity and file type;
- require unchanged regular-file size, mtime, and ctime metadata.

Root-component link validation remains in place after the directory-frame
pass. Any open, list, or stat failure rejects the audit.

## Race regression

The new regression visits `a`, scans sibling `z`, and creates
`a/late.txt` while reading `z/trigger.txt`.

Before the correction:

```text
1 failed, 27 deselected
```

After the correction:

```text
1 passed, 27 deselected
```

## Verification

```text
uv run --no-sync pytest tests/tools/test_retirement_audit.py -q
28 passed

uv run --no-sync pytest tests/tools/test_retirement_audit.py tests/tools/test_absorption_*.py -q
67 passed

uv run --no-sync pytest tests/tools -q
168 passed

uv run --no-sync ruff check tools/memflow_absorption tests/tools/test_retirement_audit.py
All checks passed!

uv run --no-sync ruff format --check tools/memflow_absorption/inventory.py tests/tools/test_retirement_audit.py
2 files already formatted

uv run --no-sync mypy tools/memflow_absorption
Success: no issues found in 15 source files

git diff --check
passed
```

## Material limitation

The required proof retains one open descriptor per visited directory until the
final pass. Descriptor use therefore scales with the total directory count,
not only maximum traversal depth. If the process reaches `RLIMIT_NOFILE`,
`os.open` fails and the audit refuses verification. No descriptor-reopening or
path-based fallback is used because either would recreate the unsafe identity
window this correction closes.

## Scope

Only `tools/memflow_absorption/inventory.py`,
`tests/tools/test_retirement_audit.py`, and this report were changed. No
cleanup, deletion, service, LaunchAgent, runtime, configuration, state,
repository, release, or documentation mutation was performed.
