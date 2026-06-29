# Per-Project Storage + 3-Tier Recall Relevance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store new memory `.md` files in per-project folders (derived from the existing `project:` tag) and add a 3-tier soft recall boost (current-project > global/cross-cutting > other-projects), keeping the search pool global.

**Architecture:** The `project:` tag in frontmatter stays the single source of truth. From it we derive (a) the on-disk bucket folder and (b) the recall ranking tier. The sqlite index already globs recursively, so search remains global with zero index changes. Storage and ranking are two independent, derived components.

**Tech Stack:** Python 3.13, Click CLI, sqlite-vec, MLX (deferred imports), pytest, ruff, mypy. Package manager: `uv`.

## Global Constraints

- Run all commands from the repo root with `uv run --no-sync <cmd>`.
- **MLX invariants:** `MLXEmbedder.embed()` takes `Sequence[str]` never a bare `str`; query prefix only on queries; `mlx`/`mlx-lm` imports stay deferred inside functions. Pure storage/ranking logic here needs NO MLX; tests that don't embed must use `defer_embed=True` or fabricated hit objects so they run without MLX.
- **Markdown is source of truth:** the `project:` tag (frontmatter) defines the project; the folder and the index are derived. Never make the folder authoritative.
- **Flags:** use `flag_bool/int/float/str`; never `os.environ.get("MEMO_...")` inline. Register every flag in the matching `flags_<group>.py`.
- **Test isolation** (`tests/conftest.py`): use `tmp_cfg`; never read/write the real vault; `CliRunner` sets `MEMO_NONINTERACTIVE=1` + `MEMO_DATA_DIR` + `MEMO_STATE_DIR`.
- **Retrieval-regression discipline:** the ranking change must keep `memo eval recall` precision high / noise low across the labeled set — do not regress per-query.
- **Green gate every commit:** `uv run --no-sync ruff check src/ tests/`, `uv run --no-sync mypy src/memo/`, `uv run --no-sync pytest tests/` must pass.

---

### Task 1: `project_bucket` helper — the single tag→folder mapping

**Files:**
- Modify: `src/memo/project.py` (add constant + function after `slugify_project`, ~line 29)
- Test: `tests/test_project.py` (create if absent; else append)

**Interfaces:**
- Produces: `GLOBAL_BUCKET: str = "_global"` and `project_bucket(tags: list[str]) -> str` — returns the slug of the first `project:` tag, or `GLOBAL_BUCKET` when none. Reused by the save path (Task 2) and the migration (Task 5) so they never diverge.

- [ ] **Step 1: Write the failing test**

Create/append `tests/test_project.py`:

```python
from memo.project import GLOBAL_BUCKET, project_bucket


def test_project_bucket_returns_slug_of_first_project_tag():
    assert project_bucket(["note", "project:memo", "db"]) == "memo"


def test_project_bucket_untagged_is_global():
    assert project_bucket(["note", "db"]) == GLOBAL_BUCKET


def test_project_bucket_empty_tags_is_global():
    assert project_bucket([]) == "_global"


def test_global_bucket_constant_value():
    assert GLOBAL_BUCKET == "_global"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_project.py -q`
Expected: FAIL with `ImportError: cannot import name 'GLOBAL_BUCKET'`

- [ ] **Step 3: Write minimal implementation**

In `src/memo/project.py`, after `slugify_project` (~line 29), add:

```python
GLOBAL_BUCKET = "_global"


def project_bucket(tags: list[str]) -> str:
    """On-disk folder bucket for a memory: the project slug, or `_global`.

    Derived from the first `project:` tag (already slugified at save time).
    Memories with no project tag share the `_global` bucket. This is the one
    mapping used by both the save path and `memo migrate --bucket-by-project`,
    so on-disk layout never diverges from the tag.
    """
    for tag in tags:
        if tag.startswith(_PROJECT_PREFIX):
            slug = tag[len(_PROJECT_PREFIX) :]
            return slug or GLOBAL_BUCKET
    return GLOBAL_BUCKET
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_project.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/memo/project.py tests/test_project.py
git commit -m "feat: project_bucket helper (tag -> folder mapping)"
```

---

### Task 2: Save writes the `.md` into the per-project bucket folder

**Files:**
- Modify: `src/memo/flags_misc.py` (add `MEMO_STORE_BY_PROJECT` spec next to `MEMO_SINGLE_DB`, ~line 312)
- Modify: `src/memo/memory/write_ops.py` (`_build_rel_path` ~line 792; call site ~line 455)
- Test: `tests/test_write_ops_buckets.py` (create)

**Interfaces:**
- Consumes: `project_bucket` (Task 1), `flag_bool` (already imported in write_ops), `_slugify` (already imported).
- Produces: saved `MemoryRecord.path` is `"<bucket>/<date>-<slug>.md"` when `MEMO_STORE_BY_PROJECT` is on, else the flat `"<date>-<slug>.md"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_write_ops_buckets.py`:

```python
from memo.memory.facade import Memory


def test_save_buckets_md_under_project_folder(tmp_cfg, monkeypatch):
    monkeypatch.setenv("MEMO_STORE_BY_PROJECT", "1")
    mem = Memory(tmp_cfg)
    rec = mem.save(
        content="cuerpo",
        title="Hola",
        type_="note",
        tags=["project:memo"],
        auto_project=False,
        defer_embed=True,
    )
    assert rec.path.startswith("memo/")
    assert (tmp_cfg.memory_dir / rec.path).is_file()


def test_save_untagged_goes_to_global_bucket(tmp_cfg, monkeypatch):
    monkeypatch.setenv("MEMO_STORE_BY_PROJECT", "1")
    mem = Memory(tmp_cfg)
    rec = mem.save(
        content="cuerpo", title="Hola", type_="note", auto_project=False, defer_embed=True
    )
    assert rec.path.startswith("_global/")
    assert (tmp_cfg.memory_dir / rec.path).is_file()


def test_store_by_project_off_keeps_flat_layout(tmp_cfg, monkeypatch):
    monkeypatch.setenv("MEMO_STORE_BY_PROJECT", "0")
    mem = Memory(tmp_cfg)
    rec = mem.save(
        content="cuerpo",
        title="Hola",
        type_="note",
        tags=["project:memo"],
        auto_project=False,
        defer_embed=True,
    )
    assert "/" not in rec.path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_write_ops_buckets.py -q`
Expected: FAIL (paths have no `memo/` / `_global/` prefix because bucketing isn't implemented)

- [ ] **Step 3a: Add the flag**

In `src/memo/flags_misc.py`, immediately after the `MEMO_SINGLE_DB` spec (~line 312), add a spec using the SAME group string as its `MEMO_SINGLE_DB` / `MEMO_MEMORIES_IN_VAULT` neighbors:

```python
    _spec(
        "MEMO_STORE_BY_PROJECT",
        "bool",
        True,
        "misc",
        "Store new memory .md files in a per-project folder "
        "(memory_dir/<project>/, or _global/ when untagged) derived from the "
        "project: tag. The sqlite index globs recursively so search stays "
        "global — this is on-disk organization only. Existing flat files are "
        "untouched until `memo migrate --bucket-by-project`.",
    ),
```

(If the neighbors pass a different 4th-arg group than `"misc"`, copy theirs verbatim.)

- [ ] **Step 3b: Bucket the path in `_build_rel_path`**

In `src/memo/memory/write_ops.py`, replace `_build_rel_path` (~lines 792-809) with:

```python
    def _build_rel_path(
        self, title: str, now_iso: str, tags: list[str] | None = None
    ) -> str:
        date = now_iso.split("T", 1)[0]
        slug = _slugify(title)[:80] or "untitled"
        # Per-project bucket folder, derived from the project: tag. The sqlite
        # index globs recursively, so this is on-disk organization only — search
        # stays global. Gated; flat and foldered layouts coexist.
        prefix = ""
        if flag_bool("MEMO_STORE_BY_PROJECT"):
            from memo.project import project_bucket

            prefix = f"{project_bucket(tags or [])}/"
        # POSIX path joins. Path is relative to `cfg.memory_dir`.
        base = f"{prefix}{date}-{slug}"
        candidate = f"{base}.md"
        # `meta.path` is UNIQUE. Two saves with the same title on the same day
        # would collide. Append a numeric suffix until the path is free —
        # checking both the index and the on-disk file (per-bucket).
        n = 2
        while (
            self.store.get_by_path(candidate) is not None
            or (self.cfg.memory_dir / candidate).exists()
        ):
            candidate = f"{base}-{n}.md"
            n += 1
        return candidate
```

- [ ] **Step 3c: Pass tags at the call site**

In `src/memo/memory/write_ops.py` ~line 455, change:

```python
            rel_path = existing_path if existing_path else self._build_rel_path(title, now_iso)
```

to:

```python
            rel_path = (
                existing_path
                if existing_path
                else self._build_rel_path(title, now_iso, norm_tags)
            )
```

(`norm_tags` is in scope here — it already carries the auto-added `project:` tag from ~line 325. The existing `abs_path.parent.mkdir(parents=True, exist_ok=True)` at ~line 457 creates the bucket folder.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_write_ops_buckets.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Validate flags + commit**

Run: `uv run --no-sync memo config validate` (expect no errors), then `uv run --no-sync ruff check src/ tests/` and `uv run --no-sync mypy src/memo/`.

```bash
git add src/memo/flags_misc.py src/memo/memory/write_ops.py tests/test_write_ops_buckets.py
git commit -m "feat: store new memories in per-project folders (MEMO_STORE_BY_PROJECT)"
```

---

### Task 3: `_apply_project_tiers` — 3-tier soft ranking + flags

**Files:**
- Modify: `src/memo/flags_recall.py` (`MEMO_RECALL_PROJECT_BOOST` default 0.15→0.25, ~line 56; add `MEMO_RECALL_GLOBAL_BOOST`)
- Modify: `src/memo/recall_logic.py` (add `_apply_project_tiers` after `_apply_project_boost`, ~line 142)
- Test: `tests/test_project_tiers.py` (create)

**Interfaces:**
- Consumes: `has_project_tag` (`memo.project`), `replace` (already imported in recall_logic).
- Produces: `_apply_project_tiers(hits: list, project_tag: str | None, project_boost: float, global_boost: float) -> list` — returns a new list, re-sorted by boosted score. Tier precedence: global/cross-cutting (no `project:` tag OR type in `{preference, feedback}`) → `+global_boost`; else current project (`project_tag` in tags) → `+project_boost`; else `+0`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_project_tiers.py`:

```python
from dataclasses import dataclass, field

from memo.recall_logic import _apply_project_tiers


@dataclass(frozen=True)
class _Hit:
    id: str
    title: str = ""
    body: str = ""
    type: str = "note"
    score: float | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


def test_current_project_outranks_other_at_equal_similarity():
    hits = [
        _Hit("other", score=0.80, tags=("project:synapse",)),
        _Hit("cur", score=0.80, tags=("project:memo",)),
    ]
    out = _apply_project_tiers(hits, "project:memo", 0.25, 0.10)
    assert out[0].id == "cur"  # 0.80+0.25 beats 0.80


def test_global_preference_stays_afloat_over_other_project():
    hits = [
        _Hit("other", score=0.70, type="note", tags=("project:synapse",)),
        _Hit("pref", score=0.65, type="preference", tags=()),
    ]
    out = _apply_project_tiers(hits, "project:memo", 0.25, 0.10)
    assert out[0].id == "pref"  # 0.65+0.10=0.75 beats 0.70 (other gets +0)


def test_much_more_similar_other_still_wins_soft():
    hits = [
        _Hit("cur", score=0.60, tags=("project:memo",)),
        _Hit("other", score=0.95, tags=("project:synapse",)),
    ]
    out = _apply_project_tiers(hits, "project:memo", 0.25, 0.10)
    assert out[0].id == "other"  # 0.95 beats 0.60+0.25=0.85 (soft, not a hard filter)


def test_preference_with_current_project_tag_uses_global_tier():
    # Precedence: preference/feedback -> tier-2 even with the current project tag.
    hits = [_Hit("p", score=0.50, type="preference", tags=("project:memo",))]
    out = _apply_project_tiers(hits, "project:memo", 0.25, 0.10)
    assert out[0].score == 0.60  # +0.10 (global), NOT +0.25 (project)


def test_none_score_hits_pass_through_untouched():
    hits = [_Hit("n", score=None, tags=("project:memo",))]
    out = _apply_project_tiers(hits, "project:memo", 0.25, 0.10)
    assert out[0].score is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_project_tiers.py -q`
Expected: FAIL with `ImportError: cannot import name '_apply_project_tiers'`

- [ ] **Step 3a: Add `_apply_project_tiers`**

In `src/memo/recall_logic.py`, after `_apply_project_boost` (~line 142), add:

```python
_GLOBAL_TIER_TYPES = {"preference", "feedback"}


def _apply_project_tiers(
    hits: list[Any],
    project_tag: str | None,
    project_boost: float,
    global_boost: float,
) -> list[Any]:
    """3-tier soft project ranking, re-sorted by boosted score.

    Per-hit precedence (a hit may match several tiers):
      - tier-2 global/cross-cutting: no `project:` tag OR type in
        {preference, feedback}                       -> +global_boost
        (wins over tier-1 even with a project tag)
      - tier-1 current project: `project_tag` in tags -> +project_boost
      - tier-3 other projects: everything else        -> +0

    Additive + soft: a much-more-similar global / other-project hit still wins,
    so the search pool stays effectively "one folder" with relevance weighting.
    """
    from memo.project import has_project_tag

    out: list[Any] = []
    for h in hits:
        if h.score is None:
            out.append(h)
            continue
        tags = h.tags or []
        is_global = (not has_project_tag(list(tags))) or (
            getattr(h, "type", "") in _GLOBAL_TIER_TYPES
        )
        if is_global:
            out.append(replace(h, score=h.score + global_boost))
        elif project_tag and project_tag in tags:
            out.append(replace(h, score=h.score + project_boost))
        else:
            out.append(h)
    out.sort(key=lambda h: h.score or 0.0, reverse=True)
    return out
```

- [ ] **Step 3b: Update flags**

In `src/memo/flags_recall.py`, change the `MEMO_RECALL_PROJECT_BOOST` default from `0.15` to `0.25` (and its description to mention tier-1). Immediately after that spec, add:

```python
    _spec(
        "MEMO_RECALL_GLOBAL_BOOST",
        "float",
        0.10,
        "recall",
        "Score boost for global / cross-cutting memories (no project: tag, or "
        "type preference/feedback) so they stay afloat in any project. Tier-2 of "
        "the 3-tier soft project ranking (current > global > other-projects).",
        min_val=0.0,
        max_val=1.0,
    ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_project_tiers.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/memo/recall_logic.py src/memo/flags_recall.py tests/test_project_tiers.py
git commit -m "feat: _apply_project_tiers 3-tier soft recall ranking + MEMO_RECALL_GLOBAL_BOOST"
```

---

### Task 4: Wire the tiers into the recall paths

**Files:**
- Modify: `src/memo/recall_logic.py` (recall path ~lines 234-235, 312)
- Modify: `src/memo/cli_recall_hook.py` (~lines 210-211, 281-283)
- Modify: `src/memo/recall_server.py` (export `_apply_project_tiers`, ~lines 17, 56)
- Test: `tests/test_recall_tiers_wiring.py` (create)

**Interfaces:**
- Consumes: `_apply_project_tiers` (Task 3).
- Produces: both the in-process recall path and the recall-hook path apply 3-tier boosts; `_apply_project_tiers` is importable from `memo.recall_server` (the import the hook uses).

- [ ] **Step 1: Write the failing test**

Create `tests/test_recall_tiers_wiring.py`:

```python
def test_recall_server_reexports_apply_project_tiers():
    from memo.recall_server import _apply_project_tiers

    assert callable(_apply_project_tiers)


def test_project_boost_default_is_025():
    from memo.flags import flag_float

    # default comes from the registry when unset
    assert flag_float("MEMO_RECALL_PROJECT_BOOST") == 0.25


def test_global_boost_default_is_010():
    from memo.flags import flag_float

    assert flag_float("MEMO_RECALL_GLOBAL_BOOST") == 0.10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_recall_tiers_wiring.py -q`
Expected: FAIL (import error for `_apply_project_tiers` from recall_server; and the boost defaults still resolve to old values via hardcoded fallbacks)

- [ ] **Step 3a: Re-export from `recall_server.py`**

In `src/memo/recall_server.py`, add `_apply_project_tiers` to the import block from `recall_logic` (next to `_apply_project_boost`, ~line 17) and to `__all__` (~line 56).

- [ ] **Step 3b: Wire the in-process recall path (`recall_logic.py`)**

At ~lines 234-235, change the hardcoded fallback and add the global-boost read:

```python
    _pb = _flag_float("MEMO_RECALL_PROJECT_BOOST")
    project_boost = 0.25 if _pb is None else _pb
    _gb = _flag_float("MEMO_RECALL_GLOBAL_BOOST")
    global_boost = 0.10 if _gb is None else _gb
```

At ~line 312, change the call from `_apply_project_boost(raw, project_tag, project_boost)` to:

```python
            raw = _apply_project_tiers(raw, project_tag, project_boost, global_boost)
```

Note: the function this lives in already gates on `project_tag` before line 312 — keep that guard, but `_apply_project_tiers` is also safe to call with `project_tag=None` (it still lifts global hits), so if the guard is `if project_tag:` consider widening it to always call when `global_boost > 0`. Minimal change: leave the existing guard; the global tier still applies whenever `project_tag` is set, which covers the in-repo case.

- [ ] **Step 3c: Wire the recall-hook path (`cli_recall_hook.py`)**

At ~lines 210-211, change the fallback and add the global read:

```python
    _pb = flag_float("MEMO_RECALL_PROJECT_BOOST")
    project_boost = 0.25 if _pb is None else _pb
    _gb = flag_float("MEMO_RECALL_GLOBAL_BOOST")
    global_boost = 0.10 if _gb is None else _gb
```

At ~lines 280-283, change the block to call the tiers function (and apply it whenever global_boost or project_tag is active so global memories surface even outside a repo):

```python
        if project_tag or global_boost > 0:
            from memo.recall_server import _apply_project_tiers

            hits = _apply_project_tiers(hits, project_tag, project_boost, global_boost)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_recall_tiers_wiring.py tests/test_recall_hooks.py tests/test_recall_server.py -q`
Expected: PASS (the new wiring tests pass; the existing `_apply_project_boost` tests still pass — that function is untouched)

- [ ] **Step 5: Commit**

```bash
git add src/memo/recall_logic.py src/memo/cli_recall_hook.py src/memo/recall_server.py tests/test_recall_tiers_wiring.py
git commit -m "feat: wire 3-tier project ranking into recall + recall-hook paths"
```

---

### Task 5: `memo migrate --bucket-by-project`

**Files:**
- Modify: `src/memo/runtime/migrate.py` (add `--bucket-by-project` option + `_bucket_by_project` helper)
- Test: `tests/test_migrate_buckets.py` (create)

**Interfaces:**
- Consumes: `project_bucket` (Task 1), `Config`, `Memory.reindex`.
- Produces: `memo migrate --bucket-by-project` moves flat-root `.md` files into `<bucket>/` by their `project:` tag (untagged → `_global/`), then reindexes. Idempotent (files already in a bucket are skipped), non-destructive (moves only).

- [ ] **Step 1: Write the failing test**

Create `tests/test_migrate_buckets.py`:

```python
import frontmatter

from memo.runtime.migrate import _bucket_by_project


def _write_flat(memory_dir, name, tags):
    memory_dir.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post("body", id=name, title=name, type="note", tags=tags)
    (memory_dir / f"{name}.md").write_text(frontmatter.dumps(post), encoding="utf-8")


def test_bucket_by_project_moves_tagged_and_untagged(tmp_cfg):
    md = tmp_cfg.memory_dir
    _write_flat(md, "a", ["project:memo"])
    _write_flat(md, "b", [])
    moved = _bucket_by_project(tmp_cfg)
    assert (md / "memo" / "a.md").is_file()
    assert (md / "_global" / "b.md").is_file()
    assert not (md / "a.md").exists()
    assert moved == 2


def test_bucket_by_project_is_idempotent(tmp_cfg):
    md = tmp_cfg.memory_dir
    _write_flat(md, "a", ["project:memo"])
    _bucket_by_project(tmp_cfg)
    moved_again = _bucket_by_project(tmp_cfg)  # already bucketed -> 0
    assert moved_again == 0
    assert (md / "memo" / "a.md").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_migrate_buckets.py -q`
Expected: FAIL with `ImportError: cannot import name '_bucket_by_project'`

- [ ] **Step 3a: Add the `_bucket_by_project` helper**

In `src/memo/runtime/migrate.py`, add (module level, near `_consolidate_sidecar_dbs`):

```python
def _bucket_by_project(cfg: "Config") -> int:
    """Move flat-root .md files into per-project bucket folders by their
    `project:` tag (untagged -> `_global/`). Idempotent + non-destructive.
    Returns the number of files moved.
    """
    import frontmatter

    from memo.project import project_bucket

    md_root = cfg.memory_dir
    moved = 0
    # Only the FLAT root level (already-bucketed files live one level deeper
    # and are skipped, making this idempotent).
    for md in sorted(md_root.glob("*.md")):
        try:
            post = frontmatter.loads(md.read_text(encoding="utf-8"))
        except Exception:
            continue
        tags = list(post.get("tags") or [])
        bucket = project_bucket(tags)
        dest_dir = md_root / bucket
        dest = dest_dir / md.name
        if dest.resolve() == md.resolve():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue  # name collision in bucket — leave the original in place
        md.rename(dest)
        moved += 1
    return moved
```

- [ ] **Step 3b: Add the CLI flag + early-return branch**

In `src/memo/runtime/migrate.py`, add a Click option to `migrate_vault` (next to `--consolidate-db`):

```python
@click.option(
    "--bucket-by-project",
    is_flag=True,
    help="Move existing flat .md files into per-project folders "
    "(memory_dir/<project>/, _global/ when untagged) by their project: tag, "
    "then reindex. Non-destructive (moves only), idempotent.",
)
```

Add `bucket_by_project: bool` to the function signature, and right after the `--consolidate-db` early-return block (~line 147), add:

```python
    if bucket_by_project:
        cfg = Config.from_env()
        moved = _bucket_by_project(cfg)
        console.print(f"[green]✓[/green] bucketed {moved} memory file(s) by project")
        from memo.memory import Memory

        mem = Memory(cfg)
        mem.reindex()
        console.print(
            "[dim]reindexed (paths updated). [[id]] wikilinks are unaffected; "
            "Obsidian path-links to moved files would change.[/dim]"
        )
        return
```

(Confirm `Memory.reindex()` is the correct no-arg facade method that `memo reindex` calls; if its signature differs, call it the same way `memo reindex` does in `src/memo/cli.py`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_migrate_buckets.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/memo/runtime/migrate.py tests/test_migrate_buckets.py
git commit -m "feat: memo migrate --bucket-by-project (flat -> per-project folders)"
```

---

### Task 6: Regression gate + flag docs

**Files:**
- Modify: `CLAUDE.md` (document the 3 new/changed flags + per-project layout, in the Storage / Flags section)
- No new test file — this task runs the existing retrieval-regression gate.

**Interfaces:**
- Consumes: everything above.
- Produces: a green retrieval-regression run + updated docs.

- [ ] **Step 1: Full gate**

Run:
```bash
uv run --no-sync ruff check src/ tests/
uv run --no-sync mypy src/memo/
uv run --no-sync pytest tests/
```
Expected: all green (ruff "All checks passed!", mypy "Success", pytest all passed/skipped).

- [ ] **Step 2: Retrieval-regression gate (no MLX-free shortcut — runs against the live index)**

Run: `uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force`
Expected: precision@5 not lower and noise@5 not higher than the committed baseline (prec@5≈0.2, noise@5≈0.0). If the 3-tier boost regresses a prompt, tune `MEMO_RECALL_PROJECT_BOOST`/`MEMO_RECALL_GLOBAL_BOOST` defaults (a systemic change), never patch a single query.

- [ ] **Step 3: Document the flags**

In `CLAUDE.md`, under the Storage / Config & errors section, add a short note: per-project folders gated by `MEMO_STORE_BY_PROJECT` (default on; tag is source of truth, folder derived; index stays global/recursive); recall ranking is 3-tier soft via `MEMO_RECALL_PROJECT_BOOST` (0.25, tier-1) and `MEMO_RECALL_GLOBAL_BOOST` (0.10, tier-2 = no project tag or preference/feedback); migrate existing installs with `memo migrate --bucket-by-project`.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: per-project storage + 3-tier recall flags"
```

---

## Self-Review

**Spec coverage:**
- "Guardar por proyecto" (folders) → Task 1 (mapping) + Task 2 (save) + Task 5 (migrate existing). ✓
- "Consultar como en la misma carpeta" (global pool) → no index change needed; verified by Task 6 gate + existing recursive glob. ✓
- "Darle relevancia" (3-tier soft) → Task 3 (function) + Task 4 (wiring). ✓
- Global/cross-cutting tier (preference/feedback always tier-2) → Task 3 precedence + test `test_preference_with_current_project_tag_uses_global_tier`. ✓
- Flags `MEMO_STORE_BY_PROJECT`, `MEMO_RECALL_GLOBAL_BOOST`, `MEMO_RECALL_PROJECT_BOOST=0.25` → Tasks 2, 3, 4. ✓
- Migration non-destructive/idempotent → Task 5 + tests. ✓
- Invariants (markdown-as-truth, path stability, recursive index) → preserved (tag-derived folder; no index change). ✓
- Regression discipline → Task 6. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. Two "confirm the neighbor/method" notes (flags_misc group string in Task 2; `Memory.reindex()` signature in Task 5) point at exact, locally-verifiable facts — not deferred design.

**Type consistency:** `project_bucket(list[str]) -> str` used identically in Tasks 2 and 5. `_apply_project_tiers(hits, project_tag, project_boost, global_boost)` signature identical in Task 3 (definition), Task 4 (call sites), and re-export. `GLOBAL_BUCKET`/`_global` consistent across Tasks 1, 2, 5. Flag names spelled identically everywhere.
