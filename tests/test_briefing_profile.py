"""profile_lines — El Briefing reads the dream-maintained profile.md (B2)."""

from __future__ import annotations

from pathlib import Path

from memo.config import Config


def _cfg(tmp_path: Path) -> Config:
    data = tmp_path / "data"
    state = tmp_path / "state"
    data.mkdir(exist_ok=True)
    state.mkdir(exist_ok=True)
    return Config(data_dir=data, state_dir=state, reranker_enabled=False)


def _write_profile(cfg: Config, name: str, body: str) -> None:
    d = cfg.memory_dir / "_profile"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(
        f"---\ngenerated_by: memo dream profile\nscope: global\n---\n{body}\n",
        encoding="utf-8",
    )


def test_missing_profile_yields_no_lines(tmp_path):
    from memo.briefing import profile_lines

    assert profile_lines(_cfg(tmp_path)) == []


def test_global_profile_injected_frontmatter_stripped(tmp_path):
    from memo.briefing import profile_lines

    cfg = _cfg(tmp_path)
    _write_profile(cfg, "profile.md", "# Profile — global\n- prefers Spanish replies")
    joined = "\n".join(profile_lines(cfg))
    assert "- prefers Spanish replies" in joined
    assert "generated_by" not in joined  # frontmatter is metadata, not context


def test_project_profile_included_for_current_project(tmp_path, monkeypatch):
    from memo.briefing import profile_lines

    cfg = _cfg(tmp_path)
    _write_profile(cfg, "project-memo.md", "- memo: always uv run --no-sync")
    monkeypatch.setenv("MEMO_PROJECT_TAG", "memo")  # pin project detection
    joined = "\n".join(profile_lines(cfg))
    assert "always uv run --no-sync" in joined


def test_oversized_profile_is_capped(tmp_path):
    from memo.briefing import profile_lines

    cfg = _cfg(tmp_path)
    _write_profile(cfg, "profile.md", "x" * 20000)  # hand-edited runaway file
    joined = "\n".join(profile_lines(cfg))
    assert len(joined) < 7000


def test_native_briefing_includes_profile_by_default(mock_memory):
    from memo.briefing import memo_native_briefing_lines

    _write_profile(mock_memory.cfg, "profile.md", "- standing: pin transformers<5.13")
    joined = "\n".join(memo_native_briefing_lines(mock_memory))
    assert "pin transformers<5.13" in joined


def test_native_briefing_respects_opt_out(mock_memory, monkeypatch):
    from memo.briefing import memo_native_briefing_lines

    _write_profile(mock_memory.cfg, "profile.md", "- standing: pin transformers<5.13")
    monkeypatch.setenv("MEMO_BRIEFING_PROFILE", "0")
    joined = "\n".join(memo_native_briefing_lines(mock_memory))
    assert "pin transformers<5.13" not in joined
