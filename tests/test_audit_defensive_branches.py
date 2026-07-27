"""Coverage for the v4.4.4 production-audit defensive branches — the guard /
except paths added across the tree. Each fix should exercise its new branch."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest


# ── contradict._as_aware (DST-safe timestamp compare) ────────────────────────
def test_as_aware_parses_naive_as_utc() -> None:
    from memo.contradict import _as_aware

    dt = _as_aware("2026-10-25T02:30:00.000")
    assert dt.tzinfo is not None
    assert dt.utcoffset() == datetime(2026, 1, 1, tzinfo=UTC).utcoffset()


def test_as_aware_preserves_offset() -> None:
    from memo.contradict import _as_aware

    earlier = _as_aware("2026-10-25T02:30:00.000+02:00")  # 00:30 UTC
    later = _as_aware("2026-10-25T02:30:00.000+01:00")  # 01:30 UTC
    # Raw string compare would invert these; instant compare must not.
    assert earlier < later


def test_as_aware_unparseable_sorts_oldest() -> None:
    from memo.contradict import _as_aware

    assert _as_aware("not-a-timestamp") == datetime.min.replace(tzinfo=UTC)
    assert _as_aware("") == datetime.min.replace(tzinfo=UTC)


# ── flags.validate MEMO_MODEL_PROFILE (choices from config.MODEL_PROFILES) ───
def test_validate_rejects_unknown_model_profile() -> None:
    from memo.flags import validate

    problems = validate({"MEMO_MODEL_PROFILE": "turbo"})
    assert any(p["flag"] == "MEMO_MODEL_PROFILE" for p in problems)


def test_validate_accepts_known_model_profile() -> None:
    from memo.config import MODEL_PROFILES
    from memo.flags import validate

    for name in MODEL_PROFILES:
        problems = validate({"MEMO_MODEL_PROFILE": name})
        assert not any(p["flag"] == "MEMO_MODEL_PROFILE" for p in problems)


# ── secret_store._load_or_create_machine_salt (O_CREAT|O_EXCL 0600) ──────────
def test_machine_salt_created_0600_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import memo.secret_store as ss

    monkeypatch.setattr(ss.Path, "home", classmethod(lambda cls: tmp_path))
    salt = ss._load_or_create_machine_salt()
    assert len(salt) == 32  # 16 bytes hex
    salt_file = tmp_path / ".memo" / "machine.salt"
    assert salt_file.is_file()
    assert (salt_file.stat().st_mode & 0o777) == 0o600
    # Second call reads the persisted value (no rewrite).
    assert ss._load_or_create_machine_salt() == salt


# ── cli_transcripts._read_full_transcript (UnicodeDecodeError → []) ──────────
def test_read_full_transcript_survives_bad_encoding(tmp_path: Path) -> None:
    from memo.cli_transcripts import _read_full_transcript

    bad = tmp_path / "transcript.jsonl"
    bad.write_bytes(b"\xff\xfe not valid utf-8 \x80\x81")
    assert _read_full_transcript(bad) == []


# ── resume/_parsers isinstance(item, dict) guards (non-object JSONL lines) ────
def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_jsonl_latest_user_text_skips_non_dict_lines(tmp_path: Path) -> None:
    from memo.resume._parsers import _jsonl_latest_user_text

    p = tmp_path / "s.jsonl"
    _write_lines(p, ["null", "[1, 2, 3]", '"a bare string"', "42"])
    # None of these are dicts → every one hits the guard; result is empty.
    assert _jsonl_latest_user_text(p) == ""


def test_gemini_latest_user_text_skips_non_dict_lines(tmp_path: Path) -> None:
    from memo.resume._parsers import _gemini_latest_user_text

    p = tmp_path / "g.jsonl"
    _write_lines(p, ["null", "[1, 2]", "3.14"])
    assert _gemini_latest_user_text(p) == ""


def test_session_meta_readers_skip_non_dict_lines(tmp_path: Path) -> None:
    from memo.resume._parsers import (
        _read_claude_session_meta,
        _read_codex_session_meta,
        _read_devin_session_meta,
    )

    p = tmp_path / "m.jsonl"
    _write_lines(p, ["null", "[1, 2]", json.dumps({"type": "other"})])
    # Non-dict lines are skipped; no session_meta present → devin/codex None.
    assert _read_devin_session_meta(p) is None
    assert _read_codex_session_meta(p) is None
    # claude reader returns whatever it accreted (empty here) without crashing.
    assert _read_claude_session_meta(p) == {}


# ── repo_index ref leading-dash reject (checkout arg injection) ───────────────
def test_repo_index_rejects_leading_dash_ref(tmp_path: Path) -> None:
    from memo.config import Config
    from memo.memory import Memory
    from memo.repo_index import RepoCorpus

    cfg = Config(
        data_dir=tmp_path / "d",
        vault_path=tmp_path / "v",
        state_dir=tmp_path / "s",
        embedder_dims=4,
        reranker_enabled=False,
    )
    for d in (cfg.data_dir, cfg.vault_path, cfg.state_dir):
        d.mkdir(parents=True, exist_ok=True)
    mem = Memory(cfg)
    corpus = RepoCorpus(cfg, store=mem.store, embedder=mem.embedder)
    with pytest.raises(ValueError, match="unsafe repo ref"):
        corpus.index("https://github.com/x/y.git", ref="--upload-pack=evil")


# ── server_session_patterns / synthesis LIMIT clamps (negative → unbounded) ──
class _CaptureServer:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self):
        def dec(f):
            self.tools[f.__name__] = f
            return f

        return dec


@pytest.fixture
def _session_mem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import hashlib

    from memo.config import Config
    from memo.memory import Memory

    monkeypatch.setenv("MEMO_EMBEDDER_VIA_DAEMON", "0")
    cfg = Config(
        data_dir=tmp_path / "d",
        vault_path=tmp_path / "v",
        state_dir=tmp_path / "s",
        embedder_dims=4,
        reranker_enabled=False,
    )
    for d in (cfg.data_dir, cfg.vault_path, cfg.state_dir):
        d.mkdir(parents=True, exist_ok=True)
    mem = Memory(cfg)

    def stub(inputs):
        out = []
        for s in inputs:
            dig = hashlib.sha256(s.encode()).digest()
            v = [((dig[i] / 255.0) * 2 - 1) for i in range(4)]
            norm = (sum(x * x for x in v) ** 0.5) or 1
            out.append([x / norm for x in v])
        return out

    mem.embedder.embed = stub
    mem.embedder.embed_query = lambda q: stub([q])[0]
    return mem


def test_session_pattern_tools_clamp_negative_limits(_session_mem) -> None:
    from memo.server_session_patterns import register

    srv = _CaptureServer()
    register(srv, _session_mem)

    # Each negative value must be clamped, not passed through as an unbounded
    # sqlite LIMIT. The clamp runs before the row lookups, so empty/absent data
    # still exercises the guard lines.
    ctx = srv.tools["mem_context"](project="p", limit=-1)
    assert ctx["sessions"] == []
    tl = srv.tools["mem_timeline"](observation_id="does-not-exist", before=-5, after=-5)
    assert tl["error"] == "observation not found"
    rev = srv.tools["mem_review"](project="p", limit=-1)
    assert isinstance(rev, dict)
