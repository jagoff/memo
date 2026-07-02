"""build_system_message — the human-visible 🧠 presence line (F1a)."""

from types import SimpleNamespace

from memo.recall_logic import build_system_message


def _hit(id_: str, title: str) -> SimpleNamespace:
    return SimpleNamespace(id=id_, title=title, body="", score=0.9)


def test_empty_hits_returns_empty_string() -> None:
    assert build_system_message([]) == ""


def test_counts_and_titles() -> None:
    msg = build_system_message([_hit("a1b2c3d4e5", "sync tier decision"), _hit("f6e5d4c3b2", "delete rollback bug")])
    assert msg.startswith("🧠 memo · 2: ")
    assert "sync tier decision" in msg
    assert "delete rollback bug" in msg


def test_truncates_to_max_chars_with_ellipsis() -> None:
    hits = [_hit(f"{i:08x}", "a very long memory title that keeps going " * 3) for i in range(5)]
    msg = build_system_message(hits, max_chars=80)
    assert len(msg) <= 80
    assert msg.endswith("…")


def test_untitled_hit_falls_back_to_short_id() -> None:
    msg = build_system_message([_hit("a1b2c3d4e5f6", "")])
    assert "a1b2c3d4" in msg


def test_flag_default_is_on(monkeypatch) -> None:
    from memo.flags import flag_bool

    monkeypatch.delenv("MEMO_RECALL_SYSTEM_MESSAGE", raising=False)
    assert flag_bool("MEMO_RECALL_SYSTEM_MESSAGE") is True
    monkeypatch.setenv("MEMO_RECALL_SYSTEM_MESSAGE", "0")
    assert flag_bool("MEMO_RECALL_SYSTEM_MESSAGE") is False


def test_cite_instruction_constant() -> None:
    from memo.recall_logic import CITE_INSTRUCTION

    # Task 3's parser depends on the [hex8] cite format this line teaches.
    assert "[a1b2c3d4]" in CITE_INSTRUCTION
    assert CITE_INSTRUCTION.startswith("_") and CITE_INSTRUCTION.endswith("_")
