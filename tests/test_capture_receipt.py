from memo.cli_capture import _write_capture_notification


def test_receipt_lists_ids_and_verbs_when_flag_on(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_CAPTURE_RECEIPT", "1")
    saved = [{"id": "a1b2c3d4", "title": "Guard design", "type": "decision"}]
    _write_capture_notification(tmp_path, saved)
    txt = (tmp_path / "pending_idle_notification.txt").read_text()
    assert "a1b2c3d4" in txt
    assert "Guard design" in txt
    assert "memo undo" in txt and "memo fix" in txt


def test_receipt_legacy_line_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMO_CAPTURE_RECEIPT", raising=False)
    _write_capture_notification(tmp_path, [{"id": "x", "title": "t", "type": "note"}])
    txt = (tmp_path / "pending_idle_notification.txt").read_text()
    assert txt.strip() == "※ MEMO auto-saved"


def test_receipt_silent_when_nothing_saved(tmp_path):
    _write_capture_notification(tmp_path, [])
    assert not (tmp_path / "pending_idle_notification.txt").exists()
