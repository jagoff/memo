from memo.ops_launchd import parse_launchctl_list, render_chat_plist


def test_render_chat_plist_contents() -> None:
    plist = render_chat_plist("/usr/local/bin/memo", "/Users/tester", port=8765, dist="/x/dist")
    assert "<key>Label</key>" in plist and "com.memo.chat" in plist
    assert "/usr/local/bin/memo" in plist
    assert "serve" in plist and "8765" in plist and "/x/dist" in plist
    assert "/Users/tester/Library/Logs/memo/chat.log" in plist
    assert "KeepAlive" in plist


def test_render_without_dist_omits_flag() -> None:
    plist = render_chat_plist("/bin/memo", "/Users/t", port=8765, dist=None)
    assert "--dist" not in plist


def test_parse_launchctl_list() -> None:
    raw = "PID\tStatus\tLabel\n50864\t0\tcom.memo.recall-daemon\n-\t0\tcom.memo.nightly\n123\t0\tcom.other.thing\n"
    rows = parse_launchctl_list(raw)
    labels = {r["label"] for r in rows}
    assert labels == {"com.memo.recall-daemon", "com.memo.nightly"}
    recall = next(r for r in rows if r["label"] == "com.memo.recall-daemon")
    assert recall["pid"] == 50864 and recall["last_exit"] == 0
