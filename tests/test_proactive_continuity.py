from memo.proactive.detectors.continuity import detect_continuity
from memo.proactive.nudge import KIND_CONTINUITY


class _FakeMem:
    def open_loops(self, limit):
        return [("m9", "finish int8 magnitude study")]


def test_continuity_wraps_open_loops():
    ns = detect_continuity(_FakeMem(), now="2026-07-21T00:00:00Z")
    assert ns and ns[0].kind == KIND_CONTINUITY
    assert ns[0].evidence == ("m9",)


def test_continuity_guarded():
    class Boom:
        def open_loops(self, limit):
            raise RuntimeError("x")

    assert detect_continuity(Boom(), now="2026-07-21T00:00:00Z") == []


def test_real_facade_open_loops_wraps_recent_memory(mock_memory):
    rec = mock_memory.save(content="finish int8 magnitude study", type_="note")

    loops = mock_memory.open_loops(5)

    assert (rec.id, rec.title) in loops
