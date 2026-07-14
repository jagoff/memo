from memo.graduation.registry import default_candidates


def test_default_candidates_are_well_formed():
    cands = default_candidates()
    assert cands, "expected at least one seed candidate"
    for c in cands:
        assert c.flag.startswith("MEMO_")
        assert c.on_flags, "on_flags must express the ON state"
        # OFF state is derived by zeroing every on_flags key — they must be the
        # same flags, so the delta is attributable to the candidate alone.
        assert all(k.startswith("MEMO_") for k in c.on_flags)
        assert c.k >= 1
        assert c.epsilon >= 0.0


def test_candidate_is_frozen():
    c = default_candidates()[0]
    import dataclasses

    assert dataclasses.is_dataclass(c)
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        c.flag = "x"  # type: ignore[misc]
