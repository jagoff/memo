from pathlib import Path

from memo.proxy.plan import Context, apply_all
from memo.proxy.zones import split


class _Good:
    name = "good"
    zone = "live"

    def enabled(self) -> bool:
        return True

    def apply(self, zones, ctx) -> int:
        zones.live_messages.clear()
        return 42


class _Boom:
    name = "boom"
    zone = "live"

    def enabled(self) -> bool:
        return True

    def apply(self, zones, ctx) -> int:
        raise RuntimeError("transform exploded")


class _Off:
    name = "off"
    zone = "live"

    def enabled(self) -> bool:
        return False

    def apply(self, zones, ctx) -> int:
        raise AssertionError("must not run")


def _ctx(tmp_path: Path) -> Context:
    return Context(state_dir=tmp_path, session_key="s1", project="memo")


def test_applied_transform_is_reported_with_its_saving(tmp_path):
    zones = split({"messages": [{"role": "user", "content": "x"}]})
    result = apply_all(zones, _ctx(tmp_path), [_Good()])
    assert result.applied == ["good"]
    assert result.est_saved_tokens == 42


def test_a_raising_transform_is_skipped_not_propagated(tmp_path):
    zones = split({"messages": [{"role": "user", "content": "x"}]})
    result = apply_all(zones, _ctx(tmp_path), [_Boom(), _Good()])
    assert result.applied == ["good"]
    assert "boom" not in result.applied


def test_a_disabled_transform_never_runs(tmp_path):
    zones = split({"messages": [{"role": "user", "content": "x"}]})
    result = apply_all(zones, _ctx(tmp_path), [_Off()])
    assert result.applied == []
    assert result.est_saved_tokens == 0


class _Garbage:
    name = "garbage"
    zone = "live"

    def enabled(self) -> bool:
        return True

    def apply(self, zones, ctx):
        return "abc"


class _NoneReturn:
    name = "none"
    zone = "live"

    def enabled(self) -> bool:
        return True

    def apply(self, zones, ctx):
        return None


def test_a_non_numeric_return_does_not_propagate(tmp_path):
    zones = split({"messages": [{"role": "user", "content": "x"}]})
    result = apply_all(zones, _ctx(tmp_path), [_Garbage()])
    assert result.est_saved_tokens == 0


def test_a_none_return_counts_as_zero_saved(tmp_path):
    zones = split({"messages": [{"role": "user", "content": "x"}]})
    result = apply_all(zones, _ctx(tmp_path), [_NoneReturn()])
    assert result.est_saved_tokens == 0


def test_a_garbage_return_still_leaves_the_planner_usable(tmp_path):
    zones = split({"messages": [{"role": "user", "content": "x"}]})
    result = apply_all(zones, _ctx(tmp_path), [_Garbage(), _Good()])
    assert result.est_saved_tokens == 42


class _RaisingName:
    """`apply()` fails AND `name` is a property that also raises on access —
    the exact shape of a transform that would blow up an unguarded
    `transform.name` read inside the `except` handler."""

    zone = "live"

    def enabled(self) -> bool:
        return True

    def apply(self, zones, ctx) -> int:
        raise RuntimeError("apply exploded")

    @property
    def name(self) -> str:
        raise RuntimeError("name exploded too")


def test_a_transform_whose_name_also_raises_does_not_propagate(tmp_path):
    """`apply_all`'s except-handler reads `transform.name` to log which
    transform failed. If that access itself raises (a transform exposing
    `name` as a raising property), the failure must still be contained —
    skip the transform, keep running the plan — not crash the request the
    plan module exists to protect."""
    zones = split({"messages": [{"role": "user", "content": "x"}]})
    result = apply_all(zones, _ctx(tmp_path), [_RaisingName(), _Good()])
    assert result.applied == ["good"]
    assert result.est_saved_tokens == 42
