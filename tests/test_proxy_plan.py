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
