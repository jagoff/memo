from dataclasses import dataclass
from pathlib import Path

from memo.graduation import controller, overlay_ops
from memo.graduation.registry import Candidate


@dataclass
class _Cfg:
    state_dir: Path


def _stub(win: bool):
    def _e(mem, cand, *, k, labels):
        return {"win": win, "delta_prec": 0.02 if win else -0.02, "delta_noise": 0.0}
    return _e


CAND = Candidate(flag="MEMO_GRAPH_SIGNAL_ENABLED",
                 on_flags={"MEMO_GRAPH_SIGNAL_ENABLED": "1"}, k=3, auto_flip=True)


def test_flips_on_after_k_winning_nights(tmp_path: Path):
    cfg = _Cfg(tmp_path)
    for _ in range(2):
        r = controller.run_graduation_controller(
            cfg, object(), evaluator=_stub(True), candidates=[CAND], env={})
        assert r["candidates"][0]["status"] == "accumulating"
    assert not overlay_ops.is_flipped_on(tmp_path, CAND.flag)
    r = controller.run_graduation_controller(
        cfg, object(), evaluator=_stub(True), candidates=[CAND], env={})
    assert r["candidates"][0]["status"] == "graduated"
    assert overlay_ops.is_flipped_on(tmp_path, CAND.flag)


def test_reverts_when_live_config_regresses(tmp_path: Path):
    cfg = _Cfg(tmp_path)
    for _ in range(3):  # graduate first
        controller.run_graduation_controller(
            cfg, object(), evaluator=_stub(True), candidates=[CAND], env={})
    assert overlay_ops.is_flipped_on(tmp_path, CAND.flag)
    r = controller.run_graduation_controller(  # a losing night while live
        cfg, object(), evaluator=_stub(False), candidates=[CAND], env={})
    assert r["candidates"][0]["status"] == "reverted"
    assert not overlay_ops.is_flipped_on(tmp_path, CAND.flag)


def test_env_var_vetoes(tmp_path: Path):
    cfg = _Cfg(tmp_path)
    r = controller.run_graduation_controller(
        cfg, object(), evaluator=_stub(True), candidates=[CAND],
        env={"MEMO_GRAPH_SIGNAL_ENABLED": "0"})
    assert r["candidates"][0]["status"] == "vetoed"
    assert not overlay_ops.is_flipped_on(tmp_path, CAND.flag)


def test_report_only_never_flips(tmp_path: Path):
    cfg = _Cfg(tmp_path)
    ro = Candidate(flag="MEMO_INTERJECT_ENABLED", on_flags={"MEMO_INTERJECT_ENABLED": "1"},
                   k=1, auto_flip=False)
    r = controller.run_graduation_controller(
        cfg, object(), evaluator=_stub(True), candidates=[ro], env={})
    assert r["candidates"][0]["status"] == "report_only"
    assert not overlay_ops.is_flipped_on(tmp_path, ro.flag)


def test_dry_run_writes_nothing(tmp_path: Path):
    cfg = _Cfg(tmp_path)
    for _ in range(3):
        controller.run_graduation_controller(
            cfg, object(), evaluator=_stub(True), candidates=[CAND], env={}, dry_run=True)
    assert not overlay_ops.is_flipped_on(tmp_path, CAND.flag)
    assert not (tmp_path / "graduation" / f"{CAND.flag}.jsonl").exists()
