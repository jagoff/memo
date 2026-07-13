from dataclasses import dataclass
from pathlib import Path

from memo.graduation import controller, overlay_ops
from memo.graduation.registry import NumericCandidate


@dataclass
class _Cfg:
    state_dir: Path


def _stub(win: bool, best: float):
    def _e(mem, cand, *, k, labels):
        return {"win": win, "best_value": best if win else cand.off_value,
                "delta_prec": 0.02 if win else 0.0, "delta_noise": 0.0}
    return _e


CAND = NumericCandidate(
    flag="MEMO_RECALL_MMR_LAMBDA", field="mmr_lambda",
    off_value=0.0, on_value=0.3, grid=(0.0, 0.3), k=3, auto_flip=True,
)


def test_numeric_flips_to_best_value_after_k_wins(tmp_path: Path):
    cfg = _Cfg(tmp_path)
    for _ in range(2):
        r = controller.run_graduation_controller(
            cfg, object(), evaluator=_stub(True, 0.3), candidates=[CAND], env={})
        assert r["candidates"][0]["status"] == "accumulating"
    assert overlay_ops.overlay_value(tmp_path, CAND.flag) is None
    r = controller.run_graduation_controller(
        cfg, object(), evaluator=_stub(True, 0.3), candidates=[CAND], env={})
    assert r["candidates"][0]["status"] == "graduated"
    # flipped to the PROVEN value, not a bare True.
    assert overlay_ops.overlay_value(tmp_path, CAND.flag) == 0.3


def test_numeric_reverts_when_live_regresses(tmp_path: Path):
    cfg = _Cfg(tmp_path)
    for _ in range(3):
        controller.run_graduation_controller(
            cfg, object(), evaluator=_stub(True, 0.3), candidates=[CAND], env={})
    assert overlay_ops.overlay_value(tmp_path, CAND.flag) == 0.3
    r = controller.run_graduation_controller(
        cfg, object(), evaluator=_stub(False, 0.0), candidates=[CAND], env={})
    assert r["candidates"][0]["status"] == "reverted"
    assert overlay_ops.overlay_value(tmp_path, CAND.flag) is None  # default restored


def test_numeric_env_var_vetoes(tmp_path: Path):
    cfg = _Cfg(tmp_path)
    r = controller.run_graduation_controller(
        cfg, object(), evaluator=_stub(True, 0.3), candidates=[CAND],
        env={"MEMO_RECALL_MMR_LAMBDA": "0.7"})
    assert r["candidates"][0]["status"] == "vetoed"
    assert overlay_ops.overlay_value(tmp_path, CAND.flag) is None


def test_numeric_report_only_never_flips(tmp_path: Path):
    cfg = _Cfg(tmp_path)
    ro = NumericCandidate(flag="MEMO_RECALL_PROJECT_BOOST", field="project_boost",
                          off_value=0.25, on_value=0.35, k=1, auto_flip=False)
    r = controller.run_graduation_controller(
        cfg, object(), evaluator=_stub(True, 0.35), candidates=[ro], env={})
    assert r["candidates"][0]["status"] == "report_only"
    assert overlay_ops.overlay_value(tmp_path, ro.flag) is None
