from __future__ import annotations

import json

import pytest

from memo.config import Config
from memo.contradict import emit_anomaly
from memo.errors import OperationalError
from memo.memory import Memory
from memo.operational_activation import (
    activate_fresh_operational_v2,
    open_activated_operational_v2,
    select_operational_store,
)
from memo.operational_key_store import KeyStoreError
from tests.operational_authority import build_test_fresh_v2_authority


def _config(tmp_path) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        device_id="device-a",
        reranker_enabled=False,
    )


def test_fresh_v2_activation_roundtrips_public_operational_state(tmp_path) -> None:
    cfg = _config(tmp_path)
    authority = build_test_fresh_v2_authority(
        cfg.operational_root,
        device_id=cfg.device_id,
    ).runtime_authority()

    store = activate_fresh_operational_v2(cfg, authority=authority)
    focus = store.set_focus(project="memo", summary="Memo-only coordination")

    assert store.backend_version == 2
    assert store.state()["focus"]["memo"]["id"] == focus.id
    assert store.ledger.verify()["ok"] is True
    with pytest.raises(PermissionError, match="authorization"):
        store.ledger.append(object())

    reopened = open_activated_operational_v2(cfg, authority=authority)
    assert reopened.backend_version == 2
    assert reopened.state()["focus"]["memo"]["summary"] == "Memo-only coordination"


def test_activated_v2_persists_signal_and_receipt_facade_writes(tmp_path) -> None:
    cfg = _config(tmp_path)
    authority = build_test_fresh_v2_authority(
        cfg.operational_root,
        device_id=cfg.device_id,
    ).runtime_authority()
    store = activate_fresh_operational_v2(cfg, authority=authority)

    signal = store.remember_signal(
        marker="watcher:memo",
        epoch=4,
        fence="leader-b",
        payload={"commits": 3},
    )
    replay = store.remember_signal(
        marker="watcher:memo",
        epoch=4,
        fence="leader-b",
        payload={"commits": 99},
    )
    receipt = store.receipt(
        "save",
        subject_uri="memo://memory/mem-1",
        trace_id="trace-save-1",
        actor_id="requested-actor",
        metadata={"memory_id": "mem-1"},
    )

    assert replay == signal
    assert store.list_signals() == [signal]
    assert store.state()["signals"][signal.marker]["payload"] == {"commits": 3}
    assert store.state()["receipts"][receipt.receipt_id] == receipt.to_dict()
    assert receipt.actor_id == "memo-runtime"
    assert store.ledger.verify()["ok"] is True


def test_v2_anomaly_uses_canonical_signed_conflict_lifecycle(tmp_path) -> None:
    cfg = _config(tmp_path)
    authority = build_test_fresh_v2_authority(
        cfg.operational_root,
        device_id=cfg.device_id,
    ).runtime_authority()
    store = activate_fresh_operational_v2(cfg, authority=authority)

    anomaly_id = emit_anomaly(
        "a" * 32,
        "b" * 32,
        "contradiction",
        0.94,
        "open",
        operational=store,
    )

    assert anomaly_id is not None
    conflict = store.state()["conflicts"][anomaly_id]
    assert conflict["lifecycle_state"] == "detected"
    assert conflict["metadata"]["memory_ids"] == ["a" * 32, "b" * 32]
    assert store.ledger.verify()["ok"] is True


def test_activation_stamp_tamper_fails_closed_before_open(tmp_path) -> None:
    cfg = _config(tmp_path)
    authority = build_test_fresh_v2_authority(
        cfg.operational_root,
        device_id=cfg.device_id,
    ).runtime_authority()
    activate_fresh_operational_v2(cfg, authority=authority)
    path = cfg.operational_root / "operational-v2-activated.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    body["anchor_hash"] = "0" * 64
    path.write_text(
        json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(OperationalError, match="activation"):
        open_activated_operational_v2(cfg, authority=authority)


def test_activation_anchor_replacement_fails_closed(tmp_path) -> None:
    cfg = _config(tmp_path)
    authority = build_test_fresh_v2_authority(
        cfg.operational_root,
        device_id=cfg.device_id,
    ).runtime_authority()
    activate_fresh_operational_v2(cfg, authority=authority)
    anchor = cfg.operational_root / "journal" / "anchors" / f"{cfg.device_id}.json"
    body = json.loads(anchor.read_text(encoding="utf-8"))
    body["anchor_id"] = "replaced"
    anchor.write_text(
        json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(OperationalError):
        open_activated_operational_v2(cfg, authority=authority)


def test_selector_activates_fresh_injected_v2_and_reopens_it(tmp_path) -> None:
    cfg = _config(tmp_path)
    test_authority = build_test_fresh_v2_authority(
        cfg.operational_root,
        device_id=cfg.device_id,
    )
    configured = cfg.model_copy(
        update={
            "operational_signer": test_authority.signer,
            "operational_epoch_fence": test_authority.fence,
        }
    )

    first = select_operational_store(configured)
    first.set_focus(project="memo", summary="selected v2")
    second = select_operational_store(configured)

    assert first.backend_version == 2
    assert second.backend_version == 2
    assert second.state()["focus"]["memo"]["summary"] == "selected v2"


def test_selector_keeps_authorized_existing_v1_migration_backend(tmp_path) -> None:
    cfg = _config(tmp_path)
    legacy = cfg.state_dir / "journal"
    legacy.mkdir(parents=True)

    store = select_operational_store(
        cfg.model_copy(update={"operational_context_provider": lambda: None})
    )

    assert store.backend_version == 1


def test_selector_can_disable_only_implicit_fresh_v2_activation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    monkeypatch.setenv("MEMO_OPERATIONAL_V2_AUTO_ACTIVATE", "0")

    store = select_operational_store(cfg)

    assert store.backend_version == 1
    assert not cfg.operational_root.exists()


def test_selector_defaults_fresh_macos_to_functional_v1_without_packaged_helper(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    monkeypatch.delenv("MEMO_OPERATIONAL_V2_AUTO_ACTIVATE", raising=False)
    monkeypatch.setattr("memo.operational_activation.sys.platform", "darwin")

    def unexpected_activation(_cfg):
        raise AssertionError("default selection must not enroll preview v2 authority")

    monkeypatch.setattr(
        "memo.operational_activation.build_fresh_productive_authority",
        unexpected_activation,
    )

    store = select_operational_store(cfg)
    signal = store.remember_signal(marker="fresh-mac", epoch=1, fence="watcher-a")

    assert store.backend_version == 1
    assert store.list_signals() == [signal]
    assert not cfg.operational_root.exists()


def test_explicit_macos_v2_opt_in_fails_closed_without_helper_and_no_stamp(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    monkeypatch.setenv("MEMO_OPERATIONAL_V2_AUTO_ACTIVATE", "1")
    monkeypatch.setattr("memo.operational_activation.sys.platform", "darwin")

    def missing_helper(_cfg):
        raise KeyStoreError("packaged Secure Enclave helper is unavailable")

    monkeypatch.setattr(
        "memo.operational_activation.build_fresh_productive_authority",
        missing_helper,
    )

    with pytest.raises(KeyStoreError, match="packaged Secure Enclave helper"):
        select_operational_store(cfg)

    assert not (cfg.operational_root / "operational-v2-activated.json").exists()


def test_selector_reopens_activated_v2_when_auto_activation_is_off(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    test_authority = build_test_fresh_v2_authority(
        cfg.operational_root,
        device_id=cfg.device_id,
    )
    authority = test_authority.runtime_authority()
    activate_fresh_operational_v2(cfg, authority=authority)
    monkeypatch.setenv("MEMO_OPERATIONAL_V2_AUTO_ACTIVATE", "0")
    configured = cfg.model_copy(
        update={
            "operational_signer": test_authority.signer,
            "operational_epoch_fence": test_authority.fence,
        }
    )

    reopened = select_operational_store(configured)

    assert reopened.backend_version == 2


def test_selector_keeps_linux_fresh_install_on_v1_without_keychain(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    monkeypatch.setattr("memo.operational_activation.sys.platform", "linux")

    store = select_operational_store(cfg)

    assert store.backend_version == 1
    assert not cfg.operational_root.exists()


def test_memory_facade_selects_verified_v2_backend(tmp_path) -> None:
    cfg = _config(tmp_path)
    test_authority = build_test_fresh_v2_authority(
        cfg.operational_root,
        device_id=cfg.device_id,
    )
    configured = cfg.model_copy(
        update={
            "operational_signer": test_authority.signer,
            "operational_epoch_fence": test_authority.fence,
        }
    )

    memory = Memory(configured)
    try:
        memory.operational.set_focus(project="memo", summary="facade uses v2")
        assert memory.operational.backend_version == 2
        assert memory.operational.state()["focus"]["memo"]["summary"] == "facade uses v2"
    finally:
        memory.close()
