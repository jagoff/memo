"""Tests for the GPU lock deadlock fix in chat_with_timeout / gpu_guard."""
from __future__ import annotations

import threading


class TestGpuTlTimeout:
    """gpu_guard() uses _gpu_tl.timeout to impose a deadline on the lock acquire."""

    def test_gpu_guard_raises_when_other_thread_holds_lock(self):
        from memo.mlx_gpu import _GPU_LOCK, _gpu_tl, gpu_guard

        lock_held = threading.Event()
        release_lock = threading.Event()

        def hold_lock():
            _GPU_LOCK.acquire()
            lock_held.set()
            release_lock.wait(timeout=5.0)
            _GPU_LOCK.release()

        t = threading.Thread(target=hold_lock, daemon=True)
        t.start()
        lock_held.wait(timeout=2.0)

        try:
            _gpu_tl.timeout = 0.1  # 100 ms deadline
            import pytest

            with pytest.raises(TimeoutError), gpu_guard():
                pass
        finally:
            _gpu_tl.timeout = None
            release_lock.set()
            t.join(timeout=2.0)

    def test_gpu_guard_succeeds_when_lock_free(self, monkeypatch):
        monkeypatch.setenv("MEMO_GPU_XPROC_LOCK", "0")
        from memo.mlx_gpu import _gpu_tl, gpu_guard

        _gpu_tl.timeout = 0.5
        try:
            entered = False
            with gpu_guard():
                entered = True
            assert entered
        finally:
            _gpu_tl.timeout = None

    def test_gpu_guard_no_timeout_blocks_normally(self, monkeypatch):
        """With no timeout, gpu_guard should acquire successfully (lock is free)."""
        monkeypatch.setenv("MEMO_GPU_XPROC_LOCK", "0")
        from memo.mlx_gpu import gpu_guard

        entered = False
        with gpu_guard():
            entered = True
        assert entered


class TestChatWithTimeoutSetsThreadLocal:
    """chat_with_timeout sets _gpu_tl.timeout in the worker thread."""

    def test_timeout_propagated_to_worker(self):
        from memo.memory.record import chat_with_timeout
        from memo.mlx_gpu import _gpu_tl

        observed: list[float | None] = []

        class FakeChat:
            _gen_lock = threading.Lock()

            def chat(self, model, messages, options=None):
                observed.append(getattr(_gpu_tl, "timeout", None))
                return {"message": {"content": "ok"}}

        result = chat_with_timeout(FakeChat(), timeout=42.0, model="m", messages=[])
        assert result == {"message": {"content": "ok"}}
        assert observed == [42.0]

    def test_returns_none_on_timeout(self):
        from memo.memory.record import chat_with_timeout

        _started = threading.Event()
        _release = threading.Event()

        class SlowChat:
            _gen_lock = threading.Lock()

            def chat(self, model, messages, options=None):
                _started.set()
                _release.wait(timeout=5.0)
                return {"message": {"content": "done"}}

        result = chat_with_timeout(SlowChat(), timeout=0.1, model="m", messages=[])
        assert result is None
        _release.set()
