"""Unit tests for the client-sampling contextvar state and SamplingChat router."""

from __future__ import annotations

from typing import Any

import pytest

from memo.sampling import (
    SamplingChat,
    SamplingState,
    current_state,
    grounding_chat,
    sampling_scope,
)


class _StubMLX:
    """Stands in for MLXChat: records calls, returns a fixed Ollama-shape dict."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(
        self, model: str, messages: list[dict[str, str]], options: Any = None
    ) -> dict[str, Any]:
        self.calls.append({"model": model, "messages": messages, "options": options})
        return {"message": {"content": "MLX ANSWER"}}

    def chat_stream(self, model: str, messages: list[dict[str, str]], options: Any = None):
        yield "MLX "
        yield "STREAM"


def _state(sampler: Any, calls_left: int = 3) -> SamplingState:
    return SamplingState(sampler=sampler, calls_left=calls_left)


def test_no_state_outside_scope():
    assert current_state() is None


def test_scope_sets_and_resets_state():
    st = _state(lambda m, o: "X")
    with sampling_scope(st):
        assert current_state() is st
    assert current_state() is None


def test_router_delegates_to_mlx_without_state():
    mlx = _StubMLX()
    router = SamplingChat(lambda: mlx)
    out = router.chat("m", [{"role": "user", "content": "q"}])
    assert out == {"message": {"content": "MLX ANSWER"}}
    assert len(mlx.calls) == 1


def test_router_uses_sampler_inside_scope():
    mlx = _StubMLX()
    router = SamplingChat(lambda: mlx)
    st = _state(lambda messages, options: "CLIENT ANSWER")
    with sampling_scope(st):
        out = router.chat("m", [{"role": "user", "content": "q"}])
    assert out == {"message": {"content": "CLIENT ANSWER"}}
    assert st.used_client is True
    assert st.calls_left == 2
    assert mlx.calls == []


def test_sampler_error_is_sticky_and_falls_back():
    mlx = _StubMLX()
    router = SamplingChat(lambda: mlx)
    boom_calls = {"n": 0}

    def _boom(messages: Any, options: Any) -> str:
        boom_calls["n"] += 1
        raise RuntimeError("client refused")

    st = _state(_boom)
    with sampling_scope(st):
        out1 = router.chat("m", [{"role": "user", "content": "q1"}])
        out2 = router.chat("m", [{"role": "user", "content": "q2"}])
    assert out1["message"]["content"] == "MLX ANSWER"
    assert out2["message"]["content"] == "MLX ANSWER"
    assert boom_calls["n"] == 1  # sticky: second call never retries the sampler
    assert st.sticky_off is True
    assert st.used_client is False


def test_call_budget_exhaustion_falls_back():
    mlx = _StubMLX()
    router = SamplingChat(lambda: mlx)
    st = _state(lambda m, o: "CLIENT", calls_left=1)
    with sampling_scope(st):
        out1 = router.chat("m", [{"role": "user", "content": "q1"}])
        out2 = router.chat("m", [{"role": "user", "content": "q2"}])
    assert out1["message"]["content"] == "CLIENT"
    assert out2["message"]["content"] == "MLX ANSWER"


def test_chat_stream_always_uses_mlx():
    mlx = _StubMLX()
    router = SamplingChat(lambda: mlx)
    st = _state(lambda m, o: "CLIENT")
    with sampling_scope(st):
        parts = list(router.chat_stream("m", [{"role": "user", "content": "q"}]))
    assert parts == ["MLX ", "STREAM"]


def test_mlx_factory_is_lazy_and_cached():
    built = {"n": 0}

    def _factory() -> _StubMLX:
        built["n"] += 1
        return _StubMLX()

    router = SamplingChat(_factory)
    st = _state(lambda m, o: "CLIENT")
    with sampling_scope(st):
        router.chat("m", [{"role": "user", "content": "q"}])
    assert built["n"] == 0  # sampled: MLX never constructed
    router.chat("m", [{"role": "user", "content": "q"}])
    router.chat("m", [{"role": "user", "content": "q"}])
    assert built["n"] == 1  # constructed once, cached


def test_grounding_chat_unwraps_router_to_mlx():
    mlx = _StubMLX()
    router = SamplingChat(lambda: mlx)
    assert grounding_chat(router) is mlx
    assert grounding_chat(mlx) is mlx  # passthrough for plain backends


def test_grounding_chat_fails_open_when_mlx_unavailable():
    def _no_mlx() -> Any:
        raise RuntimeError("no MLX on this host")

    router = SamplingChat(_no_mlx)
    backend = grounding_chat(router)
    with pytest.raises(RuntimeError):
        backend.chat("m", [{"role": "user", "content": "q"}])


# --- Task 2: bridge + state_from_ctx -----------------------------------------


def test_make_bridge_joins_messages_and_calls_ctx_sample():
    import asyncio

    from memo.sampling import make_bridge

    captured: dict[str, Any] = {}

    class _FakeResult:
        text = "BRIDGED"

    class _FakeCtx:
        async def sample(self, messages, *, system_prompt=None, temperature=None, max_tokens=None):
            captured["messages"] = messages
            captured["system_prompt"] = system_prompt
            captured["temperature"] = temperature
            captured["max_tokens"] = max_tokens
            return _FakeResult()

    async def _run() -> str:
        loop = asyncio.get_running_loop()
        sampler = make_bridge(_FakeCtx(), loop=loop, timeout_s=5.0, max_tokens=2000)
        msgs = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "prior"},
        ]
        return await asyncio.to_thread(sampler, msgs, {"temperature": 0.2, "num_predict": 512})

    text = asyncio.run(_run())
    assert text == "BRIDGED"
    assert captured["system_prompt"] == "sys prompt"
    assert "hello" in captured["messages"]
    assert "prior" in captured["messages"]
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 512  # min(num_predict, cap)


def test_make_bridge_caps_max_tokens():
    import asyncio

    from memo.sampling import make_bridge

    captured: dict[str, Any] = {}

    class _FakeResult:
        text = "OK"

    class _FakeCtx:
        async def sample(self, messages, *, system_prompt=None, temperature=None, max_tokens=None):
            captured["max_tokens"] = max_tokens
            return _FakeResult()

    async def _run() -> str:
        loop = asyncio.get_running_loop()
        sampler = make_bridge(_FakeCtx(), loop=loop, timeout_s=5.0, max_tokens=100)
        return await asyncio.to_thread(
            sampler, [{"role": "user", "content": "q"}], {"num_predict": 9999}
        )

    asyncio.run(_run())
    assert captured["max_tokens"] == 100


def test_make_bridge_cancels_timed_out_sample_before_fallback_completes(monkeypatch):
    import asyncio
    import threading

    from memo.sampling import make_bridge

    completions: list[str] = []
    started = threading.Event()

    original_schedule = asyncio.run_coroutine_threadsafe

    class _TimedOutFuture:
        def __init__(self, inner):
            self.inner = inner

        def result(self, timeout):
            assert started.wait(timeout=1.0)
            raise TimeoutError

        def done(self):
            return self.inner.done()

        def cancel(self):
            return self.inner.cancel()

    def _schedule(coroutine, loop):
        return _TimedOutFuture(original_schedule(coroutine, loop))

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _schedule)

    class _FakeCtx:
        def __init__(self) -> None:
            self.release = asyncio.Event()
            self.cancelled = threading.Event()

        async def sample(self, *args, **kwargs):
            started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            completions.append("client")
            return type("Result", (), {"text": "TOO LATE"})()

    class _Fallback:
        def chat(self, model, messages, options=None):
            completions.append("fallback")
            return {"message": {"content": "FALLBACK"}}

    async def _run() -> None:
        ctx = _FakeCtx()
        bridge = make_bridge(ctx, loop=asyncio.get_running_loop(), timeout_s=1.0, max_tokens=10)
        router = SamplingChat(lambda: _Fallback())
        state = SamplingState(sampler=bridge, calls_left=1)

        with sampling_scope(state):
            result = await asyncio.to_thread(
                router.chat,
                "model",
                [{"role": "user", "content": "question"}],
            )

        assert result == {"message": {"content": "FALLBACK"}}
        assert await asyncio.to_thread(ctx.cancelled.wait, 1.0)
        ctx.release.set()
        await asyncio.sleep(0)
        assert completions == ["fallback"]

    asyncio.run(_run())


def test_make_bridge_does_not_cancel_successful_future(monkeypatch):
    import asyncio

    from memo.sampling import make_bridge

    class _Result:
        text = "DONE"

    class _CompletedFuture:
        cancel_calls = 0

        def result(self, timeout):
            return _Result()

        def done(self):
            return True

        def cancel(self):
            self.cancel_calls += 1

    future = _CompletedFuture()

    def _schedule(coroutine, loop):
        coroutine.close()
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _schedule)

    class _FakeCtx:
        async def sample(self, *args, **kwargs):  # pragma: no cover - closed by scheduler stub
            return _Result()

    sampler = make_bridge(_FakeCtx(), loop=object(), timeout_s=1.0, max_tokens=10)

    assert sampler([{"role": "user", "content": "question"}], {}) == "DONE"
    assert future.cancel_calls == 0


def test_state_from_ctx_none_when_flag_off(monkeypatch):
    from memo.sampling import state_from_ctx

    monkeypatch.delenv("MEMO_SAMPLING_SYNTH_ENABLED", raising=False)

    class _FakeCtx:
        pass

    assert state_from_ctx(_FakeCtx()) is None


def test_state_from_ctx_none_without_ctx(monkeypatch):
    from memo.sampling import state_from_ctx

    monkeypatch.setenv("MEMO_SAMPLING_SYNTH_ENABLED", "1")
    assert state_from_ctx(None) is None


def test_state_from_ctx_builds_state_when_enabled(monkeypatch):
    import asyncio

    from memo.sampling import state_from_ctx

    monkeypatch.setenv("MEMO_SAMPLING_SYNTH_ENABLED", "1")
    monkeypatch.setenv("MEMO_SAMPLING_MAX_CALLS", "5")

    class _FakeCtx:
        async def sample(self, *a, **k):  # pragma: no cover - not called here
            raise AssertionError

    async def _run():
        return state_from_ctx(_FakeCtx())

    st = asyncio.run(_run())
    assert st is not None
    assert st.calls_left == 5
    assert st.usable()


# --- Task 4: facade routing + chat_with_timeout context propagation ----------


def _mem(tmp_cfg, monkeypatch):
    from memo.config import Config
    from memo.memory import Memory

    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    return Memory(cfg)


def test_ensure_chat_returns_router_when_flag_on(tmp_cfg, monkeypatch):
    monkeypatch.setenv("MEMO_SAMPLING_SYNTH_ENABLED", "1")
    mem = _mem(tmp_cfg, monkeypatch)
    try:
        chat = mem._ensure_chat()
        assert isinstance(chat, SamplingChat)
        assert mem._ensure_chat() is chat  # cached
    finally:
        mem.close()


def test_ensure_chat_mlx_path_when_flag_off(tmp_cfg, monkeypatch):
    monkeypatch.delenv("MEMO_SAMPLING_SYNTH_ENABLED", raising=False)
    monkeypatch.setattr("memo.platform_detect.mlx_available", lambda: True)
    monkeypatch.setattr("memo.llm.MLXChat.__init__", lambda self, *args, **kwargs: None)
    mem = _mem(tmp_cfg, monkeypatch)
    try:
        chat = mem._ensure_chat()
        assert not isinstance(chat, SamplingChat)
    finally:
        mem.close()


def test_chat_with_timeout_propagates_sampling_context():
    from memo.memory.record import chat_with_timeout

    router = SamplingChat(lambda: (_ for _ in ()).throw(RuntimeError("no mlx")))
    st = SamplingState(sampler=lambda m, o: "VIA EXECUTOR", calls_left=3)
    with sampling_scope(st):
        out = chat_with_timeout(
            router, timeout=5.0, model="m", messages=[{"role": "user", "content": "q"}]
        )
    assert out == {"message": {"content": "VIA EXECUTOR"}}
    assert st.used_client is True
