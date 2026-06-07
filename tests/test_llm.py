"""Unit tests for the chat wrapper's non-MLX surface.

Covers the flag-gated prompt-cache toggle, the chat-template fallback for
tokenizers that don't accept ``enable_thinking``, and ``unload`` bookkeeping.
The actual generation path needs MLX and is exercised under
``@pytest.mark.requires_mlx`` elsewhere; ``MLXChat()`` construction is lazy.
"""

from __future__ import annotations

from memo.llm import MLXChat, _apply_chat_template, _prompt_cache_enabled

# -- _prompt_cache_enabled -------------------------------------------------


def test_prompt_cache_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MEMO_PROMPT_CACHE", raising=False)
    assert _prompt_cache_enabled() is False


def test_prompt_cache_truthy_values(monkeypatch):
    for val in ("1", "true", "yes", "on", "ON", "  True "):
        monkeypatch.setenv("MEMO_PROMPT_CACHE", val)
        assert _prompt_cache_enabled() is True, val


def test_prompt_cache_falsy_values(monkeypatch):
    for val in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("MEMO_PROMPT_CACHE", val)
        assert _prompt_cache_enabled() is False, val


# -- _apply_chat_template fallback -----------------------------------------


class _PickyTokenizer:
    """Rejects enable_thinking (mimics non-Qwon3 / older mlx-lm tokenizers)."""

    def __init__(self):
        self.calls: list[dict] = []

    def apply_chat_template(self, **kw):
        self.calls.append(kw)
        if "enable_thinking" in kw:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        return "RENDERED"


def test_apply_chat_template_drops_enable_thinking_on_typeerror():
    tok = _PickyTokenizer()
    out = _apply_chat_template(tok, messages=[{"role": "user"}], enable_thinking=True)
    assert out == "RENDERED"
    # first call raised, retry dropped enable_thinking
    assert len(tok.calls) == 2
    assert "enable_thinking" not in tok.calls[1]


def test_apply_chat_template_passthrough_when_accepted():
    class _OK:
        def apply_chat_template(self, **kw):
            return "OK"

    assert _apply_chat_template(_OK(), messages=[]) == "OK"


# -- unload bookkeeping ----------------------------------------------------


def test_unload_all_returns_false_when_nothing_loaded():
    chat = MLXChat()
    assert chat.unload() is False


def test_unload_named_clears_only_that_model():
    chat = MLXChat()
    # Inject fake loaded state (no MLX): unload manipulates these dicts.
    chat._loaded.update({"a": object(), "b": object()})
    chat._last_use.update({"a": 1.0, "b": 2.0})
    assert chat.unload("a") is True
    assert "a" not in chat._loaded
    assert "b" in chat._loaded
    assert chat.unload("missing") is False


def test_unload_all_clears_everything():
    chat = MLXChat()
    chat._loaded.update({"a": object(), "b": object()})
    assert chat.unload() is True
    assert not chat._loaded
