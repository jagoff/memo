from memo.chat.expand import allows_multi_query, classify_query, expand_query


class _FakeChat:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list = []

    def chat(self, model, messages, options=None):
        self.calls.append((model, messages, options))
        return {"message": {"content": self._content}}


def test_classify() -> None:
    assert classify_query("qué hace `rrf_fuse` acá") == "lexical_exact"
    assert classify_query("dónde se define chat_ask_stream") == "lexical_exact"
    assert classify_query("¿quién vino? ¿y cuándo?") == "multi_hop"
    assert classify_query("qué comimos en el cumpleaños") == "semantic_fuzzy"
    # Additional regression tests for identifier classification
    assert classify_query("qué pasó en 2024") == "semantic_fuzzy"
    assert classify_query("cuánto cuesta, tipo 500 pesos") == "semantic_fuzzy"
    assert classify_query("PYTHONPATH está roto") == "lexical_exact"
    assert classify_query("en serio??") == "semantic_fuzzy"


def test_gate() -> None:
    assert allows_multi_query("semantic_fuzzy") is True
    assert allows_multi_query("multi_hop") is True
    assert allows_multi_query("lexical_exact") is False


def test_expand_parses_variants() -> None:
    chat = _FakeChat('bla {"variants": ["variante uno", "variante dos", "tres"]} bla')
    out = expand_query(chat, "m", "pregunta original", n=2)
    assert out == ["variante uno", "variante dos"]
    assert chat.calls[0][2]["temperature"] == 0.0
    assert chat.calls[0][2]["max_tokens"] == 400


def test_expand_malformed_returns_empty() -> None:
    assert expand_query(_FakeChat("no json"), "m", "q") == []

    class _Boom:
        def chat(self, *a, **kw):
            raise RuntimeError("mlx down")

    assert expand_query(_Boom(), "m", "q") == []
