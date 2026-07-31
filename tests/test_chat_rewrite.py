from memo.chat.rewrite import rewrite_query

_HISTORY = [
    {"role": "user", "content": "qué sabés del proyecto memo daemon"},
    {"role": "assistant", "content": "Memo daemon es ..."},
]


def test_info_question_extracts_topic() -> None:
    assert rewrite_query("qué sabés de avature?", None) == "avature"
    assert rewrite_query("tell me about the recall daemon", None) == "the recall daemon"


def test_summary_followup_uses_history_topic() -> None:
    out = rewrite_query("resumime eso", _HISTORY)
    assert "memo" in out and "daemon" in out


def test_pronoun_prefix_injects_topic() -> None:
    out = rewrite_query("y eso cuándo fue?", _HISTORY)
    assert "daemon" in out and "cuándo" in out


def test_plain_question_passthrough() -> None:
    q = "cómo configuro el embedder"
    assert rewrite_query(q, _HISTORY) == q
    assert rewrite_query("resumime eso", None) == "resumime eso"  # sin historial no hay tópico
