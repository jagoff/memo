from memo.chat.synthesis import REFUSAL, build_messages, filter_by_relevance, synthesize_stream


def _src(sid: str, norm: float, **kw) -> dict:
    return {
        "id": sid,
        "title": f"T{sid}",
        "snippet": f"cuerpo {sid}",
        "normalized_score": norm,
        **kw,
    }


def test_floor_is_relative_to_top() -> None:
    kept = filter_by_relevance([_src("a", 1.0), _src("b", 0.3), _src("c", 0.1)], floor=0.25)
    assert [s["id"] for s in kept] == ["a", "b"]  # 0.1 < 1.0*0.25


def test_floor_never_empties_and_keep_exempt() -> None:
    assert len(filter_by_relevance([_src("a", 1.0)], floor=0.25)) == 1  # <2 no-op
    kept = filter_by_relevance([_src("a", 1.0), _src("b", 0.05, keep=True)], floor=0.25)
    assert {s["id"] for s in kept} == {"a", "b"}


def test_build_messages_contract() -> None:
    messages = build_messages("¿quién es Ana?", [_src("a", 1.0)], today="30/07/2026")
    assert messages[0]["role"] == "system"
    assert "EXCLUSIVAMENTE" in messages[0]["content"]
    assert "30/07/2026" in messages[0]["content"]
    assert REFUSAL in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "PREGUNTA: ¿quién es Ana?" in messages[1]["content"]
    assert "cuerpo a" in messages[1]["content"]


def test_synthesize_stream_passes_options() -> None:
    class _Fake:
        def chat_stream(self, model, messages, options=None):
            assert options["temperature"] == 0.1
            assert options["max_tokens"] == 1200
            yield "hola "
            yield "mundo"

    tokens = list(synthesize_stream(_Fake(), "m", "q", [_src("a", 1.0)], max_tokens=1200))
    assert "".join(tokens) == "hola mundo"


def test_floor_noop_when_all_scores_zero() -> None:
    kept = filter_by_relevance([_src("a", 0.0), _src("b", 0.0)], floor=0.25)
    assert {s["id"] for s in kept} == {"a", "b"}


def test_floor_noop_when_negative() -> None:
    kept = filter_by_relevance([_src("a", 1.0), _src("b", 0.5)], floor=-0.5)
    assert {s["id"] for s in kept} == {"a", "b"}


def test_floor_never_empties_even_when_top_fails() -> None:
    kept = filter_by_relevance([_src("a", 1.0), _src("b", 0.5)], floor=2.0)
    assert [s["id"] for s in kept] == ["a"]
