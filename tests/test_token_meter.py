from pathlib import Path

from memo import token_meter as tm


def _assistant(mid: str, out: int, *, tool: bool = False) -> dict:
    content = [{"type": "text", "text": "x"}]
    if tool:
        content.append({"type": "tool_use", "name": "Read", "input": {}})
    return {"type": "assistant", "message": {"role": "assistant", "id": mid,
            "usage": {"output_tokens": out}, "content": content}}


def _human(text: str = "hola que onda") -> dict:
    return {"type": "user", "message": {"role": "user", "content": text}}


def _tool_result() -> dict:
    return {"type": "user", "message": {"role": "user",
            "content": [{"type": "tool_result", "content": "ok"}]}}


def test_iter_prompt_turns_splits_on_human_prompts_and_dedups_message_id():
    rows = [
        _human(),
        _assistant("m1", 100, tool=True),   # tool step
        _assistant("m1", 100, tool=True),   # SAME id repeated → count once
        _tool_result(),
        _assistant("m2", 40),               # final answer of turn 0
        _human(),
        _assistant("m3", 70),               # single answer of turn 1
    ]
    turns = tm.iter_prompt_turns(rows)
    assert [t.index for t in turns] == [0, 1]
    # turn 0: answer = last assistant (m2=40); tool_tok = m1 (100, counted once)
    assert turns[0].answer_tok == 40
    assert turns[0].tool_tok == 100
    assert turns[0].n_tool_steps == 1
    # turn 1: single assistant → it is the answer, no tool spend
    assert turns[1].answer_tok == 70
    assert turns[1].tool_tok == 0


def test_iter_prompt_turns_skips_sidechain_rows():
    rows = [
        _human(),
        {"type": "assistant", "isSidechain": True,
         "message": {"role": "assistant", "id": "s1", "usage": {"output_tokens": 999},
                     "content": [{"type": "text", "text": "sub"}]}},
        _assistant("m1", 50),
    ]
    turns = tm.iter_prompt_turns(rows)
    assert len(turns) == 1
    assert turns[0].answer_tok == 50
    assert turns[0].tool_tok == 0  # sidechain 999 ignored
