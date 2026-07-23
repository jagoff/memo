from __future__ import annotations

from unittest.mock import MagicMock

from memo.server_lifecycle import register


class _Server:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, **_kwargs):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator


def test_lifecycle_tools_delegate_with_bounded_arguments() -> None:
    server = _Server()
    memory = MagicMock()
    memory.list_due_reviews.return_value = [{"id": "due"}]
    memory.mark_reviewed.return_value.to_dict.return_value = {"id": "reviewed"}
    memory.invalidate.return_value.to_dict.return_value = {"id": "invalid"}
    memory.supersede.return_value.to_dict.return_value = {"id": "successor"}
    register(server, memory)

    assert server.tools["memo_review_due"](project="memo", limit=999) == {
        "due": [{"id": "due"}],
        "count": 1,
    }
    memory.list_due_reviews.assert_called_once_with(project="memo", limit=200)

    assert server.tools["memo_mark_reviewed"]("one", evidence="proof", actor="codex") == {
        "id": "reviewed"
    }
    memory.mark_reviewed.assert_called_once_with("one", evidence="proof", actor="codex")

    assert server.tools["memo_invalidate"]("old", reason="stale", at="2026-07-23") == {
        "id": "invalid"
    }
    memory.invalidate.assert_called_once_with("old", reason="stale", at="2026-07-23")

    assert server.tools["memo_supersede"]("old", "new", reason="replaced") == {"id": "successor"}
    memory.supersede.assert_called_once_with("old", "new", reason="replaced")
