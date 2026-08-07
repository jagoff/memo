"""Admission control for the recall hook: which prompts are worth a retrieval.

`UserPromptSubmit` fires on everything the harness pushes through the prompt
channel, not only on what a human typed. Measured over 1500 consecutive hook
fires on a live install, 593 (40%) were `<task-notification>` envelopes — a
machine telling the agent a background job finished. Each one paid a full MLX
embed, a vec search, and an injected recall block, and none of them can have a
durable answer in memory.

The envelope vocabulary is NOT redefined here. `session_sources` already owns
the canonical list and the tag-pair stripper the transcript reader uses, so this
module composes them: strip the wrappers, and judge what is left. That keeps a
mixed turn — plumbing wrapped around a real question — recallable on the real
question instead of thrown away, which a prefix test alone would get wrong.
"""

from __future__ import annotations

from memo.session_sources import _COMMAND_WRAPPER_PREFIXES, _strip_command_wrappers

# Machine markers that are NOT tag pairs, so the wrapper stripper cannot see
# them. Bracketed banners the harness prepends to a turn.
_MACHINE_BANNERS: tuple[tuple[str, str], ...] = (
    ("[SYSTEM NOTIFICATION", "system notification"),
    ("[Request interrupted", "interrupted request"),
)


def admit(prompt: str) -> tuple[str | None, str]:
    """Decide whether `prompt` earns a retrieval.

    Returns ``(text, reason)``. ``text`` is what recall should run on — the
    prompt with harness wrappers stripped, which may be shorter than the input.
    ``text`` is None when nothing human is left, and ``reason`` then names the
    kind of envelope it was (for the bail log).
    """
    head = prompt.lstrip()
    if not head:
        return prompt, ""
    for banner, reason in _MACHINE_BANNERS:
        if head[: len(banner)].lower() == banner.lower():
            return None, reason
    cleaned = _strip_command_wrappers(head)
    if not cleaned:
        return None, "harness envelope"
    if cleaned == head and head.startswith(_COMMAND_WRAPPER_PREFIXES):
        # Opens with a wrapper yet nothing was stripped: the tag pair is
        # unclosed (a value cut off mid-tag), which the stripper cannot repair.
        return None, "truncated envelope"
    return cleaned, ""
