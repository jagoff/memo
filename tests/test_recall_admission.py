"""Recall admission control — harness envelopes must not buy a retrieval.

Measured on a live install (1500 consecutive `recall_hook.log` fires): 593 of
them, 40%, were `<task-notification>` blobs. Each paid an MLX embed + vec search
+ an injected block for a machine-to-machine message that no memory can answer.
"""

from __future__ import annotations

import pytest

from memo.recall_admission import admit

MACHINE = [
    "<task-notification>\n<task-id>abc123</task-id>\n</task-notification>",
    "<task-output>done</task-output>",
    "[SYSTEM NOTIFICATION - NOT USER INPUT]\nThis is an automated event.",
    "<system-reminder>\nAs you answer, use the following context\n</system-reminder>",
    "<local-command-stdout>done</local-command-stdout>",
    "<local-command-stderr>boom</local-command-stderr>",
    "<bash-stdout>ok</bash-stdout>",
    "<user-prompt-submit-hook>ctx</user-prompt-submit-hook>",
    "[Request interrupted by user]",
]

HUMAN = [
    "arregla todo no dejes nada pendiente",
    "why does the recall hook fall back to a subprocess",
    "explicame el task-notification que vi en el log",
    "how do I handle a <system-reminder> block in my own hook?",
]


@pytest.mark.parametrize("prompt", MACHINE)
def test_harness_envelopes_are_rejected(prompt):
    text, reason = admit(prompt)
    assert text is None
    assert reason


@pytest.mark.parametrize("prompt", HUMAN)
def test_human_turns_are_admitted_unchanged(prompt):
    text, reason = admit(prompt)
    assert text == prompt
    assert reason == ""


def test_mixed_turn_recalls_on_the_real_question():
    """A wrapper around a real prompt is the case a prefix test gets wrong:
    `session_sources` already strips these pairs to recover the typed text, so
    admission recalls on the question instead of discarding the turn."""
    text, _ = admit(
        "<local-command-stdout>Enabled plan mode</local-command-stdout>\nwhere does the recall daemon socket live"
    )

    assert text == "where does the recall daemon socket live"


def test_envelope_truncated_mid_tag_is_still_rejected():
    """The pair stripper cannot repair a value cut off mid-tag, so the prefix
    check is the backstop rather than the primary rule."""
    text, reason = admit("<local-command-stdout>Enabled plan mode</local-command-")

    assert text is None
    assert reason


def test_leading_whitespace_does_not_smuggle_an_envelope_through():
    text, _ = admit("\n\n  <task-notification>x</task-notification>")

    assert text is None


def test_banner_match_is_case_insensitive():
    text, reason = admit("[system notification - not user input]\nblah")

    assert text is None
    assert reason == "system notification"


def test_empty_prompt_is_left_to_the_min_chars_gate():
    """Emptiness is not an envelope; reporting it as one would put the wrong
    reason in the bail log."""
    text, reason = admit("   ")

    assert text == "   "
    assert reason == ""


def test_admission_does_not_redefine_the_envelope_vocabulary():
    """Regression: this module used to carry its own prefix list, narrower than
    the repo's canonical one — it missed <task-output>, <bash-std*> and the
    <command-*> wrappers entirely."""
    import memo.recall_admission as ra
    from memo.session_sources import _COMMAND_WRAPPER_PREFIXES

    assert ra._COMMAND_WRAPPER_PREFIXES is _COMMAND_WRAPPER_PREFIXES
