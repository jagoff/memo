"""Fail-closed tooling for absorbing Memflow capabilities into Memo."""

from tools.memflow_absorption.safety import (
    SafetyError,
    assert_safe_attempt_root,
)

__all__ = ["SafetyError", "assert_safe_attempt_root"]
