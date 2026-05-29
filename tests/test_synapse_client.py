"""Tests para synapse_client module."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

from memo import synapse_client


class TestExecutable:
    """Tests para _executable()."""

    def test_executable_returns_none_if_missing(self) -> None:
        """_executable() retorna None si synapse no está en PATH."""
        with patch("shutil.which", return_value=None):
            assert synapse_client._executable() is None

    def test_executable_returns_path_if_found(self) -> None:
        """_executable() retorna path si synapse está en PATH."""
        with patch("shutil.which", return_value="/usr/local/bin/synapse"):
            assert synapse_client._executable() == "/usr/local/bin/synapse"

    def test_executable_respects_env_override(self) -> None:
        """_executable() respeta MEMO_SYNAPSE_EXECUTABLE env var."""
        with (
            patch.dict("os.environ", {"MEMO_SYNAPSE_EXECUTABLE": "/custom/synapse"}),
            patch("os.path.exists", return_value=True),
        ):
            assert synapse_client._executable() == "/custom/synapse"

    def test_executable_returns_none_if_env_override_missing(self) -> None:
        """_executable() retorna None si env override no existe."""
        with (
            patch.dict("os.environ", {"MEMO_SYNAPSE_EXECUTABLE": "/missing/synapse"}),
            patch("os.path.exists", return_value=False),
        ):
            assert synapse_client._executable() is None


class TestTimeout:
    """Tests para _timeout()."""

    def test_timeout_default(self) -> None:
        """_timeout() retorna default si no está set."""
        with patch.dict("os.environ", {}, clear=True):
            assert synapse_client._timeout() == 8.0

    def test_timeout_from_env(self) -> None:
        """_timeout() lee MEMO_SYNAPSE_CLIENT_TIMEOUT."""
        with patch.dict("os.environ", {"MEMO_SYNAPSE_CLIENT_TIMEOUT": "15.5"}):
            assert synapse_client._timeout() == 15.5

    def test_timeout_invalid_env_uses_default(self) -> None:
        """_timeout() usa default si env value es inválido."""
        with patch.dict("os.environ", {"MEMO_SYNAPSE_CLIENT_TIMEOUT": "invalid"}):
            assert synapse_client._timeout() == 8.0

    def test_timeout_negative_uses_default(self) -> None:
        """_timeout() usa default si env value es negativo."""
        with patch.dict("os.environ", {"MEMO_SYNAPSE_CLIENT_TIMEOUT": "-5"}):
            assert synapse_client._timeout() == 8.0


class TestIsAvailable:
    """Tests para is_available()."""

    def test_is_available_true_when_executable_found(self) -> None:
        """is_available() retorna True si synapse está en PATH."""
        with patch.object(synapse_client, "_executable", return_value="/usr/bin/synapse"):
            assert synapse_client.is_available() is True

    def test_is_available_false_when_executable_missing(self) -> None:
        """is_available() retorna False si synapse no está en PATH."""
        with patch.object(synapse_client, "_executable", return_value=None):
            assert synapse_client.is_available() is False


class TestListConflicts:
    """Tests para list_conflicts()."""

    def test_list_conflicts_returns_empty_if_binary_missing(self) -> None:
        """list_conflicts() retorna [] si synapse binary no existe."""
        with patch.object(synapse_client, "_executable", return_value=None):
            result = synapse_client.list_conflicts("test query")
            assert result == []

    def test_list_conflicts_returns_empty_on_timeout(self) -> None:
        """list_conflicts() retorna [] en timeout."""
        with (
            patch.object(synapse_client, "_executable", return_value="/usr/bin/synapse"),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired("cmd", 5.0),
            ),
        ):
            result = synapse_client.list_conflicts("test query", timeout=5.0)
            assert result == []

    def test_list_conflicts_returns_empty_on_nonzero_exit(self) -> None:
        """list_conflicts() retorna [] si synapse exit != 0."""
        with patch.object(synapse_client, "_executable", return_value="/usr/bin/synapse"):
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.stderr = "error message"
            with patch("subprocess.run", return_value=mock_proc):
                result = synapse_client.list_conflicts("test query")
                assert result == []

    def test_list_conflicts_returns_empty_on_bad_json(self) -> None:
        """list_conflicts() retorna [] si output no es JSON válido."""
        with patch.object(synapse_client, "_executable", return_value="/usr/bin/synapse"):
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "invalid json"
            with patch("subprocess.run", return_value=mock_proc):
                result = synapse_client.list_conflicts("test query")
                assert result == []

    def test_list_conflicts_parses_valid_response(self) -> None:
        """list_conflicts() parsea response válido."""
        payload = {
            "schema": "synapse.conflicts.v1",
            "query": "test",
            "conflicts": [
                {
                    "conflict_id": "c1",
                    "freeze_write": True,
                    "lifecycle_state": "detected",
                    "summary": "Test conflict",
                }
            ],
        }
        with patch.object(synapse_client, "_executable", return_value="/usr/bin/synapse"):
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = json.dumps(payload)
            with patch("subprocess.run", return_value=mock_proc):
                result = synapse_client.list_conflicts("test query")
                assert len(result) == 1
                assert result[0]["conflict_id"] == "c1"

    def test_list_conflicts_filters_non_dict_conflicts(self) -> None:
        """list_conflicts() filtra conflicts que no son dicts."""
        payload = {
            "schema": "synapse.conflicts.v1",
            "query": "test",
            "conflicts": [
                {"conflict_id": "c1"},
                "invalid",
                None,
                {"conflict_id": "c2"},
            ],
        }
        with patch.object(synapse_client, "_executable", return_value="/usr/bin/synapse"):
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = json.dumps(payload)
            with patch("subprocess.run", return_value=mock_proc):
                result = synapse_client.list_conflicts("test query")
                assert len(result) == 2
                assert result[0]["conflict_id"] == "c1"
                assert result[1]["conflict_id"] == "c2"


class TestHasBlockingFreeze:
    """Tests para has_blocking_freeze()."""

    def test_has_blocking_freeze_false_when_empty(self) -> None:
        """has_blocking_freeze() retorna (False, None) si conflicts vacío."""
        blocked, conflict = synapse_client.has_blocking_freeze([])
        assert blocked is False
        assert conflict is None

    def test_has_blocking_freeze_false_when_no_freeze_write(self) -> None:
        """has_blocking_freeze() retorna False si ninguno tiene freeze_write."""
        conflicts = [
            {
                "conflict_id": "c1",
                "freeze_write": False,
                "lifecycle_state": "detected",
            },
            {
                "conflict_id": "c2",
                "freeze_write": False,
                "lifecycle_state": "acknowledged",
            },
        ]
        blocked, conflict = synapse_client.has_blocking_freeze(conflicts)
        assert blocked is False
        assert conflict is None

    def test_has_blocking_freeze_false_when_resolved(self) -> None:
        """has_blocking_freeze() retorna False si freeze_write pero resolved."""
        conflicts = [
            {
                "conflict_id": "c1",
                "freeze_write": True,
                "lifecycle_state": "resolved",
            }
        ]
        blocked, conflict = synapse_client.has_blocking_freeze(conflicts)
        assert blocked is False
        assert conflict is None

    def test_has_blocking_freeze_true_when_detected(self) -> None:
        """has_blocking_freeze() retorna True si freeze_write + detected."""
        conflicts = [
            {
                "conflict_id": "c1",
                "freeze_write": True,
                "lifecycle_state": "detected",
                "summary": "Test conflict",
            }
        ]
        blocked, conflict = synapse_client.has_blocking_freeze(conflicts)
        assert blocked is True
        assert conflict["conflict_id"] == "c1"

    def test_has_blocking_freeze_true_when_acknowledged(self) -> None:
        """has_blocking_freeze() retorna True si freeze_write + acknowledged."""
        conflicts = [
            {
                "conflict_id": "c1",
                "freeze_write": True,
                "lifecycle_state": "acknowledged",
            }
        ]
        blocked, conflict = synapse_client.has_blocking_freeze(conflicts)
        assert blocked is True
        assert conflict["conflict_id"] == "c1"

    def test_has_blocking_freeze_returns_first_blocking(self) -> None:
        """has_blocking_freeze() retorna primer conflict bloqueante."""
        conflicts = [
            {
                "conflict_id": "c1",
                "freeze_write": False,
                "lifecycle_state": "detected",
            },
            {
                "conflict_id": "c2",
                "freeze_write": True,
                "lifecycle_state": "detected",
            },
            {
                "conflict_id": "c3",
                "freeze_write": True,
                "lifecycle_state": "detected",
            },
        ]
        blocked, conflict = synapse_client.has_blocking_freeze(conflicts)
        assert blocked is True
        assert conflict["conflict_id"] == "c2"
