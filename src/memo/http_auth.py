"""Shared bearer authentication and bind safety for memo's HTTP transports."""

from __future__ import annotations

import ipaddress
import os
import secrets
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memo.config import Config
from memo.errors import MemoError

_TOKEN_ENV = "MEMO_HTTP_API_TOKEN"  # noqa: S105 - environment variable name
_TOKEN_FILENAME = "http-api-token"  # noqa: S105 - filename, not a credential
_MIN_TOKEN_CHARS = 32
_MAX_TOKEN_CHARS = 4096


class HttpApiAuthError(MemoError):
    """HTTP authentication or bind configuration is unsafe."""


class HttpAuthRejected(HttpApiAuthError):
    """A request did not present the configured bearer credential."""

    status_code = 401

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True)
class HttpAuthConfig:
    token: str | None
    allow_no_auth: bool
    host: str


def _token_file(cfg: Config) -> Path:
    return cfg.state_dir / _TOKEN_FILENAME


def _validate_token(token: str, *, source: str) -> str:
    value = token.strip()
    if not (_MIN_TOKEN_CHARS <= len(value) <= _MAX_TOKEN_CHARS):
        raise HttpApiAuthError(
            f"{source} must contain between {_MIN_TOKEN_CHARS} and "
            f"{_MAX_TOKEN_CHARS} non-whitespace characters"
        )
    if any(ch.isspace() for ch in value):
        raise HttpApiAuthError(f"{source} cannot contain whitespace")
    return value


def _read_token_file(path: Path) -> str:
    try:
        token = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HttpApiAuthError(f"cannot read HTTP token file {path}: {exc}") from exc
    try:
        path.chmod(0o600)
    except OSError as exc:
        raise HttpApiAuthError(f"cannot secure HTTP token file {path}: {exc}") from exc
    return _validate_token(token, source=f"HTTP token file {path}")


def _read_or_create_local_token(cfg: Config) -> str:
    path = _token_file(cfg)
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise HttpApiAuthError(f"cannot create HTTP token directory {path.parent}: {exc}") from exc

    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise HttpApiAuthError(f"HTTP token path must be a regular file: {path}")
        return _read_token_file(path)

    token = secrets.token_urlsafe(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        return _read_token_file(path)
    except OSError as exc:
        raise HttpApiAuthError(f"cannot create HTTP token file {path}: {exc}") from exc

    try:
        remaining = memoryview(f"{token}\n".encode())
        while remaining:
            remaining = remaining[os.write(fd, remaining) :]
        os.fsync(fd)
    except OSError as exc:
        with suppress(OSError):
            path.unlink(missing_ok=True)
        raise HttpApiAuthError(f"cannot persist HTTP token file {path}: {exc}") from exc
    finally:
        os.close(fd)
    return token


def load_http_auth_config(*, host: str, allow_no_auth: bool = False) -> HttpAuthConfig:
    """Resolve the shared REST/MCP bearer token, creating a private local one."""

    if allow_no_auth:
        return HttpAuthConfig(token=None, allow_no_auth=True, host=host)

    configured = os.environ.get(_TOKEN_ENV)
    token = (
        _validate_token(configured, source=_TOKEN_ENV)
        if configured is not None and configured.strip()
        else _read_or_create_local_token(Config.from_env())
    )
    return HttpAuthConfig(token=token, allow_no_auth=False, host=host)


def verify_http_auth(authorization: str | None, cfg: HttpAuthConfig) -> None:
    """Validate an HTTP Authorization header without importing FastAPI."""

    if cfg.allow_no_auth:
        return
    if not cfg.token:
        raise HttpAuthRejected("HTTP bearer token is not configured")
    if not authorization:
        raise HttpAuthRejected("Missing bearer token")
    scheme, separator, supplied = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not supplied:
        raise HttpAuthRejected("Missing bearer token")
    if not secrets.compare_digest(supplied, cfg.token):
        raise HttpAuthRejected("Invalid bearer token")


def is_loopback_host(host: str) -> bool:
    """Return whether a bind host is unambiguously local-only."""

    normalized = host.strip().lower().rstrip(".")
    if normalized == "localhost":
        return True
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_http_bind(
    host: str,
    cfg: HttpAuthConfig,
    *,
    allow_non_loopback: bool = False,
) -> None:
    """Reject accidental network exposure and all unauthenticated exposure."""

    if is_loopback_host(host):
        return
    if cfg.allow_no_auth or not cfg.token:
        raise HttpApiAuthError("HTTP non-loopback bind cannot run without authentication")
    if not allow_non_loopback:
        raise HttpApiAuthError(
            "HTTP non-loopback bind requires explicit acknowledgement and a bearer token"
        )


def build_mcp_auth(cfg: HttpAuthConfig) -> Any | None:
    """Build a FastMCP token verifier backed by the shared bearer secret."""

    if cfg.allow_no_auth:
        return None

    from fastmcp.server.auth import AccessToken, TokenVerifier

    class _MemoTokenVerifier(TokenVerifier):
        async def verify_token(self, token: str) -> AccessToken | None:
            if cfg.token is None or not secrets.compare_digest(token, cfg.token):
                return None
            return AccessToken(
                token=token,
                client_id="memo-http-client",
                scopes=[],
            )

    return _MemoTokenVerifier()


__all__ = [
    "HttpApiAuthError",
    "HttpAuthConfig",
    "HttpAuthRejected",
    "build_mcp_auth",
    "is_loopback_host",
    "load_http_auth_config",
    "validate_http_bind",
    "verify_http_auth",
]
