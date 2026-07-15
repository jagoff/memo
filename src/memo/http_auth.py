"""Shared bearer authentication and bind safety for memo's HTTP transports."""

from __future__ import annotations

import ipaddress
import os
import secrets
import stat
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse

from memo.config import Config
from memo.errors import MemoError

_TOKEN_ENV = "MEMO_HTTP_API_TOKEN"  # noqa: S105 - environment variable name
_TOKEN_FILENAME = "http-api-token"  # noqa: S105 - filename, not a credential
_MIN_TOKEN_CHARS = 32
_MAX_TOKEN_CHARS = 4096
_DEFAULT_RATE_LIMIT = 300
_DEFAULT_RATE_WINDOW_SECONDS = 60
_MAX_RATE_BUCKETS = 4096

_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


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


class SecurityHeadersMiddleware:
    """Apply defensive, API-safe response headers to every HTTP response."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def secure_send(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in _SECURITY_HEADERS.items():
                    headers[name] = value
            await send(message)

        await self.app(scope, receive, secure_send)


class RateLimitMiddleware:
    """Bound HTTP requests per source address with process-local windows."""

    def __init__(
        self,
        app: Any,
        *,
        max_requests: int = _DEFAULT_RATE_LIMIT,
        window_seconds: int = _DEFAULT_RATE_WINDOW_SECONDS,
    ) -> None:
        if max_requests < 1 or window_seconds < 1:
            raise ValueError("HTTP rate limit and window must be positive")
        self.app = app
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = {}
        self._lock = Lock()

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or self._allow(scope):
            await self.app(scope, receive, send)
            return
        await JSONResponse(
            {"detail": "Too many requests"},
            status_code=429,
            headers={"Retry-After": str(self.window_seconds)},
        )(scope, receive, send)

    def _allow(self, scope: dict[str, Any]) -> bool:
        client = scope.get("client")
        key = str(client[0]) if isinstance(client, (list, tuple)) and client else "unknown"
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            if key not in self._requests and len(self._requests) >= _MAX_RATE_BUCKETS:
                key = "overflow"
            bucket = self._requests.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                return False
            bucket.append(now)
            return True


def _split_authority(value: str) -> tuple[str, int | None] | None:
    """Parse an HTTP authority without accepting userinfo or malformed ports."""

    if not value or any(char.isspace() for char in value):
        return None
    try:
        parsed = urlsplit(f"//{value}")
        port = parsed.port
    except ValueError:
        return None
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        return None
    return parsed.hostname, port


def _same_loopback_origin(
    origin: str,
    *,
    host: str,
    host_port: int | None,
    request_scheme: str,
) -> bool:
    """Require browser Origin to exactly match the loopback request authority."""

    if not origin or any(char.isspace() for char in origin):
        return False
    try:
        parsed = urlsplit(origin)
        origin_port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or not is_loopback_host(parsed.hostname)
    ):
        return False

    normalized_host = host.lower().rstrip(".")
    normalized_origin_host = parsed.hostname.lower().rstrip(".")
    if normalized_origin_host != normalized_host or parsed.scheme != request_scheme:
        return False
    default_request_port = 443 if request_scheme == "https" else 80
    default_origin_port = 443 if parsed.scheme == "https" else 80
    return (origin_port or default_origin_port) == (host_port or default_request_port)


class LocalRequestGuardMiddleware:
    """Close browser CSRF and DNS-rebinding paths for unauthenticated MCP HTTP."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        host_values = headers.getlist("host")
        authority = _split_authority(host_values[0]) if len(host_values) == 1 else None
        if authority is None or not is_loopback_host(authority[0]):
            await self._reject(scope, receive, send, status_code=403)
            return

        fetch_sites = headers.getlist("sec-fetch-site")
        if any(value.strip().lower() == "cross-site" for value in fetch_sites):
            await self._reject(scope, receive, send, status_code=403)
            return

        origins = headers.getlist("origin")
        if len(origins) > 1 or (
            origins
            and not _same_loopback_origin(
                origins[0],
                host=authority[0],
                host_port=authority[1],
                request_scheme=str(scope.get("scheme") or "http").lower(),
            )
        ):
            await self._reject(scope, receive, send, status_code=403)
            return

        if scope.get("method", "").upper() == "POST":
            content_types = headers.getlist("content-type")
            content_type = (
                content_types[0].partition(";")[0].strip().lower()
                if len(content_types) == 1
                else ""
            )
            if content_type != "application/json":
                await self._reject(scope, receive, send, status_code=415)
                return

        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(
        scope: dict[str, Any],
        receive: Any,
        send: Any,
        *,
        status_code: int,
    ) -> None:
        detail = (
            "Content-Type must be application/json"
            if status_code == 415
            else "Request origin is not allowed"
        )
        await JSONResponse({"detail": detail}, status_code=status_code)(scope, receive, send)


def build_http_middleware(
    *,
    allow_no_auth: bool = False,
    max_requests: int = _DEFAULT_RATE_LIMIT,
    window_seconds: int = _DEFAULT_RATE_WINDOW_SECONDS,
) -> list[Any]:
    """Return shared Starlette middleware for REST and MCP HTTP apps."""

    from starlette.middleware import Middleware

    middleware = [
        Middleware(SecurityHeadersMiddleware),
        Middleware(
            RateLimitMiddleware,
            max_requests=max_requests,
            window_seconds=window_seconds,
        ),
    ]
    if allow_no_auth:
        middleware.append(Middleware(LocalRequestGuardMiddleware))
    return middleware


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
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise OSError("token path is not a regular file")
        os.fchmod(descriptor, 0o600)
        chunks: list[bytes] = []
        remaining = _MAX_TOKEN_CHARS + 2
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 4096))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        token = b"".join(chunks).decode("utf-8")
    except OSError as exc:
        raise HttpApiAuthError(f"cannot read HTTP token file {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise HttpApiAuthError(f"HTTP token file {path} is not UTF-8") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
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
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
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
    "LocalRequestGuardMiddleware",
    "RateLimitMiddleware",
    "SecurityHeadersMiddleware",
    "build_http_middleware",
    "build_mcp_auth",
    "is_loopback_host",
    "load_http_auth_config",
    "validate_http_bind",
    "verify_http_auth",
]
