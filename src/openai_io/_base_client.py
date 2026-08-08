"""基于 httpx 的同步与异步传输层。"""

from __future__ import annotations

import asyncio
import json
import random
import time
from types import TracebackType
from typing import Any, Final, cast

import httpx

from ._exceptions import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from ._types import DEFAULT_BASE_URL, Headers
from ._utils import merge_headers

__all__ = ["AsyncAPIClient", "BaseClient", "SyncAPIClient"]

#: 可重试的 HTTP 状态码。
RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({408, 409, 429, 500, 502, 503, 504})

#: 默认超时（秒）。
DEFAULT_TIMEOUT: Final[float] = 600.0

#: 默认最大重试次数（不含首次请求）。
DEFAULT_MAX_RETRIES: Final[int] = 2


def _build_status_error(response: httpx.Response) -> APIStatusError:
    """根据状态码构造对应的异常。"""
    body = _try_parse_json(response)
    message = _extract_error_message(body)
    request = response.request
    kwargs: dict[str, object] = {"message": message, "request": request, "response": response, "body": body}
    status = response.status_code
    if status == 400:
        return BadRequestError(**kwargs)  # type: ignore[arg-type]
    if status == 401:
        return AuthenticationError(**kwargs)  # type: ignore[arg-type]
    if status == 403:
        return PermissionDeniedError(**kwargs)  # type: ignore[arg-type]
    if status == 404:
        return NotFoundError(**kwargs)  # type: ignore[arg-type]
    if status == 409:
        return ConflictError(**kwargs)  # type: ignore[arg-type]
    if status == 422:
        return UnprocessableEntityError(**kwargs)  # type: ignore[arg-type]
    if status == 429:
        return RateLimitError(**kwargs)  # type: ignore[arg-type]
    if status >= 500:
        return InternalServerError(**kwargs)  # type: ignore[arg-type]
    return APIStatusError(**kwargs)  # type: ignore[arg-type]


def _try_parse_json(response: httpx.Response) -> object:
    try:
        return response.json()
    except json.JSONDecodeError:
        return None


def _extract_error_message(body: object) -> str:
    """从 openai 风格的错误响应体中提取 message。"""
    if isinstance(body, dict):
        data = cast("dict[str, object]", body)
        error = data.get("error")
        if isinstance(error, dict):
            error_data = cast("dict[str, object]", error)
            message = error_data.get("message")
            if isinstance(message, str) and message:
                return message
            return f"未知错误: {error!r}"
        if isinstance(error, str) and error:
            return error
        message = data.get("message")
        if isinstance(message, str) and message:
            return message
    return "请求失败"


class BaseClient:
    """HTTP 客户端共享逻辑（配置、认证头、URL、错误构造、重试决策）。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float | httpx.Timeout = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Headers | None = None,
    ) -> None:
        self.api_key = api_key
        self.organization = organization
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_headers = dict(default_headers or {})

    @property
    def auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.organization is not None:
            headers["OpenAI-Organization"] = self.organization
        return headers

    def _build_headers(self, headers: Headers | None = None) -> dict[str, str]:
        return merge_headers(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                **self.default_headers,
                **self.auth_headers,
            },
            headers,
        )

    def _build_stream_headers(self, headers: Headers | None = None) -> dict[str, str]:
        return merge_headers(self._build_headers(headers), {"Accept": "text/event-stream"})

    def _full_url(self, url: str) -> str:
        """把相对路径拼上 base_url；绝对 URL 原样返回。"""
        if url.startswith(("http://", "https://")):
            return url
        return f"{self.base_url}{url}"

    @staticmethod
    def _should_retry(response: httpx.Response | None) -> bool:
        return response is not None and response.status_code in RETRYABLE_STATUS_CODES

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        """指数退避 + 随机抖动，上限 8 秒。"""
        base = 0.25 * (2**attempt)
        return min(base + random.uniform(0, base / 2), 8.0)


class SyncAPIClient(BaseClient):
    """基于 ``httpx.Client`` 的同步传输层。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float | httpx.Timeout = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Headers | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        self._http_client = http_client or httpx.Client(timeout=timeout)

    def request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: Headers | None = None,
    ) -> httpx.Response:
        """发送普通请求并返回响应（非 2xx 时抛对应异常）。"""
        return self._send(method, url, json_body=json_body, params=params, headers=headers, stream=False)

    def stream(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: Headers | None = None,
    ) -> httpx.Response:
        """发送流式请求并返回响应（配合 SSE 逐行读取）。"""
        return self._send(method, url, json_body=json_body, params=params, headers=headers, stream=True)

    def _send(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None,
        params: dict[str, Any] | None,
        headers: Headers | None,
        stream: bool,
    ) -> httpx.Response:
        request_headers = self._build_stream_headers(headers) if stream else self._build_headers(headers)
        request = self._http_client.build_request(
            method,
            self._full_url(url),
            json=json_body,
            params=params,
            headers=request_headers,
        )
        for attempt in range(self.max_retries + 1):
            try:
                response = self._http_client.send(request, stream=stream)
            except httpx.TimeoutException as exc:
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay(attempt))
                    continue
                raise APITimeoutError(request=request, cause=exc) from exc
            except httpx.TransportError as exc:
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay(attempt))
                    continue
                raise APIConnectionError(request=request, cause=exc) from exc

            if response.status_code >= 300:
                if self._should_retry(response) and attempt < self.max_retries:
                    response.close()
                    time.sleep(self._retry_delay(attempt))
                    continue
                if stream:
                    try:
                        response.read()
                    except httpx.TimeoutException as exc:
                        response.close()
                        raise APITimeoutError(request=request, cause=exc) from exc
                    except httpx.TransportError as exc:
                        response.close()
                        raise APIConnectionError(request=request, cause=exc) from exc
                raise _build_status_error(response)
            return response
        raise APIConnectionError(request=request, cause=RuntimeError("重试次数耗尽"))

    def close(self) -> None:
        self._http_client.close()

    def __enter__(self) -> SyncAPIClient:
        self._http_client.__enter__()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self._http_client.__exit__(exc_type, exc, tb)


class AsyncAPIClient(BaseClient):
    """基于 ``httpx.AsyncClient`` 的异步传输层。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float | httpx.Timeout = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Headers | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        self._http_client = http_client or httpx.AsyncClient(timeout=timeout)

    async def request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: Headers | None = None,
    ) -> httpx.Response:
        """发送普通请求并返回响应（非 2xx 时抛对应异常）。"""
        return await self._send(method, url, json_body=json_body, params=params, headers=headers, stream=False)

    async def stream(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: Headers | None = None,
    ) -> httpx.Response:
        """发送流式请求并返回响应（配合 SSE 逐行读取）。"""
        return await self._send(method, url, json_body=json_body, params=params, headers=headers, stream=True)

    async def _send(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None,
        params: dict[str, Any] | None,
        headers: Headers | None,
        stream: bool,
    ) -> httpx.Response:
        request_headers = self._build_stream_headers(headers) if stream else self._build_headers(headers)
        request = self._http_client.build_request(
            method,
            self._full_url(url),
            json=json_body,
            params=params,
            headers=request_headers,
        )
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._http_client.send(request, stream=stream)
            except httpx.TimeoutException as exc:
                if attempt < self.max_retries:
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                raise APITimeoutError(request=request, cause=exc) from exc
            except httpx.TransportError as exc:
                if attempt < self.max_retries:
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                raise APIConnectionError(request=request, cause=exc) from exc

            if response.status_code >= 300:
                if self._should_retry(response) and attempt < self.max_retries:
                    await response.aclose()
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                if stream:
                    try:
                        await response.aread()
                    except httpx.TimeoutException as exc:
                        await response.aclose()
                        raise APITimeoutError(request=request, cause=exc) from exc
                    except httpx.TransportError as exc:
                        await response.aclose()
                        raise APIConnectionError(request=request, cause=exc) from exc
                raise _build_status_error(response)
            return response
        raise APIConnectionError(request=request, cause=RuntimeError("重试次数耗尽"))

    async def close(self) -> None:
        await self._http_client.aclose()

    async def __aenter__(self) -> AsyncAPIClient:
        await self._http_client.__aenter__()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self._http_client.__aexit__(exc_type, exc, tb)
