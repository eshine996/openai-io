"""请求、连接和 HTTP 状态异常。"""

from __future__ import annotations

import httpx

__all__ = [
    "APIConnectionError",
    "APIError",
    "APIStatusError",
    "APITimeoutError",
    "AuthenticationError",
    "BadRequestError",
    "ConflictError",
    "InternalServerError",
    "NotFoundError",
    "OpenAIError",
    "PermissionDeniedError",
    "RateLimitError",
    "UnprocessableEntityError",
]


class OpenAIError(Exception):
    """所有由本库引发的异常的基类。"""

    def __init__(self, message: str, *, body: object = None) -> None:
        super().__init__(message)
        self.message = message
        self.body = body

    def __str__(self) -> str:
        return self.message


class APIError(OpenAIError):
    """API 返回错误或请求过程中发生错误的基类。"""

    request: httpx.Request
    response: httpx.Response

    def __init__(
        self,
        message: str,
        *,
        request: httpx.Request,
        response: httpx.Response,
        body: object = None,
    ) -> None:
        super().__init__(message, body=body)
        self.request = request
        self.response = response

    def __str__(self) -> str:
        return f"{type(self).__name__}: {self.message}"


class APIConnectionError(APIError):
    """无法连接 API 服务器（网络层错误）。"""

    def __init__(
        self,
        *,
        request: httpx.Request,
        cause: Exception,
    ) -> None:
        message = f"连接出错: {cause}"
        super().__init__(message, request=request, response=httpx.Response(0, request=request), body=None)
        self.cause = cause

    def __str__(self) -> str:
        return f"{type(self).__name__}: {self.message} (cause: {self.cause!r})"


class APITimeoutError(APIConnectionError):
    """请求超时。"""

    def __init__(self, *, request: httpx.Request, cause: Exception | None = None) -> None:
        super().__init__(request=request, cause=cause or TimeoutError("请求超时"))

    def __str__(self) -> str:
        return f"{type(self).__name__}: {self.message}"


class APIStatusError(APIError):
    """API 返回非 2xx 状态码。"""

    status_code: int

    def __init__(
        self,
        message: str,
        *,
        request: httpx.Request,
        response: httpx.Response,
        body: object = None,
    ) -> None:
        super().__init__(message, request=request, response=response, body=body)
        self.status_code = response.status_code

    def __str__(self) -> str:
        err_msg = self.message
        if self.body is not None:
            err_msg += f", 响应体: {self.body!r}"
        return f"{type(self).__name__} (status={self.status_code}): {err_msg}"


class BadRequestError(APIStatusError):
    """400：请求参数错误。"""


class AuthenticationError(APIStatusError):
    """401：API key 无效或缺失。"""


class PermissionDeniedError(APIStatusError):
    """403：没有权限。"""


class NotFoundError(APIStatusError):
    """404：资源不存在。"""


class ConflictError(APIStatusError):
    """409：资源冲突。"""


class UnprocessableEntityError(APIStatusError):
    """422：无法处理的实体。"""


class RateLimitError(APIStatusError):
    """429：触发限流。"""


class InternalServerError(APIStatusError):
    """5xx：服务端内部错误。"""
