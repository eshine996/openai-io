"""SSE 流式响应处理（同步与异步迭代器）。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
from pydantic import BaseModel

__all__ = ["AsyncStream", "Stream"]

_DONE = "[DONE]"


def _iter_sse_events(response: httpx.Response) -> Iterator[dict[str, Any]]:
    """从流式响应中解析 ``data:`` 事件，跳过空行与注释行。"""
    for line in response.iter_lines():
        stripped = line.strip()
        if not stripped or stripped.startswith(":") or stripped.startswith("event:"):
            continue
        if stripped.startswith("data:"):
            payload = stripped[len("data:") :].strip()
            if payload == _DONE:
                return
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                # 忽略无法解析的行，保持流健壮
                continue


async def _aiter_sse_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """异步版本：从流式响应中解析 ``data:`` 事件。"""
    async for line in response.aiter_lines():
        stripped = line.strip()
        if not stripped or stripped.startswith(":") or stripped.startswith("event:"):
            continue
        if stripped.startswith("data:"):
            payload = stripped[len("data:") :].strip()
            if payload == _DONE:
                return
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue


class Stream[Model: BaseModel]:
    """同步流式迭代器：``for chunk in stream: ...``。

    迭代完整结束后自动关闭底层响应；提前退出时请使用 ``with`` 或手动
    :meth:`close` 避免连接泄漏。
    """

    def __init__(self, *, cast_to: type[Model], response: httpx.Response) -> None:
        self._cast_to = cast_to
        self._response = response

    @property
    def response(self) -> httpx.Response:
        """底层 httpx 响应。"""
        return self._response

    def close(self) -> None:
        self._response.close()

    def __iter__(self) -> Iterator[Model]:
        try:
            for data in _iter_sse_events(self._response):
                yield self._cast_to.model_validate(data)
        finally:
            self._response.close()

    def __enter__(self) -> Stream[Model]:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class AsyncStream[Model: BaseModel]:
    """异步流式迭代器：``async for chunk in stream: ...``。"""

    def __init__(self, *, cast_to: type[Model], response: httpx.Response) -> None:
        self._cast_to = cast_to
        self._response = response

    @property
    def response(self) -> httpx.Response:
        return self._response

    async def close(self) -> None:
        await self._response.aclose()

    def __aiter__(self) -> AsyncIterator[Model]:
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[Model]:
        try:
            async for data in _aiter_sse_events(self._response):
                yield self._cast_to.model_validate(data)
        finally:
            await self._response.aclose()

    async def __aenter__(self) -> AsyncStream[Model]:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()
