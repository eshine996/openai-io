"""异步客户端 :class:`AsyncOpenAI`。

示例::

    import asyncio

    from openai_io import AsyncOpenAI
    from openai_io.messages import HumanMessage

    async def main() -> None:
        client = AsyncOpenAI(api_key="sk-...")
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[HumanMessage(content="你好")],
        )
        print(resp.choices[0].message.content)

    asyncio.run(main())
"""

from __future__ import annotations

from types import TracebackType

import httpx

from ._base_client import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, AsyncAPIClient
from ._types import API_KEY_ENV_VAR, DEFAULT_BASE_URL, Headers, get_env
from .resources import AsyncChat, AsyncCompletions, AsyncEmbeddings

__all__ = ["AsyncOpenAI"]


class AsyncOpenAI:
    """异步 OpenAI 客户端。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | httpx.URL = DEFAULT_BASE_URL,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Headers | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """构造异步客户端，参数与 :class:`OpenAI` 一致。"""
        api_key = api_key if api_key is not None else get_env(API_KEY_ENV_VAR)
        self._transport = AsyncAPIClient(
            api_key=api_key,
            organization=organization,
            base_url=str(base_url).rstrip("/"),
            timeout=httpx.Timeout(timeout) if timeout is not None else httpx.Timeout(None),
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
        )
        self.chat = AsyncChat(self._transport)
        self.completions = AsyncCompletions(self._transport)
        self.embeddings = AsyncEmbeddings(self._transport)

    @property
    def api_key(self) -> str | None:
        return self._transport.api_key

    @property
    def base_url(self) -> str:
        return self._transport.base_url

    async def close(self) -> None:
        """关闭底层连接。"""
        await self._transport.close()

    async def __aenter__(self) -> AsyncOpenAI:
        await self._transport.__aenter__()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self._transport.__aexit__(exc_type, exc, tb)
