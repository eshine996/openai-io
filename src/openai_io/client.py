"""同步客户端 :class:`OpenAI`。

用法与 openai SDK 一致：:

    from openai_io import OpenAI
    from openai_io.messages import HumanMessage

    client = OpenAI(api_key="sk-...")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[HumanMessage(content="你好")],
    )
    print(resp.choices[0].message.content)
"""

from __future__ import annotations

from types import TracebackType

import httpx

from ._base_client import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, SyncAPIClient
from ._types import API_KEY_ENV_VAR, DEFAULT_BASE_URL, Headers, get_env
from .resources import Chat, Completions, Embeddings

__all__ = ["OpenAI"]


class OpenAI:
    """同步 OpenAI 客户端。"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | httpx.URL = DEFAULT_BASE_URL,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Headers | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        """构造同步客户端。

        Args:
            api_key: API key，缺省时读取环境变量 ``OPENAI_API_KEY``。
            organization: 可选的 OpenAI Organization ID。
            base_url: API 基础地址，默认 ``https://api.openai.com/v1``。
            timeout: 请求超时（秒），None 表示无超时。
            max_retries: 最大重试次数（不含首次请求）。
            default_headers: 附加到每个请求的默认请求头。
            http_client: 自定义 ``httpx.Client``（可用于测试注入 MockTransport）。
        """
        api_key = api_key if api_key is not None else get_env(API_KEY_ENV_VAR)
        self._transport = SyncAPIClient(
            api_key=api_key,
            organization=organization,
            base_url=str(base_url).rstrip("/"),
            timeout=httpx.Timeout(timeout) if timeout is not None else httpx.Timeout(None),
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
        )
        self.chat = Chat(self._transport)
        self.completions = Completions(self._transport)
        self.embeddings = Embeddings(self._transport)

    @property
    def api_key(self) -> str | None:
        return self._transport.api_key

    @property
    def base_url(self) -> str:
        return self._transport.base_url

    def close(self) -> None:
        """关闭底层连接。"""
        self._transport.close()

    def __enter__(self) -> OpenAI:
        self._transport.__enter__()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self._transport.__exit__(exc_type, exc, tb)
