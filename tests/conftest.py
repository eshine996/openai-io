"""测试共用 fixture：基于 httpx.MockTransport 注入 mock 响应。"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from openai_io import AsyncOpenAI, OpenAI

#: mock handler：接收请求返回响应。
type Handler = Callable[[httpx.Request], httpx.Response]

TEST_BASE_URL = "https://api.test.local/v1"
TEST_API_KEY = "test-key"


@pytest.fixture
def sync_client() -> Callable[[Handler], OpenAI]:
    """构造注入 MockTransport 的同步客户端。"""

    def build(handler: Handler) -> OpenAI:
        transport = httpx.MockTransport(handler)
        return OpenAI(
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            http_client=httpx.Client(transport=transport),
        )

    return build


@pytest.fixture
def async_client() -> Callable[[Handler], AsyncOpenAI]:
    """构造注入 MockTransport 的异步客户端。"""

    def build(handler: Handler) -> AsyncOpenAI:
        transport = httpx.MockTransport(handler)
        return AsyncOpenAI(
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            http_client=httpx.AsyncClient(transport=transport),
        )

    return build
