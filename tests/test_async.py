"""异步客户端的资源测试（MockTransport 注入）。"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from openai_io import AsyncOpenAI, NotFoundError
from openai_io.messages import HumanMessage
from tests.conftest import Handler

CHAT_RESPONSE = {
    "id": "chatcmpl-a1",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "异步你好！"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
}


async def test_async_chat_completion_create(async_client: Callable[[Handler], AsyncOpenAI]) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = async_client(handler)
    response = await client.chat.completions.create(model="gpt-4o-mini", messages=[HumanMessage(content="你好")])

    assert response.choices[0].message.content == "异步你好！"
    assert response.usage is not None
    assert response.usage.total_tokens == 9
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["messages"] == [{"role": "user", "content": "你好"}]


async def test_async_chat_completion_stream(async_client: Callable[[Handler], AsyncOpenAI]) -> None:
    chunk1 = {
        "id": "chatcmpl-a1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "流"}, "finish_reason": None}],
    }
    chunk2 = {
        "id": "chatcmpl-a1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"content": "式"}, "finish_reason": "stop"}],
    }
    sse = "data: " + json.dumps(chunk1) + "\n\ndata: " + json.dumps(chunk2) + "\n\ndata: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    client = async_client(handler)
    stream = await client.chat.completions.create(
        model="gpt-4o-mini", messages=[HumanMessage(content="hi")], stream=True
    )

    chunks: list[object] = []
    async for chunk in stream:
        chunks.append(chunk)
    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "流"  # type: ignore[union-attr]
    assert chunks[1].choices[0].delta.content == "式"  # type: ignore[union-attr]


async def test_async_embeddings_create(async_client: Callable[[Handler], AsyncOpenAI]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.9, 0.8]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    client = async_client(handler)
    response = await client.embeddings.create(model="text-embedding-3-small", input="你好")
    assert response.data[0].embedding == [0.9, 0.8]


async def test_async_error_mapping(async_client: Callable[[Handler], AsyncOpenAI]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "模型不存在"}})

    client = async_client(handler)
    with pytest.raises(NotFoundError):
        await client.chat.completions.create(model="m", messages=[HumanMessage(content="hi")])


async def test_async_context_manager(async_client: Callable[[Handler], AsyncOpenAI]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = async_client(handler)
    async with client:
        response = await client.chat.completions.create(model="m", messages=[HumanMessage(content="hi")])
        assert response.id == "chatcmpl-a1"
