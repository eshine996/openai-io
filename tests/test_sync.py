"""同步客户端的资源测试（MockTransport 注入）。"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from openai_io import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from openai_io.messages import HumanMessage
from tests.conftest import TEST_API_KEY, TEST_BASE_URL, Handler

CHAT_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "你好！"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
}


def test_chat_completion_create(sync_client: Callable[[Handler], OpenAI]) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = sync_client(handler)
    response = client.chat.completions.create(model="gpt-4o-mini", messages=[HumanMessage(content="你好")])

    assert response.choices[0].message.content == "你好！"
    assert response.usage is not None
    assert response.usage.total_tokens == 9
    assert captured["url"] == f"{TEST_BASE_URL}/chat/completions"
    assert captured["auth"] == f"Bearer {TEST_API_KEY}"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"] == [{"role": "user", "content": "你好"}]
    # 未显式传参的字段（NotGiven）不应出现在请求体中
    assert "temperature" not in body
    assert "stream" not in body


def test_chat_completion_with_optional_params(sync_client: Callable[[Handler], OpenAI]) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = sync_client(handler)
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[HumanMessage(content="hi")],
        temperature=0.7,
        max_tokens=100,
        stop=["\n"],
        user="u-1",
        stream=False,
    )
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 100
    assert body["stop"] == ["\n"]
    assert body["user"] == "u-1"
    assert body["stream"] is False


def test_chat_completion_stream(sync_client: Callable[[Handler], OpenAI]) -> None:
    chunk1 = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "你"}, "finish_reason": None}],
    }
    chunk2 = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"content": "好"}, "finish_reason": "stop"}],
    }
    sse = "data: " + json.dumps(chunk1) + "\n\ndata: " + json.dumps(chunk2) + "\n\ndata: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    client = sync_client(handler)
    stream = client.chat.completions.create(model="gpt-4o-mini", messages=[HumanMessage(content="hi")], stream=True)

    chunks = list(stream)
    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "你"
    assert chunks[1].choices[0].delta.content == "好"
    assert chunks[1].choices[0].finish_reason == "stop"


def test_completions_create(sync_client: Callable[[Handler], OpenAI]) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 1700000000,
                "model": "text-davinci-003",
                "choices": [{"text": "42", "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    client = sync_client(handler)
    response = client.completions.create(model="text-davinci-003", prompt="1+1=?")

    assert response.choices[0].text == "42"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["prompt"] == "1+1=?"
    assert body["model"] == "text-davinci-003"


def test_completions_prompt_generator_normalized(sync_client: Callable[[Handler], OpenAI]) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 1700000000,
                "model": "m",
                "choices": [{"text": "x", "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = sync_client(handler)
    client.completions.create(model="m", prompt=iter([1, 2, 3]))
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["prompt"] == [1, 2, 3]


def test_embeddings_create(sync_client: Callable[[Handler], OpenAI]) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 8, "total_tokens": 8},
            },
        )

    client = sync_client(handler)
    response = client.embeddings.create(model="text-embedding-3-small", input="你好")

    assert response.data[0].embedding == [0.1, 0.2, 0.3]
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["input"] == "你好"
    assert body["model"] == "text-embedding-3-small"


def test_embeddings_input_generator_normalized(sync_client: Callable[[Handler], OpenAI]) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            },
        )

    client = sync_client(handler)
    client.embeddings.create(model="text-embedding-3-small", input=iter(["a", "b"]))
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["input"] == ["a", "b"]


def test_error_status_mapping(sync_client: Callable[[Handler], OpenAI]) -> None:
    cases: list[tuple[int, type[Exception]]] = [
        (400, BadRequestError),
        (401, AuthenticationError),
        (403, PermissionDeniedError),
        (404, NotFoundError),
        (409, ConflictError),
        (422, UnprocessableEntityError),
        (429, RateLimitError),
        (500, InternalServerError),
    ]
    for status, expected in cases:

        def handler(request: httpx.Request, *, status_code: int = status) -> httpx.Response:
            return httpx.Response(status_code, json={"error": {"message": "出错了", "type": "test_error"}})

        client = sync_client(handler)
        with pytest.raises(expected):
            client.chat.completions.create(model="m", messages=[HumanMessage(content="hi")])


def test_retry_on_429_then_success(sync_client: Callable[[Handler], OpenAI]) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = sync_client(handler)
    response = client.chat.completions.create(model="m", messages=[HumanMessage(content="hi")])
    assert response.id == "chatcmpl-1"
    assert calls == 2


def test_retry_on_connection_error_then_success(sync_client: Callable[[Handler], OpenAI]) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("连接失败")
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = sync_client(handler)
    response = client.chat.completions.create(model="m", messages=[HumanMessage(content="hi")])
    assert response.id == "chatcmpl-1"
    assert calls == 2


def test_connection_error_raised_after_retries(sync_client: Callable[[Handler], OpenAI]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("连接失败")

    client = sync_client(handler)
    with pytest.raises(APIConnectionError) as exc_info:
        client.chat.completions.create(model="m", messages=[HumanMessage(content="hi")])
    assert isinstance(exc_info.value.cause, httpx.ConnectError)


def test_stream_closed_after_full_iteration(sync_client: Callable[[Handler], OpenAI]) -> None:
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
    }
    sse = "data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    client = sync_client(handler)
    stream = client.chat.completions.create(model="m", messages=[HumanMessage(content="hi")], stream=True)
    assert list(stream)
    assert stream.response.is_closed


def test_stream_closed_on_early_exit(sync_client: Callable[[Handler], OpenAI]) -> None:
    chunk1 = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": None}],
    }
    chunk2 = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"content": "b"}, "finish_reason": "stop"}],
    }
    sse = "data: " + json.dumps(chunk1) + "\n\ndata: " + json.dumps(chunk2) + "\n\ndata: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    client = sync_client(handler)
    stream = client.chat.completions.create(model="m", messages=[HumanMessage(content="hi")], stream=True)
    with stream:
        for chunk in stream:
            assert chunk.choices[0].delta.content == "a"
            break  # 提前退出
    assert stream.response.is_closed


def test_stream_ignores_invalid_json_line(sync_client: Callable[[Handler], OpenAI]) -> None:
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
    }
    sse = "data: {坏掉的 json}\n\ndata: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    client = sync_client(handler)
    stream = client.chat.completions.create(model="m", messages=[HumanMessage(content="hi")], stream=True)
    chunks = list(stream)
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "hi"


def test_chat_completion_stream_none_sends_null(sync_client: Callable[[Handler], OpenAI]) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = sync_client(handler)
    client.chat.completions.create(model="m", messages=[HumanMessage(content="hi")], stream=None)
    body = captured["body"]
    assert isinstance(body, dict)
    # 显式传 None 遵循 openai 语义：发送 null（与"不传"的 NotGiven 不同）
    assert body["stream"] is None


def test_error_message_extracted(sync_client: Callable[[Handler], OpenAI]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "模型不存在", "type": "invalid_request_error"}})

    client = sync_client(handler)
    with pytest.raises(NotFoundError) as exc_info:
        client.chat.completions.create(model="m", messages=[HumanMessage(content="hi")])
    assert "模型不存在" in str(exc_info.value)
    assert exc_info.value.status_code == 404


def test_context_manager(sync_client: Callable[[Handler], OpenAI]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CHAT_RESPONSE)

    client = sync_client(handler)
    with client:
        response = client.chat.completions.create(model="m", messages=[HumanMessage(content="hi")])
        assert response.id == "chatcmpl-1"
