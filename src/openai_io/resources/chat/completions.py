"""chat.completions 资源（同步与异步）。

``create`` 的入口参数与 openai SDK 的 ``client.chat.completions.create`` 对齐，
仅 ``messages`` 改为 langchain 风格（见 :mod:`openai_io.messages`）。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, overload

from ..._base_client import AsyncAPIClient, SyncAPIClient
from ..._streaming import AsyncStream, Stream
from ..._types import NOT_GIVEN, NotGiven
from ..._utils import remove_not_given
from ...messages import MessageLike, to_openai_messages
from ...types.chat import ChatCompletion, ChatCompletionChunk

__all__ = ["AsyncChatCompletions", "ChatCompletions"]


class ChatCompletions:
    """chat.completions 同步资源。"""

    def __init__(self, client: SyncAPIClient) -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Iterable[MessageLike],
        stream: Literal[True],
        stream_options: dict[str, Any] | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        n: int | NotGiven = NOT_GIVEN,
        stop: str | list[str] | NotGiven = NOT_GIVEN,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven = NOT_GIVEN,
        logprobs: bool | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        tools: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        tool_choice: str | dict[str, Any] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        function_call: str | dict[str, Any] | NotGiven = NOT_GIVEN,
        functions: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        response_format: dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        metadata: dict[str, Any] | NotGiven = NOT_GIVEN,
        service_tier: str | NotGiven = NOT_GIVEN,
        reasoning_effort: str | NotGiven = NOT_GIVEN,
    ) -> Stream[ChatCompletionChunk]: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Iterable[MessageLike],
        stream: Literal[False] | None = None,
        stream_options: dict[str, Any] | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        n: int | NotGiven = NOT_GIVEN,
        stop: str | list[str] | NotGiven = NOT_GIVEN,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven = NOT_GIVEN,
        logprobs: bool | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        tools: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        tool_choice: str | dict[str, Any] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        function_call: str | dict[str, Any] | NotGiven = NOT_GIVEN,
        functions: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        response_format: dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        metadata: dict[str, Any] | NotGiven = NOT_GIVEN,
        service_tier: str | NotGiven = NOT_GIVEN,
        reasoning_effort: str | NotGiven = NOT_GIVEN,
    ) -> ChatCompletion: ...

    def create(
        self,
        *,
        model: str,
        messages: Iterable[MessageLike],
        stream: bool | NotGiven | None = NOT_GIVEN,
        stream_options: dict[str, Any] | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        n: int | NotGiven = NOT_GIVEN,
        stop: str | list[str] | NotGiven = NOT_GIVEN,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven = NOT_GIVEN,
        logprobs: bool | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        tools: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        tool_choice: str | dict[str, Any] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        function_call: str | dict[str, Any] | NotGiven = NOT_GIVEN,
        functions: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        response_format: dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        metadata: dict[str, Any] | NotGiven = NOT_GIVEN,
        service_tier: str | NotGiven = NOT_GIVEN,
        reasoning_effort: str | NotGiven = NOT_GIVEN,
    ) -> Stream[ChatCompletionChunk] | ChatCompletion:
        body = remove_not_given(
            {
                "model": model,
                "messages": to_openai_messages(messages),
                "stream": stream,
                "stream_options": stream_options,
                "temperature": temperature,
                "top_p": top_p,
                "n": n,
                "stop": stop,
                "max_tokens": max_tokens,
                "max_completion_tokens": max_completion_tokens,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "logit_bias": logit_bias,
                "logprobs": logprobs,
                "top_logprobs": top_logprobs,
                "user": user,
                "tools": list(tools) if not isinstance(tools, NotGiven) else tools,
                "tool_choice": tool_choice,
                "parallel_tool_calls": parallel_tool_calls,
                "function_call": function_call,
                "functions": list(functions) if not isinstance(functions, NotGiven) else functions,
                "response_format": response_format,
                "seed": seed,
                "metadata": metadata,
                "service_tier": service_tier,
                "reasoning_effort": reasoning_effort,
            }
        )
        if stream is True:
            response = self._client.stream("post", "/chat/completions", json_body=body)
            return Stream(cast_to=ChatCompletionChunk, response=response)
        response = self._client.request("post", "/chat/completions", json_body=body)
        return ChatCompletion.model_validate_json(response.content)


class AsyncChatCompletions:
    """chat.completions 异步资源。"""

    def __init__(self, client: AsyncAPIClient) -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Iterable[MessageLike],
        stream: Literal[True],
        stream_options: dict[str, Any] | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        n: int | NotGiven = NOT_GIVEN,
        stop: str | list[str] | NotGiven = NOT_GIVEN,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven = NOT_GIVEN,
        logprobs: bool | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        tools: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        tool_choice: str | dict[str, Any] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        function_call: str | dict[str, Any] | NotGiven = NOT_GIVEN,
        functions: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        response_format: dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        metadata: dict[str, Any] | NotGiven = NOT_GIVEN,
        service_tier: str | NotGiven = NOT_GIVEN,
        reasoning_effort: str | NotGiven = NOT_GIVEN,
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Iterable[MessageLike],
        stream: Literal[False] | None = None,
        stream_options: dict[str, Any] | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        n: int | NotGiven = NOT_GIVEN,
        stop: str | list[str] | NotGiven = NOT_GIVEN,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven = NOT_GIVEN,
        logprobs: bool | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        tools: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        tool_choice: str | dict[str, Any] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        function_call: str | dict[str, Any] | NotGiven = NOT_GIVEN,
        functions: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        response_format: dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        metadata: dict[str, Any] | NotGiven = NOT_GIVEN,
        service_tier: str | NotGiven = NOT_GIVEN,
        reasoning_effort: str | NotGiven = NOT_GIVEN,
    ) -> ChatCompletion: ...

    async def create(
        self,
        *,
        model: str,
        messages: Iterable[MessageLike],
        stream: bool | NotGiven | None = NOT_GIVEN,
        stream_options: dict[str, Any] | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        n: int | NotGiven = NOT_GIVEN,
        stop: str | list[str] | NotGiven = NOT_GIVEN,
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven = NOT_GIVEN,
        logprobs: bool | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        tools: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        tool_choice: str | dict[str, Any] | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        function_call: str | dict[str, Any] | NotGiven = NOT_GIVEN,
        functions: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        response_format: dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        metadata: dict[str, Any] | NotGiven = NOT_GIVEN,
        service_tier: str | NotGiven = NOT_GIVEN,
        reasoning_effort: str | NotGiven = NOT_GIVEN,
    ) -> AsyncStream[ChatCompletionChunk] | ChatCompletion:
        body = remove_not_given(
            {
                "model": model,
                "messages": to_openai_messages(messages),
                "stream": stream,
                "stream_options": stream_options,
                "temperature": temperature,
                "top_p": top_p,
                "n": n,
                "stop": stop,
                "max_tokens": max_tokens,
                "max_completion_tokens": max_completion_tokens,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "logit_bias": logit_bias,
                "logprobs": logprobs,
                "top_logprobs": top_logprobs,
                "user": user,
                "tools": list(tools) if not isinstance(tools, NotGiven) else tools,
                "tool_choice": tool_choice,
                "parallel_tool_calls": parallel_tool_calls,
                "function_call": function_call,
                "functions": list(functions) if not isinstance(functions, NotGiven) else functions,
                "response_format": response_format,
                "seed": seed,
                "metadata": metadata,
                "service_tier": service_tier,
                "reasoning_effort": reasoning_effort,
            }
        )
        if stream is True:
            response = await self._client.stream("post", "/chat/completions", json_body=body)
            return AsyncStream(cast_to=ChatCompletionChunk, response=response)
        response = await self._client.request("post", "/chat/completions", json_body=body)
        return ChatCompletion.model_validate_json(response.content)
