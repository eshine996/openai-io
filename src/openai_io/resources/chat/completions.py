"""Chat Completions 同步与异步资源。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, NotRequired, TypedDict, overload

from ..._base_client import AsyncAPIClient, SyncAPIClient
from ..._streaming import AsyncStream, Stream
from ..._types import NOT_GIVEN, NotGiven
from ..._utils import remove_not_given
from ...messages import MessageLike, to_openai_messages
from ...types.chat import ChatCompletion, ChatCompletionChunk

__all__ = ["AsyncChatCompletions", "ChatCompletions"]


class StreamOptions(TypedDict):
    include_usage: NotRequired[bool]
    include_obfuscation: NotRequired[bool]


type ToolChoice = Literal["none", "auto", "required"] | dict[str, Any]
type FunctionCallOption = Literal["none", "auto"] | dict[str, Any]
type ServiceTier = Literal["auto", "default", "flex", "scale", "priority", "fast"]
type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

_BODY_FIELDS = (
    "model",
    "messages",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "n",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "user",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "function_call",
    "functions",
    "response_format",
    "seed",
    "metadata",
    "service_tier",
    "reasoning_effort",
)


def _build_create_body(params: Mapping[str, Any]) -> dict[str, Any]:
    body = {field: params[field] for field in _BODY_FIELDS}
    body["messages"] = to_openai_messages(body["messages"])
    for field in ("tools", "functions"):
        value = body[field]
        if not isinstance(value, NotGiven):
            body[field] = list(value)
    return remove_not_given(body)


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
        stream_options: StreamOptions | NotGiven | None = NOT_GIVEN,
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
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        function_call: FunctionCallOption | NotGiven = NOT_GIVEN,
        functions: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        response_format: dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        metadata: dict[str, str] | NotGiven | None = NOT_GIVEN,
        service_tier: ServiceTier | NotGiven | None = NOT_GIVEN,
        reasoning_effort: ReasoningEffort | NotGiven | None = NOT_GIVEN,
    ) -> Stream[ChatCompletionChunk]: ...

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Iterable[MessageLike],
        stream: Literal[False] | None = None,
        stream_options: StreamOptions | NotGiven | None = NOT_GIVEN,
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
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        function_call: FunctionCallOption | NotGiven = NOT_GIVEN,
        functions: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        response_format: dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        metadata: dict[str, str] | NotGiven | None = NOT_GIVEN,
        service_tier: ServiceTier | NotGiven | None = NOT_GIVEN,
        reasoning_effort: ReasoningEffort | NotGiven | None = NOT_GIVEN,
    ) -> ChatCompletion: ...

    def create(
        self,
        *,
        model: str,
        messages: Iterable[MessageLike],
        stream: bool | NotGiven | None = NOT_GIVEN,
        stream_options: StreamOptions | NotGiven | None = NOT_GIVEN,
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
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        function_call: FunctionCallOption | NotGiven = NOT_GIVEN,
        functions: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        response_format: dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        metadata: dict[str, str] | NotGiven | None = NOT_GIVEN,
        service_tier: ServiceTier | NotGiven | None = NOT_GIVEN,
        reasoning_effort: ReasoningEffort | NotGiven | None = NOT_GIVEN,
    ) -> Stream[ChatCompletionChunk] | ChatCompletion:
        body = _build_create_body(locals())
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
        stream_options: StreamOptions | NotGiven | None = NOT_GIVEN,
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
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        function_call: FunctionCallOption | NotGiven = NOT_GIVEN,
        functions: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        response_format: dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        metadata: dict[str, str] | NotGiven | None = NOT_GIVEN,
        service_tier: ServiceTier | NotGiven | None = NOT_GIVEN,
        reasoning_effort: ReasoningEffort | NotGiven | None = NOT_GIVEN,
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Iterable[MessageLike],
        stream: Literal[False] | None = None,
        stream_options: StreamOptions | NotGiven | None = NOT_GIVEN,
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
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        function_call: FunctionCallOption | NotGiven = NOT_GIVEN,
        functions: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        response_format: dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        metadata: dict[str, str] | NotGiven | None = NOT_GIVEN,
        service_tier: ServiceTier | NotGiven | None = NOT_GIVEN,
        reasoning_effort: ReasoningEffort | NotGiven | None = NOT_GIVEN,
    ) -> ChatCompletion: ...

    async def create(
        self,
        *,
        model: str,
        messages: Iterable[MessageLike],
        stream: bool | NotGiven | None = NOT_GIVEN,
        stream_options: StreamOptions | NotGiven | None = NOT_GIVEN,
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
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        function_call: FunctionCallOption | NotGiven = NOT_GIVEN,
        functions: Iterable[dict[str, Any]] | NotGiven = NOT_GIVEN,
        response_format: dict[str, Any] | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        metadata: dict[str, str] | NotGiven | None = NOT_GIVEN,
        service_tier: ServiceTier | NotGiven | None = NOT_GIVEN,
        reasoning_effort: ReasoningEffort | NotGiven | None = NOT_GIVEN,
    ) -> AsyncStream[ChatCompletionChunk] | ChatCompletion:
        body = _build_create_body(locals())
        if stream is True:
            response = await self._client.stream("post", "/chat/completions", json_body=body)
            return AsyncStream(cast_to=ChatCompletionChunk, response=response)
        response = await self._client.request("post", "/chat/completions", json_body=body)
        return ChatCompletion.model_validate_json(response.content)
