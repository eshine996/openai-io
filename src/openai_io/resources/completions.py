"""completions（旧版文本补全）资源（同步与异步）。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, NotRequired, TypedDict, overload

from .._base_client import AsyncAPIClient, SyncAPIClient
from .._streaming import AsyncStream, Stream
from .._types import NOT_GIVEN, NotGiven
from .._utils import normalize_iterable_input, remove_not_given
from ..types.completion import Completion, CompletionChunk

__all__ = ["AsyncCompletions", "Completions", "PromptInput"]

#: completions 的 prompt 输入类型。
type PromptInput = str | Iterable[str] | Iterable[int] | Iterable[Iterable[int]] | None


class StreamOptions(TypedDict):
    include_usage: NotRequired[bool]
    include_obfuscation: NotRequired[bool]


_BODY_FIELDS = (
    "model",
    "prompt",
    "stream",
    "stream_options",
    "suffix",
    "max_tokens",
    "temperature",
    "top_p",
    "n",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "echo",
    "best_of",
    "seed",
    "logprobs",
)


def _build_create_body(params: Mapping[str, Any]) -> dict[str, Any]:
    body = {field: params[field] for field in _BODY_FIELDS}
    body["prompt"] = normalize_iterable_input(body["prompt"])
    return remove_not_given(body)


class Completions:
    """completions 同步资源。"""

    def __init__(self, client: SyncAPIClient) -> None:
        self._client = client

    @overload
    def create(
        self,
        *,
        model: str,
        prompt: PromptInput,
        stream: Literal[True],
        stream_options: StreamOptions | NotGiven | None = NOT_GIVEN,
        suffix: str | NotGiven = NOT_GIVEN,
        max_tokens: int | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        n: int | NotGiven = NOT_GIVEN,
        stop: str | list[str] | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        echo: bool | NotGiven = NOT_GIVEN,
        best_of: int | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        logprobs: int | NotGiven = NOT_GIVEN,
    ) -> Stream[CompletionChunk]: ...

    @overload
    def create(
        self,
        *,
        model: str,
        prompt: PromptInput,
        stream: Literal[False] | None = None,
        stream_options: StreamOptions | NotGiven | None = NOT_GIVEN,
        suffix: str | NotGiven = NOT_GIVEN,
        max_tokens: int | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        n: int | NotGiven = NOT_GIVEN,
        stop: str | list[str] | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        echo: bool | NotGiven = NOT_GIVEN,
        best_of: int | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        logprobs: int | NotGiven = NOT_GIVEN,
    ) -> Completion: ...

    def create(
        self,
        *,
        model: str,
        prompt: PromptInput,
        stream: bool | NotGiven | None = NOT_GIVEN,
        stream_options: StreamOptions | NotGiven | None = NOT_GIVEN,
        suffix: str | NotGiven = NOT_GIVEN,
        max_tokens: int | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        n: int | NotGiven = NOT_GIVEN,
        stop: str | list[str] | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        echo: bool | NotGiven = NOT_GIVEN,
        best_of: int | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        logprobs: int | NotGiven = NOT_GIVEN,
    ) -> Stream[CompletionChunk] | Completion:
        body = _build_create_body(locals())
        if stream is True:
            response = self._client.stream("post", "/completions", json_body=body)
            return Stream(cast_to=CompletionChunk, response=response)
        response = self._client.request("post", "/completions", json_body=body)
        return Completion.model_validate_json(response.content)


class AsyncCompletions:
    """completions 异步资源。"""

    def __init__(self, client: AsyncAPIClient) -> None:
        self._client = client

    @overload
    async def create(
        self,
        *,
        model: str,
        prompt: PromptInput,
        stream: Literal[True],
        stream_options: StreamOptions | NotGiven | None = NOT_GIVEN,
        suffix: str | NotGiven = NOT_GIVEN,
        max_tokens: int | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        n: int | NotGiven = NOT_GIVEN,
        stop: str | list[str] | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        echo: bool | NotGiven = NOT_GIVEN,
        best_of: int | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        logprobs: int | NotGiven = NOT_GIVEN,
    ) -> AsyncStream[CompletionChunk]: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        prompt: PromptInput,
        stream: Literal[False] | None = None,
        stream_options: StreamOptions | NotGiven | None = NOT_GIVEN,
        suffix: str | NotGiven = NOT_GIVEN,
        max_tokens: int | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        n: int | NotGiven = NOT_GIVEN,
        stop: str | list[str] | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        echo: bool | NotGiven = NOT_GIVEN,
        best_of: int | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        logprobs: int | NotGiven = NOT_GIVEN,
    ) -> Completion: ...

    async def create(
        self,
        *,
        model: str,
        prompt: PromptInput,
        stream: bool | NotGiven | None = NOT_GIVEN,
        stream_options: StreamOptions | NotGiven | None = NOT_GIVEN,
        suffix: str | NotGiven = NOT_GIVEN,
        max_tokens: int | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        top_p: float | NotGiven = NOT_GIVEN,
        n: int | NotGiven = NOT_GIVEN,
        stop: str | list[str] | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven = NOT_GIVEN,
        frequency_penalty: float | NotGiven = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        echo: bool | NotGiven = NOT_GIVEN,
        best_of: int | NotGiven = NOT_GIVEN,
        seed: int | NotGiven = NOT_GIVEN,
        logprobs: int | NotGiven = NOT_GIVEN,
    ) -> AsyncStream[CompletionChunk] | Completion:
        body = _build_create_body(locals())
        if stream is True:
            response = await self._client.stream("post", "/completions", json_body=body)
            return AsyncStream(cast_to=CompletionChunk, response=response)
        response = await self._client.request("post", "/completions", json_body=body)
        return Completion.model_validate_json(response.content)
