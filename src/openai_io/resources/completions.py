"""completions（旧版文本补全）资源（同步与异步）。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, overload

from .._base_client import AsyncAPIClient, SyncAPIClient
from .._streaming import AsyncStream, Stream
from .._types import NOT_GIVEN, NotGiven
from .._utils import remove_not_given
from ..types.completion import Completion, CompletionChunk

__all__ = ["AsyncCompletions", "Completions", "PromptInput"]

#: completions 的 prompt 输入类型。
type PromptInput = str | Iterable[str] | Iterable[int] | Iterable[Iterable[int]]


def _normalize_prompt(prompt: PromptInput) -> str | list[str] | list[int] | list[list[int]]:
    """把 prompt 转为可 JSON 序列化的形式。"""
    if isinstance(prompt, str):
        return prompt
    return list(prompt)  # type: ignore[return-value]


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
        body = remove_not_given(
            {
                "model": model,
                "prompt": _normalize_prompt(prompt),
                "stream": stream,
                "suffix": suffix,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": n,
                "stop": stop,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "logit_bias": logit_bias,
                "user": user,
                "echo": echo,
                "best_of": best_of,
                "seed": seed,
                "logprobs": logprobs,
            }
        )
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
        body = remove_not_given(
            {
                "model": model,
                "prompt": _normalize_prompt(prompt),
                "stream": stream,
                "suffix": suffix,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": n,
                "stop": stop,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "logit_bias": logit_bias,
                "user": user,
                "echo": echo,
                "best_of": best_of,
                "seed": seed,
                "logprobs": logprobs,
            }
        )
        if stream is True:
            response = await self._client.stream("post", "/completions", json_body=body)
            return AsyncStream(cast_to=CompletionChunk, response=response)
        response = await self._client.request("post", "/completions", json_body=body)
        return Completion.model_validate_json(response.content)
