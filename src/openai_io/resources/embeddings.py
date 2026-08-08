"""embeddings 资源（同步与异步）。"""

from __future__ import annotations

from collections.abc import Iterable

from .._base_client import AsyncAPIClient, SyncAPIClient
from .._types import NOT_GIVEN, NotGiven
from .._utils import remove_not_given
from ..types.embedding import CreateEmbeddingResponse

__all__ = ["AsyncEmbeddings", "EmbeddingInput", "Embeddings"]

#: embeddings 的 input 类型。
type EmbeddingInput = str | Iterable[str] | Iterable[int] | Iterable[Iterable[int]]


def _normalize_input(value: EmbeddingInput) -> str | list[str] | list[int] | list[list[int]]:
    """把 input 转为可 JSON 序列化的形式。"""
    if isinstance(value, str):
        return value
    return list(value)  # type: ignore[return-value]


class Embeddings:
    """embeddings 同步资源。"""

    def __init__(self, client: SyncAPIClient) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        input: EmbeddingInput,
        encoding_format: str | NotGiven = NOT_GIVEN,
        dimensions: int | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
    ) -> CreateEmbeddingResponse:
        body = remove_not_given(
            {
                "model": model,
                "input": _normalize_input(input),
                "encoding_format": encoding_format,
                "dimensions": dimensions,
                "user": user,
            }
        )
        response = self._client.request("post", "/embeddings", json_body=body)
        return CreateEmbeddingResponse.model_validate_json(response.content)


class AsyncEmbeddings:
    """embeddings 异步资源。"""

    def __init__(self, client: AsyncAPIClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        input: EmbeddingInput,
        encoding_format: str | NotGiven = NOT_GIVEN,
        dimensions: int | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
    ) -> CreateEmbeddingResponse:
        body = remove_not_given(
            {
                "model": model,
                "input": _normalize_input(input),
                "encoding_format": encoding_format,
                "dimensions": dimensions,
                "user": user,
            }
        )
        response = await self._client.request("post", "/embeddings", json_body=body)
        return CreateEmbeddingResponse.model_validate_json(response.content)
