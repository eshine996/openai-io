"""embeddings 资源（同步与异步）。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from .._base_client import AsyncAPIClient, SyncAPIClient
from .._types import NOT_GIVEN, NotGiven
from .._utils import normalize_iterable_input, remove_not_given
from ..types.embedding import CreateEmbeddingResponse

__all__ = ["AsyncEmbeddings", "EmbeddingInput", "Embeddings"]

#: embeddings 的 input 类型。
type EmbeddingInput = str | Iterable[str] | Iterable[int] | Iterable[Iterable[int]]


_BODY_FIELDS = ("model", "input", "encoding_format", "dimensions", "user")


def _build_create_body(params: Mapping[str, Any]) -> dict[str, Any]:
    body = {field: params[field] for field in _BODY_FIELDS}
    body["input"] = normalize_iterable_input(body["input"])
    return remove_not_given(body)


class Embeddings:
    """embeddings 同步资源。"""

    def __init__(self, client: SyncAPIClient) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        input: EmbeddingInput,
        encoding_format: Literal["float", "base64"] | NotGiven = NOT_GIVEN,
        dimensions: int | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
    ) -> CreateEmbeddingResponse:
        body = _build_create_body(locals())
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
        encoding_format: Literal["float", "base64"] | NotGiven = NOT_GIVEN,
        dimensions: int | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
    ) -> CreateEmbeddingResponse:
        body = _build_create_body(locals())
        response = await self._client.request("post", "/embeddings", json_body=body)
        return CreateEmbeddingResponse.model_validate_json(response.content)
